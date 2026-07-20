"""Training loop: AMP, checkpoint/resume, ETA print (R4, R5, R7, Section 4).

Trains one fold of the plain 4-level U-Net from random init (no BraTS weights,
R4) with a soft-Dice + BCE loss (R5). All slices are kept; class imbalance is
handled by the loss plus an optional foreground-biased batch sampler (R5). Every
epoch checkpoints to the working dir and ``--resume`` continues after a Kaggle
session cap (R7). After epoch 1 it prints a measured ETA for the whole run.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from .data.dataset import SliceDataset, subset_for_patients
from .metrics import dice_bce_loss, slice_dice
from .models.unet import build_unet, count_params
from .utils import set_seed


@dataclass
class TrainConfig:
    epochs: int = 60
    batch_size: int = 16
    lr: float = 1e-3
    in_channels: int = 3
    channels: str = "reference"
    batchnorm: bool = True
    augment: bool = True
    amp: bool = True
    num_workers: int = 2
    foreground_bias: float = 0.0  # 0 = natural sampling; >0 up-weights tumor slices
    dice_weight: float = 1.0
    bce_weight: float = 1.0
    seed: int = 42
    grad_clip: float = 0.0


def _make_loader(df: pd.DataFrame, cfg: TrainConfig, train: bool) -> DataLoader:
    ds = SliceDataset(df, in_channels=cfg.in_channels, augment=(train and cfg.augment))
    if train and cfg.foreground_bias > 0:
        # Up-weight tumor slices without discarding empties (R5).
        w = np.where(df["has_tumor"].values, 1.0 + cfg.foreground_bias, 1.0)
        sampler = WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), len(w), replacement=True)
        return DataLoader(ds, batch_size=cfg.batch_size, sampler=sampler,
                          num_workers=cfg.num_workers, pin_memory=True, drop_last=False)
    return DataLoader(ds, batch_size=cfg.batch_size, shuffle=train,
                      num_workers=cfg.num_workers, pin_memory=True, drop_last=False)


def _val_mean_slice_dice(model, loader, device) -> float:
    model.eval()
    scores = []
    with torch.no_grad():
        for imgs, masks, _ in loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            pred = (torch.sigmoid(logits) > 0.5).cpu().numpy().astype(np.uint8)
            gt = masks.numpy().astype(np.uint8)
            for p, g in zip(pred, gt):
                scores.append(slice_dice(p[0], g[0]))
    return float(np.mean(scores)) if scores else 0.0


def train_fold(
    manifest: pd.DataFrame,
    train_patients: List[str],
    val_patients: List[str],
    cfg: TrainConfig,
    ckpt_path: str,
    device=None,
    resume: bool = False,
    log_path: Optional[str] = None,
) -> dict:
    """Train one fold; returns a small history dict. Checkpoints every epoch."""
    import torch

    set_seed(cfg.seed)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_df = subset_for_patients(manifest, train_patients)
    val_df = subset_for_patients(manifest, val_patients)
    train_loader = _make_loader(train_df, cfg, train=True)
    val_loader = _make_loader(val_df, cfg, train=False)

    model = build_unet(cfg.channels, in_channels=cfg.in_channels, batchnorm=cfg.batchnorm).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    use_amp = cfg.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    start_epoch = 0
    history = {"train_loss": [], "val_dice": [], "epoch_seconds": []}
    best_dice = -1.0

    os.makedirs(os.path.dirname(os.path.abspath(ckpt_path)), exist_ok=True)
    if resume and os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        sched.load_state_dict(state["sched"])
        scaler.load_state_dict(state["scaler"])
        start_epoch = state["epoch"] + 1
        history = state.get("history", history)
        best_dice = state.get("best_dice", -1.0)
        print(f"[resume] continuing from epoch {start_epoch} (best val dice {best_dice:.4f})")

    print(f"[train] params={count_params(model)/1e6:.2f}M  train_slices={len(train_df)}  "
          f"val_slices={len(val_df)}  device={device.type}  amp={scaler.is_enabled()}")

    run_start = time.time()
    for epoch in range(start_epoch, cfg.epochs):
        model.train()
        t0 = time.time()
        running = 0.0
        n = 0
        for imgs, masks, _ in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(imgs)
                loss = dice_bce_loss(logits, masks, cfg.dice_weight, cfg.bce_weight)
            scaler.scale(loss).backward()
            if cfg.grad_clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(opt)
            scaler.update()
            running += loss.item() * imgs.size(0)
            n += imgs.size(0)
        sched.step()

        val_dice = _val_mean_slice_dice(model, val_loader, device)
        dt = time.time() - t0
        train_loss = running / max(n, 1)
        history["train_loss"].append(train_loss)
        history["val_dice"].append(val_dice)
        history["epoch_seconds"].append(dt)

        is_best = val_dice > best_dice
        best_dice = max(best_dice, val_dice)
        state = {
            "epoch": epoch, "model": model.state_dict(), "opt": opt.state_dict(),
            "sched": sched.state_dict(), "scaler": scaler.state_dict(),
            "history": history, "best_dice": best_dice, "cfg": cfg.__dict__,
        }
        torch.save(state, ckpt_path)
        if is_best:
            torch.save({"model": model.state_dict(), "cfg": cfg.__dict__,
                        "epoch": epoch, "val_dice": val_dice}, ckpt_path.replace(".pt", "_best.pt"))

        msg = (f"[epoch {epoch+1}/{cfg.epochs}] loss={train_loss:.4f} "
               f"val_slice_dice={val_dice:.4f} best={best_dice:.4f} {dt:.1f}s")
        # Measured ETA after the first completed epoch (R7 / Section 4).
        if epoch == start_epoch:
            remaining = (cfg.epochs - epoch - 1) * dt
            msg += f"  | ETA full run ~{remaining/60:.1f} min ({remaining/3600:.2f} h)"
        print(msg, flush=True)
        if log_path:
            with open(log_path, "a") as fh:
                fh.write(msg + "\n")

    history["total_seconds"] = time.time() - run_start
    history["best_val_slice_dice"] = best_dice
    return history
