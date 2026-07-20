"""Knowledge distillation to a reduced-channel student U-Net (Section 6).

Trains a small ("micro") student against a trained teacher's soft outputs plus
the ground-truth Dice+BCE loss, then reports the size/accuracy tradeoff. Uses the
same patient-level splits as everything else — the student is evaluated with the
per-patient headline metric so its Dice is comparable to the teacher's (R2).
"""

from __future__ import annotations

import time
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data.dataset import SliceDataset, subset_for_patients
from .metrics import dice_bce_loss
from .models.unet import build_unet, count_params
from .utils import set_seed


def distill_student(
    manifest: pd.DataFrame,
    train_patients: List[str],
    teacher_ckpt: str,
    student_channels: str,
    ckpt_path: str,
    device,
    in_channels: int = 3,
    teacher_channels: str = "reference",
    batchnorm: bool = True,
    epochs: int = 30,
    lr: float = 1e-3,
    batch_size: int = 16,
    temperature: float = 2.0,
    alpha: float = 0.5,
    seed: int = 42,
) -> dict:
    """Train ``student_channels`` U-Net from the teacher at ``teacher_ckpt``.

    Loss = alpha * KD(student, teacher soft targets) + (1-alpha) * Dice+BCE(gt).
    KD term is tumour-aware: it uses the soft-target BCE on teacher probabilities,
    which keeps the student honest on foreground rather than collapsing to
    background (a documented failure mode of naive KD on imbalanced masks).
    """
    set_seed(seed)
    teacher = build_unet(teacher_channels, in_channels=in_channels, batchnorm=batchnorm).to(device)
    teacher.load_state_dict(torch.load(teacher_ckpt, map_location=device)["model"])
    teacher.eval()

    student = build_unet(student_channels, in_channels=in_channels, batchnorm=batchnorm).to(device)
    opt = torch.optim.Adam(student.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    df = subset_for_patients(manifest, train_patients)
    ds = SliceDataset(df, in_channels=in_channels, augment=True)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

    history = {"loss": []}
    t0 = time.time()
    for epoch in range(epochs):
        student.train()
        running, n = 0.0, 0
        for imgs, masks, _ in loader:
            imgs, masks = imgs.to(device), masks.to(device)
            with torch.no_grad():
                soft = torch.sigmoid(teacher(imgs) / temperature)
            logits = student(imgs)
            kd = F.binary_cross_entropy_with_logits(logits / temperature, soft) * (temperature ** 2)
            hard = dice_bce_loss(logits, masks)
            loss = alpha * kd + (1 - alpha) * hard
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            running += loss.item() * imgs.size(0)
            n += imgs.size(0)
        sched.step()
        history["loss"].append(running / max(n, 1))
        print(f"[distill epoch {epoch+1}/{epochs}] loss={history['loss'][-1]:.4f}", flush=True)

    torch.save({"model": student.state_dict(), "channels": student_channels,
                "epoch": epochs - 1}, ckpt_path)
    history["total_seconds"] = time.time() - t0
    history["student_params_M"] = count_params(student) / 1e6
    return history
