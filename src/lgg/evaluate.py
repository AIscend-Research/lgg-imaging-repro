"""Per-patient (volume-aggregated) evaluation — the headline metric (R2, R3).

For each fold we load the trained checkpoint, run inference over that fold's
validation patients, regroup the slices back into per-patient volumes, and
compute volume-aggregated Dice / IoU / HD95 per patient. Because the folds
partition the roster, pooling all folds gives exactly one score per patient,
from which we report mean / median (the numbers compared to 0.82 / 0.85).

Per-slice Dice is computed alongside but returned under a clearly separate key
so it is never mistaken for the headline (R2).
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .data.dataset import SliceDataset, subset_for_patients
from .metrics import (
    aggregate_patient_metrics,
    hd95_volume,
    patient_volume_dice,
    patient_volume_iou,
    slice_dice,
)
from .models.unet import build_unet


def _load_model(ckpt_path, in_channels, channels, batchnorm, device):
    state = torch.load(ckpt_path, map_location=device)
    model = build_unet(channels, in_channels=in_channels, batchnorm=batchnorm).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    return model


@torch.no_grad()
def predict_patients(
    model,
    manifest: pd.DataFrame,
    patients: List[str],
    device,
    in_channels: int = 3,
    batch_size: int = 16,
    threshold: float = 0.5,
    perturb: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> Dict[str, dict]:
    """Return {patient: {'preds':[...], 'gts':[...], 'slice_dice':[...]}}.

    ``perturb`` optionally transforms each (C,H,W) input array before inference
    (used by the robustness module) — inference-only, applied after normalization.
    """
    df = subset_for_patients(manifest, patients)
    ds = SliceDataset(df, in_channels=in_channels, augment=False)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    grouped: Dict[str, dict] = {p: {"preds": [], "gts": [], "slice_dice": []} for p in patients}
    for imgs, masks, idxs in loader:
        if perturb is not None:
            arr = imgs.numpy()
            arr = np.stack([perturb(a) for a in arr], axis=0)
            imgs = torch.from_numpy(arr).float()
        imgs = imgs.to(device)
        logits = model(imgs)
        pred = (torch.sigmoid(logits) > threshold).cpu().numpy().astype(np.uint8)
        gt = masks.numpy().astype(np.uint8)
        for k in range(pred.shape[0]):
            patient = df.iloc[int(idxs[k])]["patient"]
            p, g = pred[k, 0], gt[k, 0]
            grouped[patient]["preds"].append(p)
            grouped[patient]["gts"].append(g)
            grouped[patient]["slice_dice"].append(slice_dice(p, g))
    return grouped


def patient_metrics_from_grouped(grouped: Dict[str, dict], with_hd95: bool = True) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for patient, d in grouped.items():
        if not d["preds"]:
            continue
        out[patient] = {
            "dice": patient_volume_dice(d["preds"], d["gts"]),
            "iou": patient_volume_iou(d["preds"], d["gts"]),
            "hd95": hd95_volume(d["preds"], d["gts"]) if with_hd95 else None,
            "n_slices": len(d["preds"]),
            "n_tumor_slices": int(sum(g.sum() > 0 for g in d["gts"])),
        }
    return out


def evaluate_cv(
    manifest: pd.DataFrame,
    folds: List[dict],
    ckpt_for_fold: Callable[[int], str],
    device,
    in_channels: int = 3,
    channels: str = "reference",
    batchnorm: bool = True,
    with_hd95: bool = True,
    perturb: Optional[Callable] = None,
) -> dict:
    """Evaluate every fold's val patients, pool per-patient scores, summarize.

    Returns {'per_patient': {...}, 'summary': {...}, 'per_slice': {...}}.
    """
    per_patient: Dict[str, dict] = {}
    per_slice_scores: List[float] = []
    per_slice_tumor: List[float] = []

    for fold in folds:
        model = _load_model(ckpt_for_fold(fold["fold"]), in_channels, channels, batchnorm, device)
        grouped = predict_patients(
            model, manifest, fold["val"], device, in_channels=in_channels, perturb=perturb
        )
        pm = patient_metrics_from_grouped(grouped, with_hd95=with_hd95)
        for patient, m in pm.items():
            m["institution"] = manifest[manifest["patient"] == patient]["institution"].iloc[0]
            m["fold"] = fold["fold"]
            per_patient[patient] = m
        for patient, d in grouped.items():
            for sd, g in zip(d["slice_dice"], d["gts"]):
                per_slice_scores.append(sd)
                if g.sum() > 0:
                    per_slice_tumor.append(sd)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    summary = aggregate_patient_metrics(per_patient)
    summary["per_slice_dice_all_mean"] = float(np.mean(per_slice_scores)) if per_slice_scores else 0.0
    summary["per_slice_dice_tumor_mean"] = float(np.mean(per_slice_tumor)) if per_slice_tumor else 0.0
    summary["per_slice_note"] = (
        "per-slice Dice is NOT the headline; the 2,556 empty slices inflate the all-slice number."
    )
    return {"per_patient": per_patient, "summary": summary}
