"""Metric definitions and empty-mask conventions (R2, R3).

Headline metric = **per-patient (volume-aggregated) Dice** (R2): for each patient
we sum intersections and sums across ALL that patient's slices, then form one
Dice per patient, then take mean / median over patients. This is the number that
compares to the paper's 0.82 mean / 0.85 median.

Per-slice Dice is also computed but ALWAYS labelled separately and never
presented as the headline (R2). IoU and HD95 are additions (the paper reports
neither) and are kept visually distinct (R3).

Empty-mask convention (stated explicitly, R2):
  * both prediction and ground truth empty -> Dice = 1.0, IoU = 1.0
  * exactly one empty                       -> Dice = 0.0, IoU = 0.0
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np


def _counts(pred: np.ndarray, gt: np.ndarray):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    return int(inter), int(pred.sum()), int(gt.sum())


def dice_from_counts(inter: int, pred_sum: int, gt_sum: int) -> float:
    if pred_sum == 0 and gt_sum == 0:
        return 1.0
    if pred_sum == 0 or gt_sum == 0:
        return 0.0
    return 2.0 * inter / (pred_sum + gt_sum)


def iou_from_counts(inter: int, pred_sum: int, gt_sum: int) -> float:
    union = pred_sum + gt_sum - inter
    if union == 0:
        return 1.0
    if pred_sum == 0 or gt_sum == 0:
        return 0.0
    return inter / union


def slice_dice(pred: np.ndarray, gt: np.ndarray) -> float:
    return dice_from_counts(*_counts(pred, gt))


def patient_volume_dice(preds: List[np.ndarray], gts: List[np.ndarray]) -> float:
    """Volume-aggregated Dice over one patient's slices (R2)."""
    ti = tp = tg = 0
    for p, g in zip(preds, gts):
        i, ps, gs = _counts(p, g)
        ti += i
        tp += ps
        tg += gs
    return dice_from_counts(ti, tp, tg)


def patient_volume_iou(preds: List[np.ndarray], gts: List[np.ndarray]) -> float:
    ti = tp = tg = 0
    for p, g in zip(preds, gts):
        i, ps, gs = _counts(p, g)
        ti += i
        tp += ps
        tg += gs
    return iou_from_counts(ti, tp, tg)


def hd95_volume(preds: List[np.ndarray], gts: List[np.ndarray], spacing=1.0) -> Optional[float]:
    """95th-percentile Hausdorff distance for one patient (R3).

    Uses medpy if available. Guards against empty predictions/GT: if either the
    stacked prediction or GT volume is empty, HD95 is undefined -> returns None
    (and the caller reports the fraction of patients where it was computable).
    Computed per patient on the stacked 3D volume.
    """
    try:
        from medpy.metric.binary import hd95 as _hd95
    except Exception:
        return None
    pv = np.stack([p.astype(bool) for p in preds], axis=0)
    gv = np.stack([g.astype(bool) for g in gts], axis=0)
    if pv.sum() == 0 or gv.sum() == 0:
        return None
    try:
        return float(_hd95(pv, gv, voxelspacing=spacing))
    except Exception:
        return None


def aggregate_patient_metrics(per_patient: Dict[str, dict]) -> dict:
    """Given {patient: {'dice':..,'iou':..,'hd95':..}} return mean/median summary."""
    dice = np.array([m["dice"] for m in per_patient.values()], dtype=float)
    iou = np.array([m["iou"] for m in per_patient.values()], dtype=float)
    hd = [m["hd95"] for m in per_patient.values() if m.get("hd95") is not None]
    out = {
        "n_patients": len(per_patient),
        "dice_mean": float(dice.mean()),
        "dice_median": float(np.median(dice)),
        "dice_std": float(dice.std()),
        "iou_mean": float(iou.mean()),
        "iou_median": float(np.median(iou)),
    }
    if hd:
        out.update(
            {
                "hd95_mean": float(np.mean(hd)),
                "hd95_median": float(np.median(hd)),
                "hd95_n_computable": len(hd),
            }
        )
    else:
        out["hd95_mean"] = None
        out["hd95_median"] = None
        out["hd95_n_computable"] = 0
    return out


# ---- differentiable training losses ----------------------------------------

def soft_dice_loss(logits, target, smooth: float = 1.0):
    import torch

    pred = torch.sigmoid(logits)
    inter = (pred * target).sum(dim=(2, 3))
    denom = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    return (1 - (2 * inter + smooth) / (denom + smooth)).mean()


def dice_bce_loss(logits, target, dice_weight: float = 1.0, bce_weight: float = 1.0):
    """Soft-Dice + BCE (R5) — the imbalance handling the paper/framework uses."""
    import torch.nn.functional as F

    return dice_weight * soft_dice_loss(logits, target) + bce_weight * F.binary_cross_entropy_with_logits(
        logits, target
    )
