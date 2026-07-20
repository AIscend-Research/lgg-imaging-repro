"""Leave-one-institution-out training and evaluation (Section 6).

For each site in {CS, DU, FG, HT} we train on the other sites' patients and
evaluate on the held-out site with the per-patient headline metric. EZ (n=1) is
degenerate — a single patient cannot form a meaningful held-out site — so it is
excluded from the leave-one-out loop and flagged (Section 6 requirement).
"""

from __future__ import annotations

from typing import Dict, List

import pandas as pd

from .data.splits import institution_of
from .evaluate import patient_metrics_from_grouped, predict_patients
from .metrics import aggregate_patient_metrics
from .models.unet import build_unet
from .train import TrainConfig, train_fold

# Sites with too few patients to hold out. EZ has n=1 (degenerate).
DEGENERATE_SITES = {"EZ"}


def institutions_in(manifest: pd.DataFrame) -> Dict[str, List[str]]:
    sites: Dict[str, List[str]] = {}
    for p in sorted(manifest["patient"].unique()):
        sites.setdefault(institution_of(p), []).append(p)
    return sites


def run_leave_one_institution_out(
    manifest: pd.DataFrame,
    cfg: TrainConfig,
    ckpt_dir: str,
    device,
    epochs: int = None,
) -> dict:
    import os

    import torch

    sites = institutions_in(manifest)
    heldout_sites = [s for s in sites if s not in DEGENERATE_SITES]
    results = {"excluded_degenerate": sorted(DEGENERATE_SITES & set(sites)), "per_site": {}}

    for site in sorted(heldout_sites):
        test_patients = sites[site]
        train_patients = [p for s, ps in sites.items() if s != site for p in ps]
        ckpt = os.path.join(ckpt_dir, f"loio_{site}.pt")
        hist = train_fold(manifest, train_patients, test_patients, cfg, ckpt, device=device)

        model = build_unet(cfg.channels, in_channels=cfg.in_channels, batchnorm=cfg.batchnorm).to(device)
        model.load_state_dict(torch.load(ckpt.replace(".pt", "_best.pt"), map_location=device)["model"])
        model.eval()
        grouped = predict_patients(model, manifest, test_patients, device, in_channels=cfg.in_channels)
        pm = patient_metrics_from_grouped(grouped, with_hd95=False)
        summ = aggregate_patient_metrics(pm)
        results["per_site"][site] = {
            "n_patients": len(test_patients),
            "dice_mean": summ["dice_mean"],
            "dice_median": summ["dice_median"],
            "iou_mean": summ["iou_mean"],
            "train_seconds": hist.get("total_seconds"),
        }
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    dice = [v["dice_mean"] for v in results["per_site"].values()]
    if dice:
        results["dice_mean_over_sites"] = sum(dice) / len(dice)
        results["worst_site"] = min(results["per_site"], key=lambda s: results["per_site"][s]["dice_mean"])
        results["best_site"] = max(results["per_site"], key=lambda s: results["per_site"][s]["dice_mean"])
    return results
