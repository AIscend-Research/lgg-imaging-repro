"""Write reports/metrics.json + reports/comparison.md and paper figures (R10).

The comparison table puts our per-patient mean / median Dice next to the paper's
target (0.82 mean / 0.85 median) with signed deltas. IoU / HD95 are shown in a
clearly separate block so they are never confused with the reproduction target
(R3). Figure helpers render the paper's plots from the saved metrics.
"""

from __future__ import annotations

import json
import os
from typing import Dict, Optional

import numpy as np

TARGET_MEAN_DICE = 0.82
TARGET_MEDIAN_DICE = 0.85


def write_metrics_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2, default=_json_default)


def _json_default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(str(type(o)))


def write_comparison_md(path: str, summary: dict, meta: dict, radiogenomics: Optional[dict] = None,
                        deployment: Optional[dict] = None) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    mean_d = summary["dice_mean"]
    med_d = summary["dice_median"]
    lines = []
    kind = "full" if radiogenomics else "partial"
    lines.append("# Reproduction comparison — Buda et al. (2019)\n")
    lines.append(
        f"**{kind.capitalize()} replication.** This covers the segmentation result and"
        + (" the radiogenomics association (the paper's actual headline claim)."
           if radiogenomics else " (segmentation only; radiogenomics not run).")
        + " The segmentation headline metric is **per-patient (volume-aggregated) Dice** "
        f"over the 110 patients, patient-level cross-validation (`{meta.get('preset')}`, "
        f"seed {meta.get('seed')}, {meta.get('in_channels')}-channel input).\n"
    )
    lines.append("## Headline: per-patient Dice vs paper target\n")
    lines.append("| Metric | Paper (target) | This work | Δ |")
    lines.append("|---|---|---|---|")
    lines.append(f"| Mean Dice (per patient) | {TARGET_MEAN_DICE:.2f} | {mean_d:.3f} | {mean_d - TARGET_MEAN_DICE:+.3f} |")
    lines.append(f"| Median Dice (per patient) | {TARGET_MEDIAN_DICE:.2f} | {med_d:.3f} | {med_d - TARGET_MEDIAN_DICE:+.3f} |")
    lines.append(f"\n- Patients scored: {summary['n_patients']}")
    lines.append(f"- Cross-fold / distribution std of per-patient Dice: {summary['dice_std']:.3f}\n")

    lines.append("## Additional metrics (NOT part of the reproduction target, R3)\n")
    lines.append("| Metric | Mean | Median |")
    lines.append("|---|---|---|")
    lines.append(f"| IoU (per patient) | {summary['iou_mean']:.3f} | {summary['iou_median']:.3f} |")
    if summary.get("hd95_mean") is not None:
        lines.append(f"| HD95 (per patient, computable n={summary['hd95_n_computable']}) | "
                     f"{summary['hd95_mean']:.2f} | {summary['hd95_median']:.2f} |")
    else:
        lines.append("| HD95 | not computed (medpy unavailable or empty predictions) | — |")

    lines.append("\n## Per-slice Dice (reference only — never the headline, R2)\n")
    lines.append("The 2,556 empty slices inflate the all-slice per-slice number; it is a "
                 "different, rosier quantity and is not comparable to the paper's per-patient Dice.\n")
    lines.append(f"- All slices (tumor + empty): {summary.get('per_slice_dice_all_mean', float('nan')):.3f}")
    lines.append(f"- Tumor-bearing slices only: {summary.get('per_slice_dice_tumor_mean', float('nan')):.3f}\n")

    if radiogenomics:
        lines.extend(_radiogenomics_md(radiogenomics))

    if deployment:
        lines.append("## Deployment numbers (R8)\n")
        lines.append("| Quantity | Value |")
        lines.append("|---|---|")
        lines.append(f"| Parameters | {deployment.get('params', 'n/a'):,} ({deployment.get('params_M', 'n/a')} M) |")
        lines.append(f"| On-disk size | {deployment.get('size_mb', 'n/a')} MB |")
        if deployment.get("peak_vram_mb") is not None:
            lines.append(f"| Peak inference VRAM | {deployment['peak_vram_mb']} MB |")
            lines.append(f"| GPU latency / volume | {deployment.get('gpu_ms_per_volume', 'n/a')} ms |")
        lines.append(f"| CPU latency / volume | {deployment.get('cpu_ms_per_volume', 'n/a')} ms |")
        lines.append(f"\n_Benchmarked on {deployment.get('checkpoint', 'the trained model')} "
                     f"({deployment.get('device', '?')})._\n")

    verdict = "within" if abs(mean_d - TARGET_MEAN_DICE) <= 0.03 else "off"
    lines.append(f"**Segmentation verdict:** per-patient mean Dice {mean_d:.3f} is {verdict} the "
                 f"paper's 0.82 (Δ {mean_d - TARGET_MEAN_DICE:+.3f}).\n")
    with open(path, "w") as fh:
        fh.write("\n".join(lines))


def _pass_note(ours: Optional[float], target: float, smaller_is_better: bool) -> str:
    if ours is None:
        return "n/a"
    hit = (ours <= target) if smaller_is_better else (ours >= target)
    if hit:
        return "pass"
    near = ours <= target * 3 if smaller_is_better else ours >= target - 0.05
    return "near" if near else "miss"


def _radiogenomics_md(rg: dict) -> list:
    """Section 7 comparison: our associations/AUCs vs the paper, predicted & GT."""
    ref = rg.get("paper_reference", {})
    lines = ["## Radiogenomics — the paper's headline claim (Section 7)\n"]
    lines.append("Shape features of the tumor mask vs genomic subtype, for **predicted** and "
                 "**ground-truth** masks. The paper's point is that the automatic masks preserve "
                 "the associations about as well as the manual ones.\n")

    # Primary associations (RNASeq × BEVR, RNASeq × margin fluctuation).
    lines.append("### Primary Fisher associations (Bonferroni-corrected)\n")
    lines.append("| Association | Paper p | Predicted p | GT p | Note (predicted) |")
    lines.append("|---|---|---|---|---|")
    pairs = [("RNASeqCluster", "bevr", ref.get("rnaseq_x_bevr_p", 0.0002), "RNASeq × BEVR"),
             ("RNASeqCluster", "margin_fluctuation", ref.get("rnaseq_x_margin_fluctuation_p", 0.005),
              "RNASeq × margin fluctuation")]
    for sub, feat, paper_p, label in pairs:
        def _p(mask):
            best = rg["masks"][mask]["associations"]["best_per_pair"]
            r = next((x for x in best if x["subtype"] == sub and x["feature"] == feat), None)
            return r["p_bonferroni"] if r else None
        pp, gp = _p("predicted"), _p("ground_truth")
        note = _pass_note(pp, paper_p, smaller_is_better=True)
        lines.append(f"| {label} | < {paper_p} | {_fmt_p(pp)} | {_fmt_p(gp)} | {note} |")

    # Discrimination AUC (cluster R2 vs rest, inverse BEVR).
    lines.append("\n### Discrimination AUC (cluster ~R2 vs rest, inverse BEVR)\n")
    lines.append("| Mask | Paper AUC | This work | Note |")
    lines.append("|---|---|---|---|")
    for mask, paper_auc, name in (("predicted", ref.get("auc_model", 0.80), "model / predicted"),
                                  ("ground_truth", ref.get("auc_manual", 0.78), "manual / GT")):
        a = rg["masks"][mask]["discrimination_auc"]
        ours = a.get("auc") if a else None
        note = _pass_note(ours, paper_auc - 0.05, smaller_is_better=False)
        lines.append(f"| {name} | {paper_auc:.2f} | {ours:.3f} | {note} |" if ours is not None
                     else f"| {name} | {paper_auc:.2f} | n/a | n/a |")
    lines.append("")
    return lines


def _fmt_p(p: Optional[float]) -> str:
    if p is None:
        return "n/a"
    return f"{p:.2e}" if p < 0.01 else f"{p:.3f}"


# ---------------------------------------------------------------- figures ----

def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def fig_dice_distribution(per_patient: Dict[str, dict], out_path: str) -> None:
    plt = _mpl()
    dice = [m["dice"] for m in per_patient.values()]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(dice, bins=20, color="#4C72B0", edgecolor="white")
    ax.axvline(np.mean(dice), color="#C44E52", lw=2, label=f"mean {np.mean(dice):.3f}")
    ax.axvline(np.median(dice), color="#55A868", lw=2, ls="--", label=f"median {np.median(dice):.3f}")
    ax.axvline(TARGET_MEAN_DICE, color="k", lw=1.5, ls=":", label="paper mean 0.82")
    ax.set_xlabel("Per-patient volume Dice")
    ax.set_ylabel("Patients")
    ax.set_title("Per-patient Dice distribution (headline metric)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def fig_dice_by_institution(per_patient: Dict[str, dict], out_path: str) -> None:
    plt = _mpl()
    groups: Dict[str, list] = {}
    for m in per_patient.values():
        groups.setdefault(m["institution"], []).append(m["dice"])
    sites = sorted(groups, key=lambda s: np.mean(groups[s]))
    means = [np.mean(groups[s]) for s in sites]
    ns = [len(groups[s]) for s in sites]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar([f"{s}\n(n={n})" for s, n in zip(sites, ns)], means, color="#4C72B0")
    ax.axhline(TARGET_MEAN_DICE, color="k", ls=":", label="paper mean 0.82")
    ax.set_ylabel("Mean per-patient Dice")
    ax.set_title("Dice by institution (cross-site equity)")
    ax.set_ylim(0, 1)
    for b, v in zip(bars, means):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def fig_robustness(robustness: dict, out_path: str) -> None:
    plt = _mpl()
    names = list(robustness.keys())
    dice = [robustness[n]["dice_mean"] for n in names]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = ["#55A868" if n == "clean" else "#C44E52" for n in names]
    ax.barh(names, dice, color=colors)
    ax.axvline(robustness["clean"]["dice_mean"], color="k", ls=":", label="clean")
    ax.set_xlabel("Mean per-patient Dice")
    ax.set_title("Robustness: Dice under input degradation")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def fig_model_comparison(deploy: dict, out_path: str) -> None:
    """deploy: {model_name: {'size_mb','dice_mean','cpu_ms_per_slice','params_M'}}."""
    plt = _mpl()
    names = list(deploy.keys())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].bar(names, [deploy[n]["size_mb"] for n in names], color="#4C72B0")
    axes[0].set_ylabel("Size (MB)")
    axes[0].set_title("On-disk size")
    axes[1].bar(names, [deploy[n].get("dice_mean", 0) for n in names], color="#55A868")
    axes[1].set_ylabel("Per-patient Dice")
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Accuracy")
    fig.suptitle("Lightweight / quantized model tradeoff", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
