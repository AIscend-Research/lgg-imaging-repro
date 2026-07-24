"""All paper figures (Section 6, F1-F7) — generated from already-computed results.

Every figure here is a plot of outputs the pipeline has already written (training
logs, per-patient metrics, shape features, radiogenomics tests, genomic labels);
none triggers training. Each figure is saved as a PNG under ``figures/`` and its
underlying numbers as a sibling CSV, so a reviewer can re-plot or audit them.

  F1  training curves (per fold)              <- reports/history_fold*.json
  F2  qualitative low/moderate/high-Dice      <- checkpoints + manifest (inference
      overlays (reproduces the paper Fig. 4)      on 3 patients only)
  F3  per-patient Dice distribution           <- reports/eval_metrics.json
  F4  reproduction comparison vs 0.82/0.85    <- reports/eval_metrics.json
  F5  per-institution Dice (EZ n=1 excluded)  <- reports/eval_metrics.json
  F6  radiogenomics (primary association +    <- reports/radiogenomics.json +
      Fisher p-value heatmap, pred vs GT)         reports/shape_features_*.csv
  F7  discrimination ROC, predicted vs        <- reports/radiogenomics.json
      manual on one axes (REQUIRED)               (roc curve + AUCs)
  F8  shape-feature agreement scatter         <- reports/shape_features_*.csv
      (predicted vs manual, Pearson r)
  F9  fairness by demographic subgroup        <- per_patient + data.csv (optional)
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

TARGET_MEAN_DICE = 0.82
TARGET_MEDIAN_DICE = 0.85


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _save(fig, png_path: str):
    os.makedirs(os.path.dirname(os.path.abspath(png_path)), exist_ok=True)
    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    import matplotlib.pyplot as plt
    plt.close(fig)


def _csv(df: pd.DataFrame, png_path: str):
    df.to_csv(os.path.splitext(png_path)[0] + ".csv", index=False)


# ------------------------------------------------------------ F1 curves -------

def fig_training_curves(reports_dir: str, out_path: str) -> Optional[str]:
    plt = _mpl()
    histories = []
    for fn in sorted(os.listdir(reports_dir)):
        if fn.startswith("history_fold") and fn.endswith(".json"):
            with open(os.path.join(reports_dir, fn)) as fh:
                histories.append((fn, json.load(fh)))
    if not histories:
        return None
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    rows = []
    for fn, h in histories:
        fold = fn.replace("history_", "").replace(".json", "")
        epochs = np.arange(1, len(h.get("train_loss", [])) + 1)
        ax1.plot(epochs, h.get("train_loss", []), label=fold)
        ax2.plot(epochs, h.get("val_dice", []), label=fold)
        for e, tl, vd in zip(epochs, h.get("train_loss", []), h.get("val_dice", [])):
            rows.append({"fold": fold, "epoch": int(e), "train_loss": tl, "val_slice_dice": vd})
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Train loss (Dice+BCE)"); ax1.set_title("Training loss")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Val per-slice Dice"); ax2.set_title("Validation Dice")
    ax1.legend(fontsize=8); ax2.legend(fontsize=8)
    _save(fig, out_path)
    _csv(pd.DataFrame(rows), out_path)
    return out_path


# ------------------------------------------------ F2 qualitative overlays -----

def fig_qualitative(cfg, out_path: str) -> Optional[str]:
    """Low / moderate / high-Dice example overlays (reproduces the paper Fig. 4).

    Picks patients whose per-patient Dice is nearest 0.50 / 0.82 / 0.95, re-runs
    inference on just those 3 patients, and overlays prediction vs ground truth.
    """
    import torch
    from PIL import Image

    from .data.dataset import subset_for_patients
    from .evaluate import predict_patients
    from .models.unet import build_unet
    from .utils import get_device

    reports = cfg["paths"]["reports"]
    eval_path = os.path.join(reports, "eval_metrics.json")
    manifest = cfg["paths"]["manifest"]
    if not (os.path.exists(eval_path) and os.path.exists(manifest)):
        return None
    with open(eval_path) as fh:
        per_patient = json.load(fh)["per_patient"]
    df = pd.read_csv(manifest)

    targets = {"low (~0.50)": 0.50, "moderate (~0.82)": 0.82, "high (~0.95)": 0.95}
    chosen = {}
    used = set()
    for label, t in targets.items():
        best = None
        for p, m in per_patient.items():
            if p in used:
                continue
            d = abs(m["dice"] - t)
            if best is None or d < best[1]:
                best = (p, d)
        if best:
            chosen[label] = (best[0], per_patient[best[0]]["dice"], per_patient[best[0]].get("fold", 0))
            used.add(best[0])
    if not chosen:
        return None

    device = get_device()
    plt = _mpl()
    fig, axes = plt.subplots(1, len(chosen), figsize=(4.5 * len(chosen), 4.8))
    if len(chosen) == 1:
        axes = [axes]
    ckpt_dir = cfg["paths"]["ckpt_dir"]
    in_channels = cfg["data"]["in_channels"]
    channels = cfg["train"]["channels"]
    batchnorm = cfg["train"]["batchnorm"]

    for ax, (label, (patient, dice, fold)) in zip(axes, chosen.items()):
        ckpt = os.path.join(ckpt_dir, f"fold{fold}_best.pt")
        if not os.path.exists(ckpt):
            ax.set_axis_off(); ax.set_title(f"{label}\n(checkpoint missing)"); continue
        state = torch.load(ckpt, map_location=device)
        model = build_unet(channels, in_channels=in_channels, batchnorm=batchnorm).to(device)
        model.load_state_dict(state["model"]); model.eval()
        grouped = predict_patients(model, df, [patient], device, in_channels=in_channels)
        d = grouped[patient]
        # slice with the largest ground-truth tumor
        areas = [g.sum() for g in d["gts"]]
        k = int(np.argmax(areas)) if max(areas) > 0 else len(areas) // 2
        sub = subset_for_patients(df, [patient])
        img = np.asarray(Image.open(sub.iloc[k]["image_path"]).convert("L"), dtype=float)
        gt, pr = d["gts"][k], d["preds"][k]
        ax.imshow(img, cmap="gray")
        ax.contour(gt, levels=[0.5], colors="#55A868", linewidths=1.5)
        ax.contour(pr, levels=[0.5], colors="#C44E52", linewidths=1.5)
        ax.set_title(f"{label}\n{patient}  Dice={dice:.2f}")
        ax.set_axis_off()
    from matplotlib.lines import Line2D
    fig.legend([Line2D([0], [0], color="#55A868"), Line2D([0], [0], color="#C44E52")],
               ["ground truth", "prediction"], loc="lower center", ncol=2)
    _save(fig, out_path)
    pd.DataFrame([{"panel": l, "patient": p, "dice": dc} for l, (p, dc, _) in chosen.items()])\
        .to_csv(os.path.splitext(out_path)[0] + ".csv", index=False)
    return out_path


# --------------------------------------------- F3 / F4 / F5 from eval ---------

def _per_patient(reports_dir: str) -> Dict[str, dict]:
    with open(os.path.join(reports_dir, "eval_metrics.json")) as fh:
        return json.load(fh)["per_patient"]


def fig_dice_distribution(reports_dir: str, out_path: str) -> Optional[str]:
    plt = _mpl()
    pp = _per_patient(reports_dir)
    dice = np.array([m["dice"] for m in pp.values()])
    fig, (axh, axb) = plt.subplots(1, 2, figsize=(11, 4.5),
                                   gridspec_kw={"width_ratios": [3, 1]})
    axh.hist(dice, bins=20, color="#4C72B0", edgecolor="white")
    axh.axvline(dice.mean(), color="#C44E52", lw=2, label=f"mean {dice.mean():.3f}")
    axh.axvline(np.median(dice), color="#55A868", lw=2, ls="--", label=f"median {np.median(dice):.3f}")
    axh.axvline(TARGET_MEAN_DICE, color="k", lw=1.5, ls=":", label="paper mean 0.82")
    axh.axvline(TARGET_MEDIAN_DICE, color="grey", lw=1.5, ls=":", label="paper median 0.85")
    axh.set_xlabel("Per-patient volume Dice"); axh.set_ylabel("Patients")
    axh.set_title(f"Per-patient Dice (n={len(dice)}, headline)"); axh.legend(fontsize=8)
    axb.boxplot(dice, vert=True, widths=0.5); axb.set_xticks([]); axb.set_ylabel("Dice")
    axb.set_title("Boxplot")
    _save(fig, out_path)
    _csv(pd.DataFrame({"patient": list(pp.keys()), "dice": dice}), out_path)
    return out_path


def fig_reproduction_comparison(reports_dir: str, out_path: str) -> Optional[str]:
    plt = _mpl()
    with open(os.path.join(reports_dir, "eval_metrics.json")) as fh:
        s = json.load(fh)["summary"]
    labels = ["Mean Dice", "Median Dice"]
    paper = [TARGET_MEAN_DICE, TARGET_MEDIAN_DICE]
    ours = [s["dice_mean"], s["dice_median"]]
    x = np.arange(len(labels)); w = 0.38
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(x - w / 2, paper, w, label="Paper (Buda 2019)", color="#8C8C8C")
    ax.bar(x + w / 2, ours, w, label="This work", color="#4C72B0")
    for i, (pv, ov) in enumerate(zip(paper, ours)):
        ax.text(i - w / 2, pv + 0.01, f"{pv:.2f}", ha="center", fontsize=9)
        ax.text(i + w / 2, ov + 0.01, f"{ov:.3f}\n(Δ{ov - pv:+.3f})", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylim(0, 1)
    ax.set_ylabel("Dice"); ax.set_title("Segmentation reproduction vs paper"); ax.legend()
    _save(fig, out_path)
    _csv(pd.DataFrame({"metric": labels, "paper": paper, "this_work": ours,
                       "delta": [o - p for o, p in zip(ours, paper)]}), out_path)
    return out_path


def fig_dice_by_institution(reports_dir: str, out_path: str) -> Optional[str]:
    plt = _mpl()
    pp = _per_patient(reports_dir)
    groups: Dict[str, list] = {}
    for m in pp.values():
        groups.setdefault(m["institution"], []).append(m["dice"])
    # F5: flag EZ (n=1) and exclude it from the boxplot.
    excluded = {s: v for s, v in groups.items() if len(v) < 2}
    kept = {s: v for s, v in groups.items() if len(v) >= 2}
    sites = sorted(kept, key=lambda s: np.median(kept[s]))
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.boxplot([kept[s] for s in sites], labels=[f"{s}\n(n={len(kept[s])})" for s in sites])
    ax.axhline(TARGET_MEAN_DICE, color="k", ls=":", label="paper mean 0.82")
    ax.set_ylabel("Per-patient Dice"); ax.set_ylim(0, 1)
    title = "Per-institution Dice"
    if excluded:
        title += "  (excluded, n=1: " + ", ".join(excluded) + ")"
    ax.set_title(title); ax.legend()
    _save(fig, out_path)
    rows = [{"institution": s, "patient_dice": d} for s in kept for d in kept[s]]
    _csv(pd.DataFrame(rows), out_path)
    return out_path


# ---------------------------------------------------- F6 radiogenomics --------

def fig_radiogenomics(reports_dir: str, out_path: str) -> Optional[str]:
    plt = _mpl()
    rg_path = os.path.join(reports_dir, "radiogenomics.json")
    if not os.path.exists(rg_path):
        return None
    with open(rg_path) as fh:
        rg = json.load(fh)

    # (a) primary association: RNASeqCluster x BEVR feature across clusters,
    #     predicted mask; requires the saved shape-feature table + labels join.
    pred_csv = os.path.join(reports_dir, "shape_features_predicted.csv")
    have_strip = os.path.exists(pred_csv)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax_strip, ax_heat = axes

    if have_strip:
        feat = pd.read_csv(pred_csv)
        if {"RNASeqCluster", "bevr"}.issubset(feat.columns):
            sub = feat.dropna(subset=["RNASeqCluster", "bevr"])
            clusters = sorted(sub["RNASeqCluster"].astype(str).unique())
            data = [sub[sub["RNASeqCluster"].astype(str) == c]["bevr"].to_numpy() for c in clusters]
            ax_strip.boxplot(data, labels=clusters)
            for i, arr in enumerate(data, start=1):
                ax_strip.scatter(np.random.normal(i, 0.06, len(arr)), arr, s=12,
                                 alpha=0.6, color="#4C72B0")
            ax_strip.set_xlabel("RNASeq cluster"); ax_strip.set_ylabel("Bounding ellipsoid volume ratio")
            ax_strip.set_title("Primary association (predicted masks)")
        else:
            ax_strip.set_axis_off()
    else:
        ax_strip.set_axis_off()

    # (b) Fisher corrected-p heatmap, predicted vs ground truth side by side.
    def _matrix(mask_key):
        best = rg["masks"][mask_key]["associations"]["best_per_pair"]
        feats = rg["features"]; subs = rg["subtypes_tested"]
        M = np.full((len(feats), len(subs)), np.nan)
        for r in best:
            i, j = feats.index(r["feature"]), subs.index(r["subtype"])
            M[i, j] = -np.log10(max(r["p_bonferroni"], 1e-12))
        return M, feats, subs

    Mp, feats, subs = _matrix("predicted")
    Mg, _, _ = _matrix("ground_truth")
    combined = np.concatenate([Mp, Mg], axis=1)
    im = ax_heat.imshow(combined, aspect="auto", cmap="viridis")
    ax_heat.set_yticks(range(len(feats))); ax_heat.set_yticklabels(feats)
    xt = [f"{s}\n(pred)" for s in subs] + [f"{s}\n(GT)" for s in subs]
    ax_heat.set_xticks(range(len(xt))); ax_heat.set_xticklabels(xt, rotation=90, fontsize=7)
    ax_heat.axvline(len(subs) - 0.5, color="w", lw=2)
    ax_heat.set_title("Fisher −log10(Bonferroni p): predicted | ground truth")
    fig.colorbar(im, ax=ax_heat, label="−log10 corrected p")
    _save(fig, out_path)

    rows = []
    for mask_key in ("predicted", "ground_truth"):
        for r in rg["masks"][mask_key]["associations"]["best_per_pair"]:
            rows.append({"mask": mask_key, **r})
    _csv(pd.DataFrame(rows), out_path)
    return out_path


# ------------------------------------ F7 radiogenomic discrimination ROC ------

def fig_discrimination_roc(reports_dir: str, out_path: str) -> Optional[str]:
    """Predicted-mask vs manual-mask ROC for the paper's headline validation.

    Cluster ~R2 vs the rest using inverse BEVR: the single figure showing the
    automatic segmentation preserves the genomic association about as well as the
    manual masks (paper AUC ~= 0.80 model / 0.78 manual). REQUIRED (Section 6, F7).
    """
    plt = _mpl()
    rg_path = os.path.join(reports_dir, "radiogenomics.json")
    if not os.path.exists(rg_path):
        return None
    with open(rg_path) as fh:
        rg = json.load(fh)
    ref = rg.get("paper_reference", {})
    styles = {"predicted": ("#4C72B0", ref.get("auc_model", 0.80), "predicted (model)"),
              "ground_truth": ("#C44E52", ref.get("auc_manual", 0.78), "ground truth (manual)")}
    fig, ax = plt.subplots(figsize=(6, 6))
    rows = []
    any_curve = False
    for mask, (color, paper_auc, label) in styles.items():
        a = rg["masks"].get(mask, {}).get("discrimination_auc")
        if not a or "roc_fpr" not in a:
            continue
        any_curve = True
        fpr, tpr = a["roc_fpr"], a["roc_tpr"]
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f"{label}: AUC={a['auc']:.3f} (paper {paper_auc:.2f})")
        for f, t in zip(fpr, tpr):
            rows.append({"mask": mask, "fpr": f, "tpr": t, "auc": a["auc"]})
    if not any_curve:
        import matplotlib.pyplot as plt2  # noqa
        plt.close(fig)
        return None
    ax.plot([0, 1], [0, 1], color="grey", ls=":", lw=1)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    a0 = rg["masks"]["predicted"].get("discrimination_auc") or {}
    ax.set_title(f"RNASeq {a0.get('target_cluster', 'R2')} vs rest via inverse BEVR")
    ax.legend(loc="lower right", fontsize=9)
    _save(fig, out_path)
    _csv(pd.DataFrame(rows), out_path)
    return out_path


# ------------------------------------------ F8 shape-feature agreement --------

def fig_feature_agreement(reports_dir: str, out_path: str) -> Optional[str]:
    """Per-patient predicted-vs-manual scatter for each shape feature (Pearson r).

    Shows the automatic segmentation reproduces the feature *values*, not just the
    downstream association. Optional (Section 6, F8) but ~0 cost.
    """
    pp = os.path.join(reports_dir, "shape_features_predicted.csv")
    gp = os.path.join(reports_dir, "shape_features_ground_truth.csv")
    if not (os.path.exists(pp) and os.path.exists(gp)):
        return None
    pred = pd.read_csv(pp).set_index("patient")
    gt = pd.read_csv(gp).set_index("patient")
    feats = [f for f in ("bevr", "angular_std", "margin_fluctuation")
             if f in pred.columns and f in gt.columns]
    if not feats:
        return None
    plt = _mpl()
    fig, axes = plt.subplots(1, len(feats), figsize=(4.6 * len(feats), 4.6))
    if len(feats) == 1:
        axes = [axes]
    rows = []
    for ax, feat in zip(axes, feats):
        j = pd.DataFrame({"pred": pred[feat], "gt": gt[feat]}).dropna()
        if len(j) < 3:
            ax.set_axis_off(); ax.set_title(f"{feat}\n(insufficient data)"); continue
        r = float(np.corrcoef(j["gt"], j["pred"])[0, 1])
        ax.scatter(j["gt"], j["pred"], s=16, alpha=0.6, color="#4C72B0")
        lo = float(min(j["gt"].min(), j["pred"].min()))
        hi = float(max(j["gt"].max(), j["pred"].max()))
        ax.plot([lo, hi], [lo, hi], color="grey", ls=":", lw=1)  # identity line
        ax.set_xlabel(f"manual {feat}"); ax.set_ylabel(f"predicted {feat}")
        ax.set_title(f"{feat}\nPearson r = {r:.3f} (n={len(j)})", fontsize=9)
        for pat, row in j.iterrows():
            rows.append({"feature": feat, "patient": pat, "manual": row["gt"], "predicted": row["pred"]})
    fig.suptitle("Shape-feature agreement: predicted vs manual masks", fontweight="bold")
    _save(fig, out_path)
    _csv(pd.DataFrame(rows), out_path)
    return out_path


# ------------------------------------------------------ F9 fairness -----------

def fig_fairness(cfg, out_path: str) -> Optional[str]:
    from .radiogenomics import DEMOGRAPHIC_COLUMNS, _patient_key, load_genomic_labels

    reports = cfg["paths"]["reports"]
    pp = _per_patient(reports)
    labels = load_genomic_labels(cfg["paths"]["datapath"])
    if labels.empty:
        return None
    dice = pd.DataFrame([{"_key": _patient_key(p), "dice": m["dice"]} for p, m in pp.items()])
    dice = dice.set_index("_key")
    cols = [c for c in DEMOGRAPHIC_COLUMNS if c in labels.columns]
    if not cols:
        return None
    plt = _mpl()
    fig, axes = plt.subplots(1, len(cols), figsize=(4.5 * len(cols), 4.8))
    if len(cols) == 1:
        axes = [axes]
    rows = []
    for ax, col in zip(axes, cols):
        joined = dice.join(labels[[col]], how="inner").dropna()
        groups = {str(g): sub["dice"].to_numpy() for g, sub in joined.groupby(col)}
        keys = sorted(groups, key=lambda k: len(groups[k]), reverse=True)
        ax.boxplot([groups[k] for k in keys],
                   labels=[f"{k}\n(n={len(groups[k])})" for k in keys])
        ax.axhline(TARGET_MEAN_DICE, color="k", ls=":", lw=1)
        ax.set_title(col, fontsize=9); ax.set_ylim(0, 1)
        ax.tick_params(axis="x", labelsize=7, rotation=30)
        for k in keys:
            rows.append({"attribute": col, "group": k, "n": len(groups[k]),
                         "dice_mean": float(np.mean(groups[k]))})
    fig.suptitle("Per-patient Dice by demographic subgroup (US TCIA cohort)", fontweight="bold")
    _save(fig, out_path)
    _csv(pd.DataFrame(rows), out_path)
    return out_path


# ------------------------------------------------------------ orchestrator ----

def make_all_figures(cfg) -> List[str]:
    reports = cfg["paths"]["reports"]
    figures = cfg["paths"]["figures"]
    os.makedirs(figures, exist_ok=True)
    made = []
    plan = [
        ("F1_training_curves.png", lambda p: fig_training_curves(reports, p)),
        ("F2_qualitative.png", lambda p: fig_qualitative(cfg, p)),
        ("F3_dice_distribution.png", lambda p: fig_dice_distribution(reports, p)),
        ("F4_reproduction_comparison.png", lambda p: fig_reproduction_comparison(reports, p)),
        ("F5_dice_by_institution.png", lambda p: fig_dice_by_institution(reports, p)),
        ("F6_radiogenomics.png", lambda p: fig_radiogenomics(reports, p)),
        ("F7_discrimination_roc.png", lambda p: fig_discrimination_roc(reports, p)),
        ("F8_feature_agreement.png", lambda p: fig_feature_agreement(reports, p)),
        ("F9_fairness.png", lambda p: fig_fairness(cfg, p)),
    ]
    for name, fn in plan:
        out = os.path.join(figures, name)
        try:
            res = fn(out)
        except Exception as e:  # a missing input skips one figure, never the batch
            print(f"[figures] {name} skipped: {e}")
            res = None
        if res:
            made.append(res)
            print(f"[figures] wrote {name}")
        else:
            print(f"[figures] {name} skipped (inputs not available)")
    return made
