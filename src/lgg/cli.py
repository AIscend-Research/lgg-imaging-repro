"""Command-line entry point (R7).

Core (full replication): prepare-data, make-splits, train, evaluate,
shape-features, radiogenomics, figures, report, smoke.
Optional (Section 12 GPU extras): run-nnunet, run-robustness, quantize,
distill, cross-institution.

Everything is driven by ``config.yaml``; flags override individual fields. Run as
``python -m lgg.cli <subcommand> [--config config.yaml] [...]``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import pandas as pd

from .utils import get_device, load_config, set_seed


def _cfg(args):
    return load_config(args.config)


def _manifest(cfg):
    path = cfg["paths"]["manifest"]
    if not os.path.exists(path):
        sys.exit(f"manifest not found at {path}; run `prepare-data` first.")
    return pd.read_csv(path)


def _load_folds(cfg):
    from .data.splits import load_splits

    return load_splits(cfg["paths"]["splits"])["folds"]


def _ckpt_for_fold(cfg, best=True):
    d = cfg["paths"]["ckpt_dir"]
    suffix = "_best.pt" if best else ".pt"
    return lambda i: os.path.join(d, f"fold{i}{suffix}")


# --------------------------------------------------------------- commands ----

def cmd_prepare_data(args):
    from .data.prepare import build_manifest, summarize_manifest

    cfg = _cfg(args)
    datapath = args.datapath or cfg["paths"]["datapath"]
    df = build_manifest(datapath, cfg["paths"]["manifest"])
    summary = summarize_manifest(df)
    os.makedirs(cfg["paths"]["reports"], exist_ok=True)
    with open(os.path.join(cfg["paths"]["reports"], "dataset_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"\n[ok] manifest -> {cfg['paths']['manifest']} ({summary['n_slices']} slices, "
          f"{summary['n_patients']} patients)")


def cmd_make_splits(args):
    from .data.splits import make_folds, patient_ids_from_manifest, save_splits

    cfg = _cfg(args)
    df = _manifest(cfg)
    preset = args.preset or cfg["splits"]["preset"]
    seed = cfg["splits"]["seed"]
    patients = patient_ids_from_manifest(df)
    burden = None
    if cfg["splits"].get("stratify_by_site"):
        burden = df.groupby("patient")["tumor_pixels"].sum().to_dict()
    folds = make_folds(patients, preset=preset, seed=seed,
                       stratify_by_site=cfg["splits"].get("stratify_by_site", False),
                       stratify_burden=burden)
    save_splits(folds, cfg["paths"]["splits"], preset=preset, seed=seed,
                stratify_by_site=cfg["splits"].get("stratify_by_site", False))
    print(f"[ok] {preset}: {len(folds)} folds -> {cfg['paths']['splits']}")
    for f in folds[:3]:
        print(f"  fold {f['fold']}: {len(f['train'])} train / {len(f['val'])} val patients")


def _train_config(cfg, args):
    from .train import TrainConfig

    t = cfg["train"]
    if t.get("pretrained_encoder"):
        sys.exit("R4 violation: train.pretrained_encoder must be false for the reproduction.")
    return TrainConfig(
        epochs=args.epochs or t["epochs"],
        batch_size=t["batch_size"], lr=t["lr"], in_channels=cfg["data"]["in_channels"],
        channels=t["channels"], batchnorm=t["batchnorm"], augment=t["augment"],
        amp=t["amp"], num_workers=t["num_workers"], foreground_bias=t["foreground_bias"],
        dice_weight=t["dice_weight"], bce_weight=t["bce_weight"], seed=cfg["seed"],
        grad_clip=t.get("grad_clip", 0.0),
    )


def cmd_train(args):
    from .train import train_fold

    cfg = _cfg(args)
    set_seed(cfg["seed"])
    df = _manifest(cfg)
    folds = _load_folds(cfg)
    tc = _train_config(cfg, args)
    device = get_device()
    os.makedirs(cfg["paths"]["ckpt_dir"], exist_ok=True)
    os.makedirs(cfg["paths"]["reports"], exist_ok=True)

    target = [args.fold] if args.fold is not None else [f["fold"] for f in folds]
    for fi in target:
        fold = folds[fi]
        ckpt = os.path.join(cfg["paths"]["ckpt_dir"], f"fold{fi}.pt")
        log = os.path.join(cfg["paths"]["reports"], f"train_log_fold{fi}.txt")
        print(f"\n=== fold {fi} ===")
        hist = train_fold(df, fold["train"], fold["val"], tc, ckpt, device=device,
                          resume=args.resume, log_path=log)
        with open(os.path.join(cfg["paths"]["reports"], f"history_fold{fi}.json"), "w") as fh:
            json.dump(hist, fh, indent=2)


def cmd_evaluate(args):
    from .evaluate import evaluate_cv
    from .report import write_metrics_json

    cfg = _cfg(args)
    df = _manifest(cfg)
    folds = _load_folds(cfg)
    device = get_device()
    res = evaluate_cv(df, folds, _ckpt_for_fold(cfg), device,
                      in_channels=cfg["data"]["in_channels"], channels=cfg["train"]["channels"],
                      batchnorm=cfg["train"]["batchnorm"], with_hd95=not args.no_hd95)
    out = os.path.join(cfg["paths"]["reports"], "eval_metrics.json")
    write_metrics_json(out, res)
    pd.DataFrame.from_dict(res["per_patient"], orient="index").to_csv(
        os.path.join(cfg["paths"]["reports"], "per_patient.csv"))
    s = res["summary"]
    print(json.dumps(s, indent=2))
    print(f"\n[HEADLINE] per-patient Dice mean={s['dice_mean']:.3f} median={s['dice_median']:.3f} "
          f"(paper 0.82 / 0.85)")


def _grouped_predictions_all_folds(cfg, df, folds, device):
    """Run inference over every fold's val patients; return one grouped dict.

    Reuses the evaluation machinery (leakage-safe: each patient is predicted only
    by the fold that held it out). Slices come back in physical (slice-index)
    order, so they stack directly into per-patient volumes for shape features.
    """
    from .evaluate import _load_model, predict_patients

    in_channels = cfg["data"]["in_channels"]
    channels = cfg["train"]["channels"]
    batchnorm = cfg["train"]["batchnorm"]
    ckpt_for_fold = _ckpt_for_fold(cfg)
    grouped = {}
    for fold in folds:
        model = _load_model(ckpt_for_fold(fold["fold"]), in_channels, channels, batchnorm, device)
        g = predict_patients(model, df, fold["val"], device, in_channels=in_channels)
        grouped.update(g)
        del model
        import torch
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return grouped


def cmd_deployment(args):
    """R8: parameter count, on-disk size, peak inference VRAM, GPU + CPU latency.

    Inference-only (~0 GPU) benchmark of the trained model, written to
    reports/deployment.json and folded into metrics.json / comparison.md by report.
    """
    import torch

    from .models.unet import build_unet
    from .quantize import benchmark_model
    from .report import write_metrics_json

    cfg = _cfg(args)
    device = get_device()
    ckpt = os.path.join(cfg["paths"]["ckpt_dir"], "fold0_best.pt")
    if not os.path.exists(ckpt):
        sys.exit(f"{ckpt} missing; run `train` first.")
    model = build_unet(cfg["train"]["channels"], in_channels=cfg["data"]["in_channels"],
                       batchnorm=cfg["train"]["batchnorm"]).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device)["model"])
    model.eval()
    bench = benchmark_model(model, cfg["data"]["in_channels"], device)
    bench["checkpoint"] = "fold0_best.pt"
    bench["device"] = device.type
    os.makedirs(cfg["paths"]["reports"], exist_ok=True)
    write_metrics_json(os.path.join(cfg["paths"]["reports"], "deployment.json"), bench)
    print(json.dumps(bench, indent=2))


def cmd_shape_features(args):
    """Section 7 step 1: extract BEVR / angular-std / margin-fluctuation per patient.

    Runs inference (leakage-safe, one prediction per patient), reconstructs each
    patient's 3D mask volume for BOTH predicted and ground-truth masks, extracts
    the three shape features, joins the genomic + demographic labels from
    data.csv, and writes reports/shape_features_{predicted,ground_truth}.csv.
    CPU-only statistics downstream; this step is inference-only (~0 training).
    """
    from .radiogenomics import (DEMOGRAPHIC_COLUMNS, SUBTYPE_COLUMNS,
                                 features_to_frame, load_genomic_labels)
    from .shape_features import features_for_patients

    cfg = _cfg(args)
    df = _manifest(cfg)
    folds = _load_folds(cfg)
    device = get_device()
    reports = cfg["paths"]["reports"]
    os.makedirs(reports, exist_ok=True)

    grouped = _grouped_predictions_all_folds(cfg, df, folds, device)
    labels = load_genomic_labels(cfg["paths"]["datapath"])
    label_cols = [c for c in (SUBTYPE_COLUMNS + DEMOGRAPHIC_COLUMNS)
                  if not labels.empty and c in labels.columns]

    for which, tag in (("preds", "predicted"), ("gts", "ground_truth")):
        feats = features_for_patients(grouped, which=which)
        fdf = features_to_frame(feats)
        if label_cols:
            fdf = fdf.join(labels[label_cols], how="left")
        out = os.path.join(reports, f"shape_features_{tag}.csv")
        fdf.to_csv(out, index_label="patient")
        n_with = int(fdf[["bevr"]].dropna().shape[0]) if "bevr" in fdf else 0
        print(f"[ok] {tag}: shape features for {n_with}/{len(fdf)} patients -> {out}")
    if labels.empty:
        print("[warn] data.csv not found next to the dataset; genomic labels were "
              "not joined. radiogenomics will be skipped.")


def cmd_radiogenomics(args):
    """Section 7 steps 2-6: Fisher tests + discrimination AUC, predicted vs GT."""
    from .radiogenomics import (SUBTYPE_COLUMNS, load_genomic_labels,
                                run_radiogenomics)
    from .report import write_metrics_json
    from .shape_features import FEATURE_NAMES

    cfg = _cfg(args)
    reports = cfg["paths"]["reports"]
    labels = load_genomic_labels(cfg["paths"]["datapath"])
    if labels.empty:
        sys.exit("data.csv not found; cannot run radiogenomics. Place data.csv next "
                 "to the dataset folder (it ships with the Kaggle set).")

    def _load_feats(tag):
        p = os.path.join(reports, f"shape_features_{tag}.csv")
        if not os.path.exists(p):
            sys.exit(f"{p} missing; run `shape-features` first.")
        d = pd.read_csv(p).set_index("patient")
        return {idx: {f: row[f] for f in FEATURE_NAMES if f in row}
                for idx, row in d.iterrows()}

    pred = _load_feats("predicted")
    gt = _load_feats("ground_truth")
    subs = [c for c in SUBTYPE_COLUMNS if c in labels.columns]
    res = run_radiogenomics(pred, gt, labels, feature_names=FEATURE_NAMES, subtype_cols=subs)
    write_metrics_json(os.path.join(reports, "radiogenomics.json"), res)

    for mask in ("predicted", "ground_truth"):
        auc = res["masks"][mask]["discrimination_auc"] or {}
        best = res["masks"][mask]["associations"]["best_per_pair"]
        primary = next((r for r in best if r["feature"] == "bevr"
                        and r["subtype"] == "RNASeqCluster"), None)
        print(f"\n[{mask}] discrimination AUC (RNASeq {auc.get('target_cluster', '?')} "
              f"vs rest, inv-BEVR): {auc.get('auc')}")
        if primary:
            print(f"[{mask}] RNASeq × BEVR strongest cluster p_bonferroni="
                  f"{primary['p_bonferroni']:.2e} (raw {primary['p_raw']:.2e})")
    print("\n[paper] RNASeq×BEVR p<0.0002; RNASeq×margin p<0.005; AUC 0.80 model / 0.78 manual")


def cmd_figures(args):
    """Section 6: generate F1-F7 from already-computed results (~0 GPU)."""
    from .figures import make_all_figures

    cfg = _cfg(args)
    made = make_all_figures(cfg)
    print(f"[ok] wrote {len(made)} figures -> {cfg['paths']['figures']}")


def cmd_run_robustness(args):
    from .report import fig_robustness, write_metrics_json
    from .robustness import run_robustness

    cfg = _cfg(args)
    df = _manifest(cfg)
    folds = _load_folds(cfg)
    device = get_device()
    res = run_robustness(df, folds, _ckpt_for_fold(cfg), device,
                         cfg["data"]["in_channels"], cfg["train"]["channels"], cfg["train"]["batchnorm"])
    write_metrics_json(os.path.join(cfg["paths"]["reports"], "robustness.json"), res)
    os.makedirs(cfg["paths"]["figures"], exist_ok=True)
    fig_robustness(res, os.path.join(cfg["paths"]["figures"], "robustness.png"))
    print(json.dumps({k: {"dice_mean": v["dice_mean"], "delta": v["dice_delta_vs_clean"]}
                      for k, v in res.items()}, indent=2))


def cmd_quantize(args):
    from .models.unet import build_unet
    from .quantize import quantization_report
    from .report import write_metrics_json

    cfg = _cfg(args)
    df = _manifest(cfg)
    folds = _load_folds(cfg)
    device = get_device()
    ckpt = os.path.join(cfg["paths"]["ckpt_dir"], "fold0_best.pt")
    res = quantization_report(build_unet, ckpt, df, folds, device,
                              in_channels=cfg["data"]["in_channels"], channels=cfg["train"]["channels"],
                              batchnorm=cfg["train"]["batchnorm"])
    write_metrics_json(os.path.join(cfg["paths"]["reports"], "quantization.json"), res)
    print(json.dumps(res, indent=2))


def cmd_distill(args):
    from .distill import distill_student
    from .evaluate import patient_metrics_from_grouped, predict_patients
    from .metrics import aggregate_patient_metrics
    from .models.unet import build_unet
    from .quantize import benchmark_model
    from .report import write_metrics_json

    import torch

    cfg = _cfg(args)
    df = _manifest(cfg)
    folds = _load_folds(cfg)
    device = get_device()
    d = cfg.get("distill", {})
    fold0 = folds[0]
    teacher = os.path.join(cfg["paths"]["ckpt_dir"], "fold0_best.pt")
    student_ckpt = os.path.join(cfg["paths"]["ckpt_dir"], "student.pt")

    hist = distill_student(
        df, fold0["train"], teacher, d.get("student_channels", "micro"), student_ckpt, device,
        in_channels=cfg["data"]["in_channels"], teacher_channels=cfg["train"]["channels"],
        batchnorm=cfg["train"]["batchnorm"], epochs=args.epochs or d.get("epochs", 30),
        temperature=d.get("temperature", 2.0), alpha=d.get("alpha", 0.5), seed=cfg["seed"],
    )
    student = build_unet(d.get("student_channels", "micro"), in_channels=cfg["data"]["in_channels"],
                         batchnorm=cfg["train"]["batchnorm"]).to(device)
    student.load_state_dict(torch.load(student_ckpt, map_location=device)["model"])
    student.eval()
    grouped = predict_patients(student, df, fold0["val"], device, in_channels=cfg["data"]["in_channels"])
    summ = aggregate_patient_metrics(patient_metrics_from_grouped(grouped, with_hd95=False))
    bench = benchmark_model(student, cfg["data"]["in_channels"], device)
    out = {"history": hist, "student_dice": summ, "deployment": bench}
    write_metrics_json(os.path.join(cfg["paths"]["reports"], "distill.json"), out)
    print(json.dumps({"student_dice_mean": summ["dice_mean"], "deployment": bench}, indent=2))


def cmd_cross_institution(args):
    from .cross_institution import run_leave_one_institution_out
    from .report import write_metrics_json

    cfg = _cfg(args)
    df = _manifest(cfg)
    device = get_device()
    tc = _train_config(cfg, args)
    res = run_leave_one_institution_out(df, tc, cfg["paths"]["ckpt_dir"], device)
    write_metrics_json(os.path.join(cfg["paths"]["reports"], "cross_institution.json"), res)
    print(json.dumps(res, indent=2))


def cmd_run_nnunet(args):
    print("nnU-Net comparison is a separate optional module. See nnunet/README.md "
          "for the patient-level conversion + splits_final.json + reduced-epoch trainer steps.")


def cmd_report(args):
    from .report import (fig_dice_by_institution, fig_dice_distribution, write_comparison_md,
                         write_metrics_json)

    cfg = _cfg(args)
    reports = cfg["paths"]["reports"]
    figures = cfg["paths"]["figures"]
    os.makedirs(figures, exist_ok=True)

    with open(os.path.join(reports, "eval_metrics.json")) as fh:
        ev = json.load(fh)
    summary = ev["summary"]
    per_patient = ev["per_patient"]

    meta = {"preset": cfg["splits"]["preset"], "seed": cfg["seed"],
            "in_channels": cfg["data"]["in_channels"], "channels": cfg["train"]["channels"],
            "batchnorm": cfg["train"]["batchnorm"], "epochs": cfg["train"]["epochs"]}

    # Radiogenomics (Section 7) is the load-bearing "full replication" piece; if
    # present it upgrades the comparison from partial -> full.
    radiogenomics = None
    rg_path = os.path.join(reports, "radiogenomics.json")
    if os.path.exists(rg_path):
        with open(rg_path) as fh:
            radiogenomics = json.load(fh)

    deployment = None
    dep_path = os.path.join(reports, "deployment.json")
    if os.path.exists(dep_path):
        with open(dep_path) as fh:
            deployment = json.load(fh)

    bundle = {"meta": meta, "headline": summary}
    if radiogenomics:
        bundle["radiogenomics"] = radiogenomics
    if deployment:
        bundle["deployment"] = deployment
    # Optional Section-12 extras are folded in only if their JSON already exists.
    for name in ["robustness", "quantization", "distill", "cross_institution"]:
        p = os.path.join(reports, f"{name}.json")
        if os.path.exists(p):
            with open(p) as fh:
                bundle[name] = json.load(fh)

    write_metrics_json(os.path.join(reports, "metrics.json"), bundle)
    write_comparison_md(os.path.join(reports, "comparison.md"), summary, meta,
                        radiogenomics=radiogenomics, deployment=deployment)
    fig_dice_distribution(per_patient, os.path.join(figures, "dice_distribution.png"))
    fig_dice_by_institution(per_patient, os.path.join(figures, "dice_by_institution.png"))
    kind = "FULL" if radiogenomics else "partial"
    print(f"[ok] wrote {reports}/metrics.json, {reports}/comparison.md ({kind} replication).")
    print(f"[HEADLINE] mean={summary['dice_mean']:.3f} median={summary['dice_median']:.3f} "
          f"vs paper 0.82 / 0.85")


def cmd_smoke(args):
    """2-patient end-to-end wiring check (Section 5): prepare->train->eval->report < ~5 min."""
    from .data.splits import make_folds, save_splits
    from .evaluate import evaluate_cv
    from .report import write_comparison_md
    from .train import TrainConfig, train_fold

    cfg = _cfg(args)
    set_seed(cfg["seed"])
    df = _manifest(cfg)
    device = get_device()
    patients = sorted(df["patient"].unique())[: args.n_patients]
    df = df[df["patient"].isin(patients)].reset_index(drop=True)
    folds = [{"fold": 0, "train": patients[:1], "val": patients[1:]}]
    tc = TrainConfig(epochs=args.epochs, batch_size=4, in_channels=cfg["data"]["in_channels"],
                     channels="micro", num_workers=0, augment=False)
    ckpt = os.path.join(cfg["paths"]["ckpt_dir"], "smoke_fold0.pt")
    train_fold(df, folds[0]["train"], folds[0]["val"], tc, ckpt, device=device)
    res = evaluate_cv(df, folds, lambda i: ckpt.replace(".pt", "_best.pt"), device,
                      in_channels=cfg["data"]["in_channels"], channels="micro",
                      batchnorm=True, with_hd95=False)
    print("[smoke] per-patient dice:", res["summary"]["dice_mean"])
    print("[smoke] OK — wiring verified end to end.")


def build_parser():
    p = argparse.ArgumentParser("lgg", description="TCGA-LGG segmentation reproduction (Buda 2019)")
    p.add_argument("--config", default="config.yaml")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("prepare-data"); sp.add_argument("--datapath", default=None); sp.set_defaults(fn=cmd_prepare_data)
    sp = sub.add_parser("make-splits"); sp.add_argument("--preset", default=None); sp.set_defaults(fn=cmd_make_splits)
    sp = sub.add_parser("train"); sp.add_argument("--fold", type=int, default=None)
    sp.add_argument("--epochs", type=int, default=None); sp.add_argument("--resume", action="store_true")
    sp.set_defaults(fn=cmd_train)
    sp = sub.add_parser("evaluate"); sp.add_argument("--no-hd95", action="store_true"); sp.set_defaults(fn=cmd_evaluate)
    sp = sub.add_parser("deployment"); sp.set_defaults(fn=cmd_deployment)
    sp = sub.add_parser("shape-features"); sp.set_defaults(fn=cmd_shape_features)
    sp = sub.add_parser("radiogenomics"); sp.set_defaults(fn=cmd_radiogenomics)
    sp = sub.add_parser("figures"); sp.set_defaults(fn=cmd_figures)
    sp = sub.add_parser("run-robustness"); sp.set_defaults(fn=cmd_run_robustness)
    sp = sub.add_parser("quantize"); sp.set_defaults(fn=cmd_quantize)
    sp = sub.add_parser("distill"); sp.add_argument("--epochs", type=int, default=None); sp.set_defaults(fn=cmd_distill)
    sp = sub.add_parser("cross-institution"); sp.add_argument("--epochs", type=int, default=None); sp.set_defaults(fn=cmd_cross_institution)
    sp = sub.add_parser("run-nnunet"); sp.set_defaults(fn=cmd_run_nnunet)
    sp = sub.add_parser("report"); sp.set_defaults(fn=cmd_report)
    sp = sub.add_parser("smoke"); sp.add_argument("--n-patients", type=int, default=2)
    sp.add_argument("--epochs", type=int, default=2); sp.set_defaults(fn=cmd_smoke)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
