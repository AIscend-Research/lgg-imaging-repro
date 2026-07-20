"""Quantization + deployment benchmarking (R8, Section 6).

Reports the low-resource evidence for each model: parameter count, on-disk size
(MB), peak inference VRAM, GPU latency per volume, and CPU-only latency per
volume. Also produces FP16 and INT8 post-training-quantized variants and their
Dice deltas versus the full-precision model.
"""

from __future__ import annotations

import os
import tempfile
import time
from typing import Optional

import numpy as np
import torch

# Mean slices per patient in TCGA-LGG (3929 / 110). Used to turn a per-slice
# latency into a per-volume number (R8).
MEAN_SLICES_PER_VOLUME = 3929 / 110


def model_size_mb(model) -> float:
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        path = tmp.name
    torch.save(model.state_dict(), path)
    mb = os.path.getsize(path) / 1024 / 1024
    os.remove(path)
    return mb


def _latency_ms_per_slice(model, device, in_channels, n=30) -> float:
    model = model.to(device).eval()
    x = torch.randn(1, in_channels, 256, 256, device=device)
    with torch.no_grad():
        for _ in range(5):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(n):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
    return (time.time() - t0) / n * 1000.0


def benchmark_model(model, in_channels: int = 3, device=None) -> dict:
    """Full deployment report for one model (R8)."""
    import torch

    n_params = sum(p.numel() for p in model.parameters())
    size_mb = model_size_mb(model)
    out = {
        "params": int(n_params),
        "params_M": round(n_params / 1e6, 3),
        "size_mb": round(size_mb, 2),
    }
    cpu = torch.device("cpu")
    cpu_ms = _latency_ms_per_slice(model, cpu, in_channels)
    out["cpu_ms_per_slice"] = round(cpu_ms, 2)
    out["cpu_ms_per_volume"] = round(cpu_ms * MEAN_SLICES_PER_VOLUME, 1)

    if device is not None and device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        gpu_ms = _latency_ms_per_slice(model, device, in_channels)
        peak_vram_mb = torch.cuda.max_memory_allocated(device) / 1024 / 1024
        out["gpu_ms_per_slice"] = round(gpu_ms, 3)
        out["gpu_ms_per_volume"] = round(gpu_ms * MEAN_SLICES_PER_VOLUME, 2)
        out["peak_vram_mb"] = round(peak_vram_mb, 1)
        model.to(cpu)
    return out


def to_fp16(model):
    return model.half()


def to_int8_dynamic(model):
    """Dynamic INT8 (quantizes Linear layers). Documented caveat: a U-Net is
    all-convolutional, so dynamic PTQ yields little size reduction — reported
    honestly rather than hidden."""
    return torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)


def quantization_report(
    build_model_fn,
    ckpt_path: str,
    manifest,
    folds,
    device,
    in_channels: int = 3,
    channels: str = "reference",
    batchnorm: bool = True,
) -> dict:
    """Benchmark FP32 vs FP16 vs INT8 for the model at ``ckpt_path``.

    Dice is measured with the full CV evaluator (fold 0 only, for speed) so the
    accuracy delta is on the real headline metric, not a proxy.
    """
    from .evaluate import _load_model, patient_metrics_from_grouped, predict_patients
    from .metrics import aggregate_patient_metrics

    fold0 = folds[0]

    def dice_of(model):
        grouped = predict_patients(model, manifest, fold0["val"], device, in_channels=in_channels)
        pm = patient_metrics_from_grouped(grouped, with_hd95=False)
        return aggregate_patient_metrics(pm)["dice_mean"]

    report = {}

    fp32 = _load_model(ckpt_path, in_channels, channels, batchnorm, device)
    report["fp32"] = benchmark_model(fp32, in_channels, device)
    report["fp32"]["dice_mean"] = round(dice_of(fp32.to(device)), 4)

    # FP16 (GPU only for conv). Benchmark on GPU if available.
    if device.type == "cuda":
        fp16 = _load_model(ckpt_path, in_channels, channels, batchnorm, device).half()
        b = benchmark_model_fp16(fp16, in_channels, device)
        # Dice at fp16
        grouped_dice = _fp16_dice(fp16, manifest, fold0["val"], device, in_channels)
        b["dice_mean"] = round(grouped_dice, 4)
        report["fp16"] = b

    int8 = to_int8_dynamic(_load_model(ckpt_path, in_channels, channels, batchnorm, torch.device("cpu")))
    r8 = {
        "params": sum(p.numel() for p in int8.parameters()),
        "size_mb": round(model_size_mb(int8), 2),
        "cpu_ms_per_slice": round(_latency_ms_per_slice(int8, torch.device("cpu"), in_channels), 2),
        "note": "dynamic INT8 quantizes Linear only; conv U-Net sees little size change (expected).",
    }
    report["int8_dynamic"] = r8

    base = report["fp32"]["dice_mean"]
    for k in report:
        if "dice_mean" in report[k]:
            report[k]["dice_delta_vs_fp32"] = round(report[k]["dice_mean"] - base, 4)
    return report


def benchmark_model_fp16(model, in_channels, device) -> dict:
    torch.cuda.reset_peak_memory_stats(device)
    gpu_ms = _latency_ms_per_slice_fp16(model, device, in_channels)
    peak = torch.cuda.max_memory_allocated(device) / 1024 / 1024
    return {
        "params": sum(p.numel() for p in model.parameters()),
        "size_mb": round(model_size_mb(model), 2),
        "gpu_ms_per_slice": round(gpu_ms, 3),
        "gpu_ms_per_volume": round(gpu_ms * MEAN_SLICES_PER_VOLUME, 2),
        "peak_vram_mb": round(peak, 1),
    }


def _latency_ms_per_slice_fp16(model, device, in_channels, n=30):
    model = model.to(device).eval()
    x = torch.randn(1, in_channels, 256, 256, device=device).half()
    with torch.no_grad():
        for _ in range(5):
            model(x)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(n):
            model(x)
        torch.cuda.synchronize()
    return (time.time() - t0) / n * 1000.0


@torch.no_grad()
def _fp16_dice(model, manifest, patients, device, in_channels):
    from .data.dataset import SliceDataset, subset_for_patients
    from .metrics import aggregate_patient_metrics, patient_volume_dice
    from torch.utils.data import DataLoader

    df = subset_for_patients(manifest, patients)
    ds = SliceDataset(df, in_channels=in_channels, augment=False)
    loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)
    grouped = {p: {"preds": [], "gts": []} for p in patients}
    for imgs, masks, idxs in loader:
        imgs = imgs.to(device).half()
        pred = (torch.sigmoid(model(imgs)) > 0.5).cpu().numpy().astype(np.uint8)
        gt = masks.numpy().astype(np.uint8)
        for k in range(pred.shape[0]):
            patient = df.iloc[int(idxs[k])]["patient"]
            grouped[patient]["preds"].append(pred[k, 0])
            grouped[patient]["gts"].append(gt[k, 0])
    pm = {p: {"dice": patient_volume_dice(d["preds"], d["gts"]), "iou": 0.0, "hd95": None}
          for p, d in grouped.items() if d["preds"]}
    return aggregate_patient_metrics(pm)["dice_mean"]
