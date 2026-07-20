"""Inference-only robustness sweep (Section 6).

Re-runs the trained model on perturbed inputs and logs Dice degradation versus
clean. Perturbations act on the normalized (C, H, W) slice just before the
forward pass — training is never touched. Included: Gaussian noise, Rician
noise, downsample->upsample (low-resolution), a simulated smooth bias field, and
randomly dropped (zeroed) slices.
"""

from __future__ import annotations

from typing import Callable, Dict, List

import numpy as np


def _rng(seed: int) -> np.random.RandomState:
    return np.random.RandomState(seed)


def gaussian_noise(sigma: float, seed: int = 0) -> Callable:
    def f(x: np.ndarray) -> np.ndarray:
        r = _rng(seed + int(x.sum() * 1e3) % 9973)
        return x + r.normal(0, sigma, size=x.shape).astype(np.float32)
    return f


def rician_noise(sigma: float, seed: int = 0) -> Callable:
    def f(x: np.ndarray) -> np.ndarray:
        r = _rng(seed + int(abs(x.sum()) * 1e3) % 9973)
        n1 = r.normal(0, sigma, size=x.shape)
        n2 = r.normal(0, sigma, size=x.shape)
        return np.sqrt((x + n1) ** 2 + n2 ** 2).astype(np.float32)
    return f


def downsample(factor: float) -> Callable:
    """Downsample by ``factor`` then upsample back (bilinear) to blur detail."""
    def f(x: np.ndarray) -> np.ndarray:
        from scipy.ndimage import zoom

        out = np.empty_like(x)
        for c in range(x.shape[0]):
            small = zoom(x[c], factor, order=1)
            out[c] = zoom(small, (x.shape[1] / small.shape[0], x.shape[2] / small.shape[1]), order=1)
        return out.astype(np.float32)
    return f


def bias_field(strength: float = 0.3) -> Callable:
    """Multiply by a smooth low-frequency field to mimic MRI intensity inhomogeneity."""
    def f(x: np.ndarray) -> np.ndarray:
        h, w = x.shape[1], x.shape[2]
        yy, xx = np.meshgrid(np.linspace(-1, 1, h), np.linspace(-1, 1, w), indexing="ij")
        field = 1.0 + strength * (np.cos(1.5 * xx + 0.5) * np.sin(1.5 * yy - 0.3))
        return (x * field[None]).astype(np.float32)
    return f


def dropped_slices(prob: float, seed: int = 0) -> Callable:
    """Zero out whole slices with probability ``prob`` (missing-slice simulation)."""
    def f(x: np.ndarray) -> np.ndarray:
        r = _rng(seed + int(abs(x.sum()) * 1e3) % 9973)
        if r.rand() < prob:
            return np.zeros_like(x)
        return x
    return f


def default_conditions() -> Dict[str, Callable]:
    return {
        "clean": None,
        "gaussian_sigma0.1": gaussian_noise(0.1),
        "gaussian_sigma0.25": gaussian_noise(0.25),
        "rician_sigma0.1": rician_noise(0.1),
        "downsample_0.5x": downsample(0.5),
        "downsample_0.25x": downsample(0.25),
        "bias_field_0.3": bias_field(0.3),
        "dropped_slices_0.2": dropped_slices(0.2),
    }


def run_robustness(manifest, folds, ckpt_for_fold, device, in_channels, channels, batchnorm) -> dict:
    """Evaluate every condition; return {condition: patient-summary} + degradation."""
    from .evaluate import evaluate_cv

    results = {}
    for name, perturb in default_conditions().items():
        res = evaluate_cv(
            manifest, folds, ckpt_for_fold, device,
            in_channels=in_channels, channels=channels, batchnorm=batchnorm,
            with_hd95=False, perturb=perturb,
        )
        results[name] = res["summary"]
    clean = results["clean"]["dice_mean"]
    for name in results:
        results[name]["dice_delta_vs_clean"] = round(results[name]["dice_mean"] - clean, 4)
    return results
