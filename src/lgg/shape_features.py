"""Tumor shape features for the radiogenomics reproduction (Section 7).

The paper (Buda et al. 2019) associates *genomic subtype* with three automatically
extracted **shape features** of the FLAIR-abnormality mask. Reproducing that
association — for masks the U-Net predicted, and for the manual ground truth — is
the load-bearing "full replication" piece, and it is CPU-only.

The three features and their reproduction here:

  * ``bevr`` — **bounding ellipsoid volume ratio**: tumor volume divided by the
    volume of its minimum enclosing ellipsoid. A compact, sphere-like tumor has a
    ratio near 1; an irregular / elongated tumor fills less of its ellipsoid, so
    the ratio drops. Computed on the 3D volume. The paper's discrimination check
    uses the *inverse* of this ratio, so we also expose ``inv_bevr = 1/bevr``.
  * ``angular_std`` — **angular standard deviation**: the normalized standard
    deviation of the boundary's radial distance from the tumor centroid, sampled
    over angle. Measures overall margin irregularity.
  * ``margin_fluctuation`` — **margin fluctuation**: the normalized standard
    deviation of the *high-frequency* part of that radial signature (the residual
    after removing a low-order smooth fit), capturing fine boundary roughness.

The precise definitions in the paper trace to its reference [6]; where the text is
ambiguous we implement the most faithful reading and record it under "Deviations"
in the README. The objective is to reproduce the **direction and significance** of
the genomic associations, not bit-identical feature values.

All features are computed from the binary mask alone (no image intensities), so
the same code runs unchanged on predicted and ground-truth masks. Voxels are
treated as isotropic unit cubes: the Kaggle dataset does not ship per-slice
spacing, and every feature used here is a dimensionless *ratio* or normalized
dispersion, hence scale-invariant to an isotropic voxel size (recorded as a
deviation).
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np


# --------------------------------------------------------------- 3D feature ---

def bounding_ellipsoid_volume_ratio(volume: np.ndarray) -> Optional[float]:
    """Tumor volume / volume of its minimum enclosing ellipsoid (PCA reading).

    ``volume`` is a (Z, H, W) binary array. We take the tumor voxel coordinates,
    center them, rotate onto their principal axes (eigenvectors of the coordinate
    covariance), and take the ellipsoid whose semi-axes are the maximum absolute
    projection onto each axis — the smallest axis-aligned ellipsoid, in the PCA
    frame, that encloses every voxel. Returns None if there is no tumor.
    """
    pts = np.ascontiguousarray(np.argwhere(volume > 0), dtype=np.float64)
    if pts.shape[0] < 4:  # need enough voxels to define an ellipsoid
        return None
    tumor_vol = float(pts.shape[0])  # one voxel = one unit cube
    centered = pts - pts.mean(axis=0)
    cov = np.cov(centered, rowvar=False)
    # Principal axes (eigenvectors of the coordinate covariance); project onto them.
    _, eigvecs = np.linalg.eigh(cov)
    # errstate guards a spurious "divide by zero in matmul" warning seen only on
    # the macOS Accelerate BLAS backend; the projection itself is exact.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        proj = np.ascontiguousarray(centered) @ np.ascontiguousarray(eigvecs)
    radii = np.abs(proj).max(axis=0)
    radii = np.maximum(radii, 0.5)  # a flat (single-slice) tumor still has depth
    # Scale the axis-aligned (PCA-frame) ellipsoid so it truly ENCLOSES every
    # voxel: find the largest normalized radius and grow the ellipsoid to it.
    # This keeps BEVR in (0, 1], as a bounding-ellipsoid volume ratio should be.
    norm = np.sqrt(((proj / radii) ** 2).sum(axis=1)).max()
    if norm > 1.0:
        radii = radii * norm
    ell_vol = (4.0 / 3.0) * np.pi * float(np.prod(radii))
    if ell_vol <= 0:
        return None
    return min(1.0, tumor_vol / ell_vol)


# ------------------------------------------------------ 2D boundary features --

def _largest_slice(volume: np.ndarray) -> Optional[np.ndarray]:
    """Return the 2D slice with the most tumor pixels, or None if empty."""
    areas = volume.reshape(volume.shape[0], -1).sum(axis=1)
    if areas.max() == 0:
        return None
    return volume[int(np.argmax(areas))]


def _radial_signature(mask2d: np.ndarray, n_angles: int = 180) -> Optional[np.ndarray]:
    """Boundary radius r(theta) sampled at ``n_angles`` angles about the centroid.

    For each angle we take the farthest tumor pixel in that direction, giving the
    outer margin's distance from the centroid as a function of angle.
    """
    ys, xs = np.nonzero(mask2d)
    if ys.size < 8:
        return None
    cy, cx = ys.mean(), xs.mean()
    dy, dx = ys - cy, xs - cx
    ang = np.arctan2(dy, dx)  # [-pi, pi]
    rad = np.hypot(dy, dx)
    bins = np.linspace(-np.pi, np.pi, n_angles + 1)
    idx = np.clip(np.digitize(ang, bins) - 1, 0, n_angles - 1)
    sig = np.zeros(n_angles)
    for a in range(n_angles):
        r = rad[idx == a]
        sig[a] = r.max() if r.size else np.nan
    # fill empty angular bins by circular interpolation so the signature is dense.
    if np.isnan(sig).any():
        good = ~np.isnan(sig)
        if good.sum() < 4:
            return None
        xp = np.nonzero(good)[0]
        sig = np.interp(np.arange(n_angles), xp, sig[good], period=n_angles)
    return sig


def angular_standard_deviation(mask2d: np.ndarray) -> Optional[float]:
    """Normalized std of the radial boundary signature (margin irregularity)."""
    sig = _radial_signature(mask2d)
    if sig is None:
        return None
    mean = sig.mean()
    if mean <= 0:
        return None
    return float(sig.std() / mean)


def margin_fluctuation(mask2d: np.ndarray, smooth_harmonics: int = 5) -> Optional[float]:
    """Normalized std of the high-frequency residual of the radial signature.

    We low-pass the periodic signature by keeping only the first
    ``smooth_harmonics`` Fourier components, then measure the dispersion of what
    is left — the fine, high-frequency boundary roughness.
    """
    sig = _radial_signature(mask2d)
    if sig is None:
        return None
    mean = sig.mean()
    if mean <= 0:
        return None
    fft = np.fft.rfft(sig)
    fft[smooth_harmonics + 1:] = 0.0
    smooth = np.fft.irfft(fft, n=sig.shape[0])
    residual = sig - smooth
    return float(residual.std() / mean)


# ------------------------------------------------------------ orchestration ---

FEATURE_NAMES = ["bevr", "inv_bevr", "angular_std", "margin_fluctuation"]


def features_from_volume(volume: np.ndarray) -> Dict[str, Optional[float]]:
    """All shape features for one patient's (Z, H, W) binary mask volume."""
    bevr = bounding_ellipsoid_volume_ratio(volume)
    sl = _largest_slice(volume)
    feats: Dict[str, Optional[float]] = {
        "bevr": bevr,
        "inv_bevr": (1.0 / bevr) if (bevr and bevr > 0) else None,
        "angular_std": angular_standard_deviation(sl) if sl is not None else None,
        "margin_fluctuation": margin_fluctuation(sl) if sl is not None else None,
    }
    return feats


def volume_from_slices(slices: List[np.ndarray]) -> np.ndarray:
    """Stack an ordered list of 2D binary slices into a (Z, H, W) volume."""
    return np.stack([s.astype(np.uint8) for s in slices], axis=0)


def features_for_patients(grouped: Dict[str, dict], which: str = "preds") -> Dict[str, dict]:
    """Extract shape features for every patient from grouped predictions.

    ``grouped`` is the structure returned by ``evaluate.predict_patients``:
    ``{patient: {'preds': [...], 'gts': [...], ...}}`` with per-patient slice
    lists in physical (slice-index) order. ``which`` selects ``'preds'`` or
    ``'gts'``. Patients with no tumor in the chosen mask get all-None features.
    """
    out: Dict[str, dict] = {}
    for patient, d in grouped.items():
        masks = d.get(which)
        if not masks:
            continue
        out[patient] = features_from_volume(volume_from_slices(masks))
    return out
