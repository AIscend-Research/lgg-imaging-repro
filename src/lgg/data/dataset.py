"""Slice dataset + patient grouping for evaluation (R2, R5, R9).

Design choices tied to the brief:
  * ALL slices are kept — empty (tumor-free) slices are never discarded (R5).
  * Channels are explicit (R9): ``in_channels=3`` uses the public benchmark's
    pre/FLAIR/post ordering; ``in_channels=1`` is FLAIR-only (the green channel,
    index 1, of the RGB .tif).
  * The dataset exposes each slice's patient id so the evaluator can regroup
    slices into per-patient volumes for volume-aggregated Dice (R2).
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

# FLAIR is the middle channel of the pre/FLAIR/post RGB .tif.
FLAIR_CHANNEL = 1


def _load_image(path: str, in_channels: int) -> np.ndarray:
    """Return a (C, H, W) float32 array in [0, 1]. C == in_channels."""
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0  # (H, W, 3)
    if in_channels == 1:
        arr = arr[:, :, FLAIR_CHANNEL:FLAIR_CHANNEL + 1]
    elif in_channels == 3:
        pass
    else:
        raise ValueError(f"in_channels must be 1 or 3, got {in_channels}")
    return np.transpose(arr, (2, 0, 1))  # (C, H, W)


def _load_mask(path: str) -> np.ndarray:
    m = np.asarray(Image.open(path).convert("L"), dtype=np.float32)
    return (m > 0).astype(np.float32)[None]  # (1, H, W)


class SliceDataset(Dataset):
    """Per-slice dataset built from a manifest subset.

    Augmentation is intentionally light (paper-faithful): optional horizontal /
    vertical flips and small rotations only, applied identically to image + mask.
    """

    def __init__(
        self,
        manifest: pd.DataFrame,
        in_channels: int = 3,
        augment: bool = False,
        normalize: bool = True,
    ):
        self.df = manifest.reset_index(drop=True)
        self.in_channels = in_channels
        self.augment = augment
        self.normalize = normalize

    def __len__(self) -> int:
        return len(self.df)

    def _augment(self, img: np.ndarray, mask: np.ndarray):
        # img: (C,H,W), mask: (1,H,W). Flips/rot90 are label-preserving & cheap.
        if np.random.rand() < 0.5:
            img = img[:, :, ::-1].copy()
            mask = mask[:, :, ::-1].copy()
        if np.random.rand() < 0.5:
            img = img[:, ::-1, :].copy()
            mask = mask[:, ::-1, :].copy()
        k = np.random.randint(0, 4)
        if k:
            img = np.rot90(img, k, axes=(1, 2)).copy()
            mask = np.rot90(mask, k, axes=(1, 2)).copy()
        return img, mask

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img = _load_image(row["image_path"], self.in_channels)
        mask = _load_mask(row["mask_path"])
        if self.augment:
            img, mask = self._augment(img, mask)
        if self.normalize:
            # Per-image z-score over non-zero (brain) pixels, per channel.
            for c in range(img.shape[0]):
                ch = img[c]
                brain = ch[ch > 0]
                if brain.size > 0:
                    img[c] = (ch - brain.mean()) / (brain.std() + 1e-6)
        return (
            torch.from_numpy(img).float(),
            torch.from_numpy(mask).float(),
            idx,  # row index -> lets the evaluator recover patient id
        )


def subset_for_patients(manifest: pd.DataFrame, patients: List[str]) -> pd.DataFrame:
    return manifest[manifest["patient"].isin(set(patients))].reset_index(drop=True)


def patient_of_row(manifest: pd.DataFrame, idx: int) -> str:
    return manifest.iloc[idx]["patient"]
