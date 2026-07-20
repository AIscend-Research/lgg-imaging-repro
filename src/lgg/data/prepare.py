"""Build a slice-level manifest from the Kaggle LGG folder (R1 provenance).

The Kaggle dataset (``mateuszbuda/lgg-mri-segmentation``) is laid out as::

    kaggle_3m/
        TCGA_CS_4941_19960909/
            TCGA_CS_4941_19960909_1.tif        <- 3-channel image (pre / FLAIR / post)
            TCGA_CS_4941_19960909_1_mask.tif   <- binary FLAIR-abnormality mask
            ...

Each patient folder is ONE patient. The 3 image channels are, in order,
pre-contrast / FLAIR / post-contrast (this is the channel convention used by the
public benchmark and recorded in the README, R9). We do not convert to NIfTI —
the reproduction operates directly on the 2D slices, keeping all empty slices
(R5). The manifest is the single source of truth downstream.
"""

from __future__ import annotations

import os
from typing import List

import numpy as np
import pandas as pd
from PIL import Image

from .splits import institution_of


def _slice_index(stem: str) -> int:
    """`TCGA_HT_8018_19970411_14` -> 14 (trailing integer after the last underscore)."""
    try:
        return int(stem.rsplit("_", 1)[1])
    except (IndexError, ValueError):
        return -1


def build_manifest(datapath: str, out_csv: str) -> pd.DataFrame:
    """Scan the Kaggle folder and write a per-slice manifest CSV.

    Columns: patient, institution, slice_index, image_path, mask_path,
    tumor_pixels, has_tumor.
    """
    datapath = os.path.abspath(datapath)
    if not os.path.isdir(datapath):
        raise FileNotFoundError(f"dataset folder not found: {datapath}")

    rows: List[dict] = []
    patient_dirs = sorted(
        d for d in os.listdir(datapath)
        if os.path.isdir(os.path.join(datapath, d)) and d.startswith("TCGA_")
    )
    if not patient_dirs:
        raise FileNotFoundError(
            f"no TCGA_* patient folders under {datapath}. "
            "Point --datapath at the extracted kaggle_3m/ (or lgg-mri-segmentation/) folder."
        )

    for patient in patient_dirs:
        ppath = os.path.join(datapath, patient)
        images = sorted(
            f for f in os.listdir(ppath)
            if f.endswith(".tif") and "_mask" not in f
        )
        for img in images:
            mask = img.replace(".tif", "_mask.tif")
            mask_path = os.path.join(ppath, mask)
            if not os.path.exists(mask_path):
                continue
            m = np.array(Image.open(mask_path).convert("L"))
            tumor_pixels = int((m > 0).sum())
            stem = img[:-4]
            rows.append(
                {
                    "patient": patient,
                    "institution": institution_of(patient),
                    "slice_index": _slice_index(stem),
                    "image_path": os.path.join(ppath, img),
                    "mask_path": mask_path,
                    "tumor_pixels": tumor_pixels,
                    "has_tumor": tumor_pixels > 0,
                }
            )

    df = pd.DataFrame(rows).sort_values(["patient", "slice_index"]).reset_index(drop=True)
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df


def summarize_manifest(df: pd.DataFrame) -> dict:
    """Sanity-check summary printed after prepare-data; also saved to reports/."""
    per_site = (
        df.groupby("institution")
        .agg(patients=("patient", "nunique"), slices=("image_path", "size"),
             tumor_slices=("has_tumor", "sum"))
        .reset_index()
        .sort_values("patients", ascending=False)
    )
    return {
        "n_patients": int(df["patient"].nunique()),
        "n_slices": int(len(df)),
        "n_tumor_slices": int(df["has_tumor"].sum()),
        "n_empty_slices": int((~df["has_tumor"]).sum()),
        "per_institution": per_site.to_dict(orient="records"),
    }
