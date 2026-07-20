"""Patient-level cross-validation splits (R1, R6).

Two presets:
  * ``buda22`` — 22 non-overlapping subsets of 5 patients each, matching the
    paper's exact cross-validation protocol (Buda et al. 2019).
  * ``kfold5`` — patient-level 5-fold, the compute-friendly default.

Every fold assigns *whole patients* to train/val; slices never span folds. The
validation folds always partition the full patient roster (each patient is held
out exactly once), which is what ``tests/test_no_leakage.py`` enforces.
"""

from __future__ import annotations

import json
import os
import re
from typing import Dict, List, Optional

_SITE_RE = re.compile(r"TCGA_([A-Za-z]{2})_")


def institution_of(patient_id: str) -> str:
    """Extract the 2-letter TCIA site code, e.g. 'TCGA_HT_8018_...' -> 'HT'."""
    m = _SITE_RE.search(patient_id)
    if not m:
        raise ValueError(f"cannot parse institution from patient id: {patient_id!r}")
    return m.group(1).upper()


def _round_robin_folds(ordered: List[str], n_folds: int) -> List[List[str]]:
    """Deterministically deal an ordered patient list into n_folds val buckets."""
    buckets: List[List[str]] = [[] for _ in range(n_folds)]
    for i, p in enumerate(ordered):
        buckets[i % n_folds].append(p)
    return buckets


def _seeded_shuffle(items: List[str], seed: int) -> List[str]:
    import random

    rng = random.Random(seed)
    out = list(items)
    rng.shuffle(out)
    return out


def make_folds(
    patients: List[str],
    preset: str = "kfold5",
    seed: int = 42,
    stratify_by_site: bool = False,
    stratify_burden: Optional[Dict[str, float]] = None,
) -> List[Dict]:
    """Build patient-level folds.

    Parameters
    ----------
    patients : list of patient ids (folder names).
    preset : 'kfold5' or 'buda22'.
    seed : RNG seed (fixed for reproducibility).
    stratify_by_site : if True, balance institutions across folds while keeping
        the partition/leakage guarantees.
    stratify_burden : optional {patient_id: tumor_burden} to further order
        within each stratum (documented flag, R6).

    Returns a list of {'fold', 'train', 'val'} dicts. Validation sets partition
    the roster (each patient held out exactly once).
    """
    patients = sorted(set(patients))
    if len(patients) == 0:
        raise ValueError("no patients supplied")

    if preset == "buda22":
        n_folds = 22
    elif preset == "kfold5":
        n_folds = 5
    elif preset.startswith("kfold"):
        n_folds = int(preset[len("kfold"):])
    else:
        raise ValueError(f"unknown preset: {preset!r} (use 'kfold5' or 'buda22')")

    if stratify_by_site:
        # Group by site, order each group (optionally by tumor burden), then
        # deal each group round-robin so every fold gets a balanced site mix.
        groups: Dict[str, List[str]] = {}
        for p in patients:
            groups.setdefault(institution_of(p), []).append(p)
        buckets: List[List[str]] = [[] for _ in range(n_folds)]
        offset = 0
        for site in sorted(groups):
            members = groups[site]
            if stratify_burden is not None:
                members = sorted(members, key=lambda p: (stratify_burden.get(p, 0.0), p))
            else:
                members = _seeded_shuffle(members, seed + hash(site) % 1000)
            for i, p in enumerate(members):
                buckets[(i + offset) % n_folds].append(p)
            offset += len(members)  # rotate start so small sites don't all land in fold 0
    else:
        ordered = _seeded_shuffle(patients, seed)
        buckets = _round_robin_folds(ordered, n_folds)

    all_set = set(patients)
    folds = []
    for i, val in enumerate(buckets):
        val_sorted = sorted(val)
        train_sorted = sorted(all_set - set(val_sorted))
        folds.append({"fold": i, "train": train_sorted, "val": val_sorted})
    return folds


def patient_ids_from_manifest(df) -> List[str]:
    return sorted(df["patient"].unique().tolist())


def save_splits(folds: List[Dict], path: str, preset: str, seed: int, **meta) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    payload = {
        "preset": preset,
        "seed": seed,
        "n_folds": len(folds),
        "n_patients": len(set(p for f in folds for p in f["val"])),
        "meta": meta,
        "folds": folds,
    }
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2)


def load_splits(path: str) -> Dict:
    with open(path) as fh:
        return json.load(fh)
