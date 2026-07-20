"""Patient-level split leakage tests (R1 / R6).

These are written BEFORE the training loop and MUST pass before any training
runs. They enforce the single most important faithfulness property of the
reproduction: whole patients are assigned to folds, and no patient's slices are
ever allowed to span a train/val boundary or two validation folds.

The tests build the splits from a synthetic 110-patient roster that mirrors the
real TCGA-LGG institution distribution (DU=45, HT=34, CS=16, FG=14, EZ=1), so
they run in CI without the Kaggle dataset present. If a real manifest exists at
``reports/manifest.csv`` an extra test validates the true roster too.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from lgg.data.splits import make_folds, institution_of  # noqa: E402


# TCGA-LGG institution distribution (2-letter site code -> patient count).
SITE_COUNTS = {"DU": 45, "HT": 34, "CS": 16, "FG": 14, "EZ": 1}
N_PATIENTS = sum(SITE_COUNTS.values())  # 110


def synthetic_patients():
    """Build 110 patient ids shaped like real folder names, e.g. TCGA_DU_0001_1990."""
    patients = []
    for site, n in SITE_COUNTS.items():
        for i in range(n):
            patients.append(f"TCGA_{site}_{i:04d}_19900101")
    return sorted(patients)


ALL_PRESETS = ["kfold5", "buda22"]


@pytest.fixture(scope="module")
def patients():
    return synthetic_patients()


def test_roster_size(patients):
    assert len(patients) == N_PATIENTS == 110
    assert len(set(patients)) == len(patients), "duplicate patient ids"


@pytest.mark.parametrize("preset", ALL_PRESETS)
def test_train_val_disjoint(patients, preset):
    """(a) In every fold, train and val patient sets are disjoint."""
    folds = make_folds(patients, preset=preset, seed=42)
    for f in folds:
        train, val = set(f["train"]), set(f["val"])
        assert train.isdisjoint(val), f"{preset} fold {f['fold']}: train/val overlap"
        # No patient is dropped: train + val == all patients.
        assert train | val == set(patients), f"{preset} fold {f['fold']}: patient dropped"


@pytest.mark.parametrize("preset", ALL_PRESETS)
def test_val_folds_partition_all_patients(patients, preset):
    """(b) Across folds, val sets are pairwise disjoint and cover all 110 patients."""
    folds = make_folds(patients, preset=preset, seed=42)
    seen = set()
    for f in folds:
        val = set(f["val"])
        assert seen.isdisjoint(val), f"{preset} fold {f['fold']}: val patient seen in earlier fold"
        seen |= val
    assert seen == set(patients), "validation folds do not cover every patient exactly once"


def test_buda22_shape(patients):
    """buda22 == 22 non-overlapping subsets of 5 patients each (paper protocol)."""
    folds = make_folds(patients, preset="buda22", seed=42)
    assert len(folds) == 22
    for f in folds:
        assert len(f["val"]) == 5, f"buda22 fold {f['fold']} val size != 5"
        assert len(f["train"]) == 105


def test_kfold5_shape(patients):
    folds = make_folds(patients, preset="kfold5", seed=42)
    assert len(folds) == 5
    val_sizes = sorted(len(f["val"]) for f in folds)
    # 110 / 5 == 22 exactly.
    assert val_sizes == [22, 22, 22, 22, 22]


@pytest.mark.parametrize("preset", ALL_PRESETS)
def test_deterministic(patients, preset):
    a = make_folds(patients, preset=preset, seed=42)
    b = make_folds(patients, preset=preset, seed=42)
    assert [f["val"] for f in a] == [f["val"] for f in b], "splits not deterministic"


def test_stratified_still_partitions(patients):
    """Stratified folds must preserve the partition/leakage guarantees (R6)."""
    folds = make_folds(patients, preset="kfold5", seed=42, stratify_by_site=True)
    seen = set()
    for f in folds:
        assert set(f["train"]).isdisjoint(f["val"])
        assert seen.isdisjoint(f["val"])
        seen |= set(f["val"])
    assert seen == set(patients)


def test_institution_extraction():
    assert institution_of("TCGA_DU_0001_19900101") == "DU"
    assert institution_of("TCGA_HT_8018_19970411") == "HT"


def test_real_manifest_if_present():
    """If a real manifest has been prepared, validate the true 110-patient roster."""
    manifest = os.path.join(os.path.dirname(__file__), "..", "reports", "manifest.csv")
    if not os.path.exists(manifest):
        pytest.skip("reports/manifest.csv not present (run prepare-data first)")
    import pandas as pd

    df = pd.read_csv(manifest)
    patients = sorted(df["patient"].unique())
    assert len(patients) == 110, f"expected 110 patients, got {len(patients)}"
    for preset in ALL_PRESETS:
        folds = make_folds(patients, preset=preset, seed=42)
        seen = set()
        for f in folds:
            assert set(f["train"]).isdisjoint(f["val"])
            assert seen.isdisjoint(f["val"])
            seen |= set(f["val"])
        assert seen == set(patients)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
