"""Radiogenomics reproduction (Section 7) — the core of a *full* replication.

The paper's headline claim is not the segmentation Dice; it is that shape features
of the automatically segmented tumor are **associated with genomic subtype**, and
that the deep-learning masks preserve those associations about as well as the
manual masks. This module reproduces that — CPU-only — from the CV-predicted masks
and the genomic labels that ship in ``data.csv``.

What it does, for both predicted and ground-truth masks:

  1. **Fisher exact tests.** Each shape feature is dichotomized at its median
     (high vs low). Each genomic subtype is tested one cluster-value at a time
     against the rest (a 2x2 table), because the paper reports exactly this kind
     of "cluster R2 vs rest" contrast. Every (feature x subtype x cluster) test is
     Bonferroni-corrected over the full family of tests. Per (feature, subtype) we
     report the strongest (smallest corrected p) cluster contrast.
  2. **Discrimination AUC.** The paper's example is cluster R2 vs the rest using
     the inverse bounding ellipsoid volume ratio (AUC ~= 0.80 model / 0.78 manual).
     We report the AUC of each single feature for that contrast.

Patients with a blank label for the specific test are dropped from *that* test
only (per the brief), never globally.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# Genomic subtype columns tested (primary = RNASeqCluster). Others are included
# when present in data.csv; a missing column is simply skipped.
SUBTYPE_COLUMNS = [
    "RNASeqCluster",
    "MethylationCluster",
    "miRNACluster",
    "CNCluster",
    "COCCluster",
    "RPPACluster",
    "OncosignCluster",
]

# The paper's primary association and discrimination target.
PRIMARY_SUBTYPE = "RNASeqCluster"
PRIMARY_FEATURE = "bevr"
AUC_FEATURE = "inv_bevr"           # inverse bounding ellipsoid volume ratio
AUC_TARGET_TOKEN = "2"             # "cluster R2 vs the rest"

# Fairness / demographic columns (used by the fairness figure, F7).
DEMOGRAPHIC_COLUMNS = ["race", "ethnicity", "neoplasm_histologic_grade", "gender"]


# ------------------------------------------------------------- label loading --

def _patient_key(folder_patient: str) -> str:
    """Map a manifest patient id to a data.csv key.

    Manifest ids are folder names like ``TCGA_CS_4941_19960909``; data.csv keys
    the patient as ``TCGA_CS_4941`` (institution + case, no acquisition date).
    """
    parts = folder_patient.split("_")
    return "_".join(parts[:3]) if len(parts) >= 3 else folder_patient


def find_data_csv(datapath: str) -> Optional[str]:
    """Locate ``data.csv`` near the dataset folder (it ships with the Kaggle set)."""
    cands = []
    d = os.path.abspath(datapath)
    for _ in range(3):  # datapath, its parent, grandparent
        cands.append(os.path.join(d, "data.csv"))
        d = os.path.dirname(d)
    for root, _dirs, files in os.walk(os.path.dirname(os.path.abspath(datapath))):
        if "data.csv" in files:
            cands.append(os.path.join(root, "data.csv"))
    for c in cands:
        if os.path.exists(c):
            return c
    return None


def load_genomic_labels(datapath: str) -> pd.DataFrame:
    """Load data.csv and index it by the short patient key.

    Returns an empty frame if data.csv cannot be found (radiogenomics is then
    skipped with a clear message rather than crashing the pipeline).
    """
    path = find_data_csv(datapath)
    if path is None:
        return pd.DataFrame()
    df = pd.read_csv(path)
    # The patient identifier column is usually named "Patient".
    id_col = next((c for c in df.columns if c.lower() == "patient"), df.columns[0])
    df = df.copy()
    df["_key"] = df[id_col].astype(str).map(_patient_key)
    df = df.set_index("_key")
    return df


def features_to_frame(features: Dict[str, dict]) -> pd.DataFrame:
    """{patient: {feature: value}} -> DataFrame indexed by the short patient key."""
    rows = {}
    for patient, feats in features.items():
        rows[_patient_key(patient)] = feats
    return pd.DataFrame.from_dict(rows, orient="index")


# -------------------------------------------------------------- statistics ----

def _fisher_2x2(feature_high: np.ndarray, cluster_pos: np.ndarray) -> Optional[float]:
    """Fisher exact p for a 2x2 (feature high/low) x (cluster pos/rest) table."""
    from scipy.stats import fisher_exact

    a = int(np.sum(feature_high & cluster_pos))
    b = int(np.sum(feature_high & ~cluster_pos))
    c = int(np.sum(~feature_high & cluster_pos))
    d = int(np.sum(~feature_high & ~cluster_pos))
    if (a + b) == 0 or (c + d) == 0 or (a + c) == 0 or (b + d) == 0:
        return None
    _, p = fisher_exact([[a, b], [c, d]])
    return float(p)


def association_tests(
    feat_df: pd.DataFrame,
    labels: pd.DataFrame,
    feature_names: List[str],
    subtype_cols: List[str],
) -> dict:
    """Run every (feature x subtype x cluster) Fisher test, Bonferroni-corrected.

    Returns per (feature, subtype): the strongest cluster contrast with its raw
    and corrected p-value and n, plus the total number of tests used for the
    correction.
    """
    raw: List[dict] = []
    for feat in feature_names:
        if feat not in feat_df.columns:
            continue
        for sub in subtype_cols:
            if sub not in labels.columns:
                continue
            joined = feat_df[[feat]].join(labels[[sub]], how="inner").dropna()
            if len(joined) < 8:
                continue
            fvals = joined[feat].to_numpy(dtype=float)
            feature_high = fvals > np.median(fvals)
            clusters = joined[sub].astype(str).to_numpy()
            for cval in sorted(set(clusters)):
                cluster_pos = clusters == cval
                if cluster_pos.sum() < 2 or (~cluster_pos).sum() < 2:
                    continue
                p = _fisher_2x2(feature_high, cluster_pos)
                if p is None:
                    continue
                raw.append({"feature": feat, "subtype": sub, "cluster": cval,
                            "p_raw": p, "n": int(len(joined))})

    n_tests = len(raw)
    for r in raw:
        r["p_bonferroni"] = min(1.0, r["p_raw"] * max(1, n_tests))

    # keep the strongest (smallest corrected p) contrast per (feature, subtype)
    best: Dict[tuple, dict] = {}
    for r in raw:
        key = (r["feature"], r["subtype"])
        if key not in best or r["p_bonferroni"] < best[key]["p_bonferroni"]:
            best[key] = r
    return {"n_tests": n_tests, "best_per_pair": list(best.values()), "all_tests": raw}


def discrimination_auc(
    feat_df: pd.DataFrame,
    labels: pd.DataFrame,
    subtype: str = PRIMARY_SUBTYPE,
    feature: str = AUC_FEATURE,
    target_token: str = AUC_TARGET_TOKEN,
) -> Optional[dict]:
    """AUC of a single feature discriminating one cluster vs the rest.

    Reproduces the paper's cluster-R2-vs-rest / inverse-BEVR check. The target
    cluster is the value whose string contains ``target_token`` (e.g. "2" -> R2);
    if none matches, the most populous cluster is used and noted.
    """
    from sklearn.metrics import roc_auc_score, roc_curve

    if feature not in feat_df.columns or subtype not in labels.columns:
        return None
    joined = feat_df[[feature]].join(labels[[subtype]], how="inner").dropna()
    if len(joined) < 10:
        return None
    clusters = joined[subtype].astype(str)
    values = sorted(clusters.unique())
    target = next((v for v in values if target_token in v), None)
    used_fallback = target is None
    if target is None:
        target = clusters.value_counts().idxmax()
    y = (clusters == target).astype(int).to_numpy()
    if y.sum() < 2 or (len(y) - y.sum()) < 2:
        return None
    score = joined[feature].to_numpy(dtype=float)
    try:
        auc = float(roc_auc_score(y, score))
    except ValueError:
        return None
    # Report the AUC with the feature's FIXED direction (inverse BEVR), exactly as
    # the paper does — NOT max(auc, 1-auc). Orienting to whichever direction scores
    # higher is post-hoc and would (a) inflate chance-level results and (b) make the
    # reported value disagree with a plain roc_auc_score(y, feature) recompute.
    fpr, tpr, _ = roc_curve(y, score)
    return {"subtype": subtype, "feature": feature, "target_cluster": target,
            "auc": auc, "n": int(len(joined)), "n_positive": int(y.sum()),
            "used_fallback_cluster": used_fallback, "orientation": "fixed (inverse BEVR)",
            "roc_fpr": [float(x) for x in fpr], "roc_tpr": [float(x) for x in tpr]}


def run_radiogenomics(
    pred_features: Dict[str, dict],
    gt_features: Dict[str, dict],
    labels: pd.DataFrame,
    feature_names: Optional[List[str]] = None,
    subtype_cols: Optional[List[str]] = None,
) -> dict:
    """Full radiogenomics reproduction for predicted and ground-truth masks."""
    if feature_names is None:
        feature_names = ["bevr", "inv_bevr", "angular_std", "margin_fluctuation"]
    if subtype_cols is None:
        subtype_cols = [c for c in SUBTYPE_COLUMNS if c in labels.columns]

    out: dict = {"subtypes_tested": subtype_cols, "features": feature_names, "masks": {}}
    for name, feats in (("predicted", pred_features), ("ground_truth", gt_features)):
        fdf = features_to_frame(feats)
        assoc = association_tests(fdf, labels, feature_names, subtype_cols)
        auc = discrimination_auc(fdf, labels)
        out["masks"][name] = {"associations": assoc, "discrimination_auc": auc,
                              "n_patients_with_features": int(fdf.dropna(how="all").shape[0])}
    out["paper_reference"] = {
        "rnaseq_x_bevr_p": 0.0002,
        "rnaseq_x_margin_fluctuation_p": 0.005,
        "auc_model": 0.80,
        "auc_manual": 0.78,
    }
    return out
