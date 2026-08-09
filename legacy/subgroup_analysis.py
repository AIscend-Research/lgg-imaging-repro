"""
Genomic & demographic subgroup equity analysis for nnU-Net on TCGA-LGG.

Pools per-case validation Dice from folds 0/1/2, maps each case to its
patient, joins the clinical/genomic metadata (data.csv), and measures whether
segmentation performance varies across:
  - genomic subtype (RNASeqCluster)
  - histologic grade (Grade II vs III)
  - patient gender
  - patient age group

This is the core "responsible AI / generalization & fairness" Phase 3
deliverable: identifying which patient subgroups the model underserves.

Usage:
    python subgroup_analysis.py \
        --folds_dir ../kaggle-results \
        --metadata  ../kaggle-results/case_metadata.csv \
        --clinical  "lgg-mri-segmentation/kaggle_3m/data.csv"

Outputs:
    results/subgroup_metrics.json
    figures/subgroup_genomic.png
    figures/subgroup_grade.png
    figures/subgroup_demographic.png
"""

import os
import re
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.makedirs('results', exist_ok=True)
os.makedirs('figures', exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument('--folds_dir', default='../kaggle-results', type=str)
parser.add_argument('--metadata',  default='../kaggle-results/case_metadata.csv', type=str)
parser.add_argument('--clinical',  default='lgg-mri-segmentation/kaggle_3m/data.csv', type=str)
parser.add_argument('--folds',     default='0,1,2', type=str)
args = parser.parse_args()

fold_ids = [int(x) for x in args.folds.split(',')]

# ── Pool per-case Dice across folds ───────────────────────────────────────────
rows = []
for f in fold_ids:
    summary_path = os.path.join(args.folds_dir, f'fold_{f}', 'summary.json')
    if not os.path.exists(summary_path):
        print(f"WARNING: {summary_path} missing — skipping fold {f}")
        continue
    with open(summary_path) as fh:
        summary = json.load(fh)
    for entry in summary['metric_per_case']:
        m = entry['metrics']['1']
        case_id = re.search(r'(TCGALGG_\d+)\.nii\.gz', entry['prediction_file']).group(1)
        rows.append({'case_id': case_id, 'fold': f,
                     'Dice': m['Dice'], 'n_ref': m['n_ref']})
case_df = pd.DataFrame(rows)
print(f"Pooled {len(case_df)} validation cases across folds {fold_ids}.")

# Keep tumor slices only (Dice is NaN on empty-empty slices)
tumor_df = case_df[case_df['n_ref'] > 0].copy()
print(f"Tumor slices: {len(tumor_df)}")

# ── Map case -> patient -> short patient id ───────────────────────────────────
meta = pd.read_csv(args.metadata)
tumor_df = tumor_df.merge(meta[['case_id', 'patient']], on='case_id', how='left')
# data.csv uses 3-part IDs (TCGA_CS_4941); metadata uses 4-part (adds date)
tumor_df['patient_short'] = tumor_df['patient'].str.split('_').str[:3].str.join('_')

# Per-patient mean Dice (so big-volume patients don't dominate subgroup means)
patient_df = tumor_df.groupby('patient_short').agg(
    dice=('Dice', 'mean'),
    n_slices=('Dice', 'count'),
).reset_index()
print(f"Patients with tumor slices in pooled folds: {len(patient_df)}")

# ── Join clinical / genomic metadata ──────────────────────────────────────────
clin = pd.read_csv(args.clinical)
clin = clin.rename(columns={'Patient': 'patient_short'})
patient_df = patient_df.merge(clin, on='patient_short', how='left')

results = {'n_patients': int(len(patient_df)), 'n_tumor_slices': int(len(tumor_df)),
           'subgroups': {}}

def summarize(df, col, label_map=None, dropna=True):
    """Group by `col`, return list of dicts with Dice mean/std/n."""
    sub = df.copy()
    if dropna:
        sub = sub[sub[col].notna()]
    out = []
    for val, g in sub.groupby(col):
        label = label_map.get(val, str(val)) if label_map else str(val)
        out.append({
            'group': label,
            'dice_mean': float(g['dice'].mean()),
            'dice_std': float(g['dice'].std(ddof=0)),
            'n_patients': int(len(g)),
        })
    return sorted(out, key=lambda d: d['dice_mean'], reverse=True)

# ── Genomic subtype (RNASeqCluster) ──────────────────────────────────────────
genomic = summarize(patient_df, 'RNASeqCluster',
                    label_map={1: 'RNASeq-1', 2: 'RNASeq-2',
                               3: 'RNASeq-3', 4: 'RNASeq-4'})
results['subgroups']['genomic_RNASeqCluster'] = genomic
print("\n=== By Genomic Subtype (RNASeqCluster) ===")
for r in genomic:
    print(f"  {r['group']}: Dice={r['dice_mean']:.4f}+/-{r['dice_std']:.4f}  n={r['n_patients']}")

# ── Histologic grade ──────────────────────────────────────────────────────────
# data.csv ships no codebook for grade; present raw codes to avoid mislabeling
grade = summarize(patient_df, 'neoplasm_histologic_grade',
                  label_map={1: 'grade=1', 2: 'grade=2', 3: 'grade=3'})
results['subgroups']['histologic_grade'] = grade
print("\n=== By Histologic Grade ===")
for r in grade:
    print(f"  {r['group']}: Dice={r['dice_mean']:.4f}+/-{r['dice_std']:.4f}  n={r['n_patients']}")

# ── Gender ────────────────────────────────────────────────────────────────────
gender = summarize(patient_df, 'gender',
                   label_map={1: 'Gender group 1', 2: 'Gender group 2'})
results['subgroups']['gender'] = gender
print("\n=== By Gender ===")
for r in gender:
    print(f"  {r['group']}: Dice={r['dice_mean']:.4f}+/-{r['dice_std']:.4f}  n={r['n_patients']}")

# ── Age group ─────────────────────────────────────────────────────────────────
patient_df['age_group'] = pd.cut(
    patient_df['age_at_initial_pathologic'],
    bins=[0, 40, 55, 70, np.inf],
    labels=['<40', '40-55', '55-70', '>70'])
age = summarize(patient_df, 'age_group')
results['subgroups']['age_group'] = age
print("\n=== By Age Group ===")
for r in age:
    print(f"  {r['group']}: Dice={r['dice_mean']:.4f}+/-{r['dice_std']:.4f}  n={r['n_patients']}")

# ── Compute disparity gaps ────────────────────────────────────────────────────
def gap(sub):
    vals = [s['dice_mean'] for s in sub if s['n_patients'] >= 3]  # ignore tiny groups
    return float(max(vals) - min(vals)) if len(vals) >= 2 else None

results['disparity_gaps'] = {
    'genomic': gap(genomic),
    'grade': gap(grade),
    'gender': gap(gender),
    'age': gap(age),
}
print("\n=== Disparity gaps (max-min Dice, groups with n>=3) ===")
for k, v in results['disparity_gaps'].items():
    print(f"  {k}: {v:.4f}" if v is not None else f"  {k}: n/a")

with open('results/subgroup_metrics.json', 'w') as f:
    json.dump(results, f, indent=2)
print("\nSaved results/subgroup_metrics.json")

# ── Figures ───────────────────────────────────────────────────────────────────
def bar_fig(sub, title, fname, color='#3F51B5'):
    sub = [s for s in sub if s['n_patients'] >= 1]
    if not sub:
        return
    labels = [s['group'] for s in sub]
    means = [s['dice_mean'] for s in sub]
    stds = [s['dice_std'] for s in sub]
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(labels, means, yerr=stds, capsize=4, color=color, alpha=0.85)
    ax.axhspan(0.82, 0.92, alpha=0.08, color='green')
    ax.set_ylabel('Per-patient Dice (tumor slices)')
    ax.set_title(title, fontweight='bold')
    ax.set_ylim([0, 1.0])
    ax.grid(axis='y', alpha=0.3)
    for bar, s in zip(bars, sub):
        ax.text(bar.get_x() + bar.get_width()/2, s['dice_mean'] + s['dice_std'] + 0.02,
                f"n={s['n_patients']}", ha='center', fontsize=8)
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"Saved {fname}")

bar_fig(genomic, 'Equity by Genomic Subtype (RNASeqCluster)\nLGG molecular subtypes',
        'figures/subgroup_genomic.png', color='#3F51B5')
bar_fig(grade, 'Equity by Histologic Grade\n(Grade II vs III LGG)',
        'figures/subgroup_grade.png', color='#009688')

# Demographic combined figure: gender + age
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, sub, title, color in [
    (axes[0], gender, 'By Gender', '#E91E63'),
    (axes[1], age, 'By Age Group', '#FF5722')]:
    sub = [s for s in sub if s['n_patients'] >= 1]
    labels = [s['group'] for s in sub]
    means = [s['dice_mean'] for s in sub]
    stds = [s['dice_std'] for s in sub]
    bars = ax.bar(labels, means, yerr=stds, capsize=4, color=color, alpha=0.85)
    ax.axhspan(0.82, 0.92, alpha=0.08, color='green')
    ax.set_ylabel('Per-patient Dice')
    ax.set_title(title, fontweight='bold')
    ax.set_ylim([0, 1.0])
    ax.grid(axis='y', alpha=0.3)
    for bar, s in zip(bars, sub):
        ax.text(bar.get_x() + bar.get_width()/2, s['dice_mean'] + s['dice_std'] + 0.02,
                f"n={s['n_patients']}", ha='center', fontsize=8)
plt.suptitle('Demographic Equity Analysis (TCGA-LGG)', fontweight='bold', fontsize=13)
plt.tight_layout()
plt.savefig('figures/subgroup_demographic.png', dpi=150)
plt.close()
print("Saved figures/subgroup_demographic.png")

print("\nDone. Subgroup equity analysis complete.")
