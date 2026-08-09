"""
Per-fold nnU-Net results analysis for TCGA-LGG.

Reads a fold's validation summary.json + case_metadata.csv and computes:
  - Overall Dice / IoU / precision / recall (foreground class)
  - Per-institution breakdown (equity / fairness analysis)
  - Performance by tumor size bucket
  - Per-case distribution

Usage:
    python analyze_fold.py \
        --summary  ../kaggle-results/fold_0/summary.json \
        --metadata ../kaggle-results/case_metadata.csv \
        --fold 0

Outputs (suffixed by fold number):
    results/fold{N}_metrics.json
    figures/fold{N}_institution.png
    figures/fold{N}_tumorsize.png
    figures/fold{N}_dice_distribution.png
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
parser.add_argument('--summary',  required=True, type=str)
parser.add_argument('--metadata', required=True, type=str)
parser.add_argument('--fold',     default=0, type=int)
args = parser.parse_args()

# ── Load summary.json (NaN is valid JSON token for Python's json) ─────────────
with open(args.summary) as f:
    summary = json.load(f)

meta = pd.read_csv(args.metadata)
meta['institution'] = meta['patient'].str.extract(r'TCGA_([A-Z]+)_')

# ── Overall metrics (foreground class "1") ────────────────────────────────────
mean1 = summary['mean']['1']
tp, fp, fn = mean1['TP'], mean1['FP'], mean1['FN']
precision = tp / (tp + fp + 1e-8)
recall    = tp / (tp + fn + 1e-8)

overall = {
    'Dice':      mean1['Dice'],
    'IoU':       mean1['IoU'],
    'Precision': precision,
    'Recall':    recall,
    'n_cases':   len(summary['metric_per_case']),
}
print(f"=== Fold {args.fold} Overall ===")
print(f"  Dice:      {overall['Dice']:.4f}")
print(f"  IoU:       {overall['IoU']:.4f}")
print(f"  Precision: {overall['Precision']:.4f}")
print(f"  Recall:    {overall['Recall']:.4f}")
print(f"  Val cases: {overall['n_cases']}")

# ── Per-case table merged with institution ────────────────────────────────────
rows = []
for entry in summary['metric_per_case']:
    m = entry['metrics']['1']
    case_id = re.search(r'(TCGALGG_\d+)\.nii\.gz', entry['prediction_file']).group(1)
    rows.append({
        'case_id': case_id,
        'Dice':    m['Dice'],
        'IoU':     m['IoU'],
        'TP':      m['TP'],
        'FP':      m['FP'],
        'FN':      m['FN'],
        'n_ref':   m['n_ref'],      # ground-truth tumor pixels (size proxy)
    })
case_df = pd.DataFrame(rows).merge(meta[['case_id', 'patient', 'institution']],
                                   on='case_id', how='left')

# Tumor vs no-tumor: NaN Dice == both pred & ref empty (true negative slice)
case_df['has_tumor'] = case_df['n_ref'] > 0
tumor_df = case_df[case_df['has_tumor']].copy()
print(f"\n  Tumor slices in val: {len(tumor_df)} / {len(case_df)}")
print(f"  Mean Dice (tumor slices only): {tumor_df['Dice'].mean():.4f}")

# ── Per-institution breakdown ─────────────────────────────────────────────────
inst_stats = tumor_df.groupby('institution').agg(
    n_cases=('Dice', 'count'),
    dice_mean=('Dice', 'mean'),
    dice_std=('Dice', 'std'),
    iou_mean=('IoU', 'mean'),
).reset_index().sort_values('dice_mean', ascending=False)

print("\n=== Per-Institution (tumor slices) ===")
for _, r in inst_stats.iterrows():
    print(f"  {r['institution']}: n={int(r['n_cases'])}  "
          f"Dice={r['dice_mean']:.4f}±{r['dice_std']:.4f}  IoU={r['iou_mean']:.4f}")

# ── Tumor-size buckets ────────────────────────────────────────────────────────
tumor_df['size_bucket'] = pd.cut(
    tumor_df['n_ref'],
    bins=[0, 200, 500, 1000, 2000, np.inf],
    labels=['XS (<200)', 'S (200-500)', 'M (500-1k)', 'L (1k-2k)', 'XL (>2k)']
)
size_stats = tumor_df.groupby('size_bucket', observed=True).agg(
    n_cases=('Dice', 'count'),
    dice_mean=('Dice', 'mean'),
    dice_std=('Dice', 'std'),
).reset_index()

print("\n=== By Tumor Size ===")
for _, r in size_stats.iterrows():
    print(f"  {r['size_bucket']}: n={int(r['n_cases'])}  Dice={r['dice_mean']:.4f}")

# ── Save metrics JSON ─────────────────────────────────────────────────────────
out = {
    'fold': args.fold,
    'overall': overall,
    'tumor_slice_dice_mean': float(tumor_df['Dice'].mean()),
    'tumor_slice_count': int(len(tumor_df)),
    'per_institution': inst_stats.to_dict(orient='records'),
    'by_tumor_size': size_stats.astype({'size_bucket': str}).to_dict(orient='records'),
    'target_dice_range': [0.82, 0.92],
}
results_path = f'results/fold{args.fold}_metrics.json'
with open(results_path, 'w') as f:
    json.dump(out, f, indent=2)
print(f"\nSaved {results_path}")

# ── Figure 1: per-institution Dice ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
colors = ['#4CAF50' if d > 0.87 else '#FF9800' if d > 0.82 else '#F44336'
          for d in inst_stats['dice_mean']]
bars = ax.bar(inst_stats['institution'], inst_stats['dice_mean'],
              yerr=inst_stats['dice_std'], capsize=4, color=colors, alpha=0.85)
ax.axhspan(0.82, 0.92, alpha=0.08, color='green')
ax.axhline(overall['Dice'], color='blue', linestyle='--', lw=1.2,
           label=f"Overall {overall['Dice']:.3f}")
ax.set_ylabel('Dice (tumor slices)')
ax.set_xlabel('Institution (TCIA center code)')
ax.set_title(f'Fold {args.fold}: Per-Institution Segmentation Performance\n'
             '(equity analysis — green band = target range)', fontweight='bold')
ax.set_ylim([0, 1.0])
ax.legend()
ax.grid(axis='y', alpha=0.3)
for bar, (_, r) in zip(bars, inst_stats.iterrows()):
    ax.text(bar.get_x() + bar.get_width()/2, r['dice_mean'] + r['dice_std'] + 0.02,
            f"n={int(r['n_cases'])}", ha='center', fontsize=8)
plt.tight_layout()
plt.savefig(f'figures/fold{args.fold}_institution.png', dpi=150)
plt.close()
print(f"Saved figures/fold{args.fold}_institution.png")

# ── Figure 2: tumor-size buckets ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(size_stats['size_bucket'].astype(str), size_stats['dice_mean'],
              yerr=size_stats['dice_std'], capsize=4, color='#9C27B0', alpha=0.8)
ax.axhspan(0.82, 0.92, alpha=0.08, color='green')
ax.set_ylabel('Dice')
ax.set_xlabel('Tumor size (ground-truth pixels)')
ax.set_title(f'Fold {args.fold}: Performance vs Tumor Size\n'
             '(small tumors are the hard, high-stakes cases)', fontweight='bold')
ax.set_ylim([0, 1.0])
ax.grid(axis='y', alpha=0.3)
for bar, (_, r) in zip(bars, size_stats.iterrows()):
    ax.text(bar.get_x() + bar.get_width()/2, r['dice_mean'] + 0.03,
            f"n={int(r['n_cases'])}", ha='center', fontsize=8)
plt.tight_layout()
plt.savefig(f'figures/fold{args.fold}_tumorsize.png', dpi=150)
plt.close()
print(f"Saved figures/fold{args.fold}_tumorsize.png")

# ── Figure 3: Dice distribution ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(tumor_df['Dice'], bins=30, color='#2196F3', alpha=0.8, edgecolor='white')
ax.axvline(tumor_df['Dice'].mean(), color='red', linestyle='--', lw=1.5,
           label=f"Mean {tumor_df['Dice'].mean():.3f}")
ax.axvline(tumor_df['Dice'].median(), color='green', linestyle=':', lw=1.5,
           label=f"Median {tumor_df['Dice'].median():.3f}")
ax.set_xlabel('Dice score')
ax.set_ylabel('Number of tumor slices')
ax.set_title(f'Fold {args.fold}: Dice Distribution Across Validation Tumor Slices',
             fontweight='bold')
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f'figures/fold{args.fold}_dice_distribution.png', dpi=150)
plt.close()
print(f"Saved figures/fold{args.fold}_dice_distribution.png")

print(f"\nDone. Fold {args.fold} analysis complete.")
