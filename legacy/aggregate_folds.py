"""
Aggregate nnU-Net 3-fold cross-validation results for TCGA-LGG.

Reads results/fold{0,1,2}_metrics.json (produced by analyze_fold.py) and
computes the final reproduction numbers: mean +/- std across folds for
overall metrics, per-institution equity, and tumor-size performance.

Usage:
    python aggregate_folds.py

Outputs:
    results/final_3fold_metrics.json
    figures/final_3fold_summary.png
    figures/final_institution_3fold.png
    figures/final_tumorsize_3fold.png
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.makedirs('results', exist_ok=True)
os.makedirs('figures', exist_ok=True)

# ── Load per-fold metrics ─────────────────────────────────────────────────────
folds = {}
for f in [0, 1, 2]:
    path = f'results/fold{f}_metrics.json'
    if not os.path.exists(path):
        print(f"WARNING: {path} missing — skipping fold {f}")
        continue
    with open(path) as fh:
        folds[f] = json.load(fh)

if len(folds) < 2:
    print("Need at least 2 folds. Run analyze_fold.py first.")
    raise SystemExit(0)

fold_ids = sorted(folds.keys())
print(f"Aggregating folds: {fold_ids}\n")

# ── Overall metrics across folds ──────────────────────────────────────────────
def agg(values):
    return {'mean': float(np.mean(values)), 'std': float(np.std(values)),
            'min': float(np.min(values)), 'max': float(np.max(values)),
            'per_fold': [float(v) for v in values]}

overall_keys = ['Dice', 'IoU', 'Precision', 'Recall']
overall_agg = {k: agg([folds[f]['overall'][k] for f in fold_ids]) for k in overall_keys}
tumor_dice  = agg([folds[f]['tumor_slice_dice_mean'] for f in fold_ids])

print("=== 3-Fold Cross-Validation: Overall (all validation cases) ===")
for k in overall_keys:
    s = overall_agg[k]
    print(f"  {k:<10}: {s['mean']:.4f} +/- {s['std']:.4f}  "
          f"[folds: {', '.join(f'{v:.3f}' for v in s['per_fold'])}]")
print(f"\n  Tumor-slice Dice: {tumor_dice['mean']:.4f} +/- {tumor_dice['std']:.4f}  "
      f"[folds: {', '.join(f'{v:.3f}' for v in tumor_dice['per_fold'])}]")

tmin, tmax = 0.82, 0.92
status = "WITHIN" if tmin <= tumor_dice['mean'] <= tmax else "OUTSIDE"
print(f"  Paper target {tmin}-{tmax} -> {status} target range.")

# ── Per-institution across folds ──────────────────────────────────────────────
inst_rows = []
for f in fold_ids:
    for rec in folds[f]['per_institution']:
        inst_rows.append({'fold': f, **rec})
inst_df = pd.DataFrame(inst_rows)
inst_summary = inst_df.groupby('institution').agg(
    dice_mean=('dice_mean', 'mean'),
    dice_std=('dice_mean', 'std'),
    total_cases=('n_cases', 'sum'),
    n_folds=('fold', 'nunique'),
).reset_index().sort_values('dice_mean', ascending=False)

print("\n=== Per-Institution (averaged across folds) ===")
for _, r in inst_summary.iterrows():
    sd = 0.0 if pd.isna(r['dice_std']) else r['dice_std']
    print(f"  {r['institution']}: Dice={r['dice_mean']:.4f} (cross-fold SD {sd:.4f})  "
          f"total tumor cases={int(r['total_cases'])}  folds={int(r['n_folds'])}")

# ── Tumor size across folds ───────────────────────────────────────────────────
size_rows = []
for f in fold_ids:
    for rec in folds[f]['by_tumor_size']:
        size_rows.append({'fold': f, **rec})
size_df = pd.DataFrame(size_rows)
# preserve natural size order
size_order = ['XS (<200)', 'S (200-500)', 'M (500-1k)', 'L (1k-2k)', 'XL (>2k)']
size_summary = size_df.groupby('size_bucket').agg(
    dice_mean=('dice_mean', 'mean'),
    dice_std=('dice_mean', 'std'),
    total_cases=('n_cases', 'sum'),
).reindex(size_order).dropna(how='all').reset_index()

print("\n=== By Tumor Size (averaged across folds) ===")
for _, r in size_summary.iterrows():
    sd = 0.0 if pd.isna(r['dice_std']) else r['dice_std']
    print(f"  {r['size_bucket']}: Dice={r['dice_mean']:.4f} (SD {sd:.4f})  "
          f"total cases={int(r['total_cases'])}")

# ── Save final JSON ───────────────────────────────────────────────────────────
final = {
    'n_folds': len(fold_ids),
    'fold_ids': fold_ids,
    'overall': overall_agg,
    'tumor_slice_dice': tumor_dice,
    'per_institution': inst_summary.fillna(0).to_dict(orient='records'),
    'by_tumor_size': size_summary.fillna(0).to_dict(orient='records'),
    'paper_target_dice': [tmin, tmax],
    'target_status': status,
}
with open('results/final_3fold_metrics.json', 'w') as f:
    json.dump(final, f, indent=2)
print("\nSaved results/final_3fold_metrics.json")

# ── Figure 1: overall metrics per fold ───────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(len(overall_keys))
width = 0.22
colors = ['#2196F3', '#4CAF50', '#FF9800']
for i, f in enumerate(fold_ids):
    vals = [folds[f]['overall'][k] for k in overall_keys]
    ax.bar(x + (i - 1) * width, vals, width, label=f'Fold {f}',
           color=colors[i % 3], alpha=0.85)
means = [overall_agg[k]['mean'] for k in overall_keys]
ax.plot(x, means, 'k_', markersize=28, markeredgewidth=2.5, label='Mean')
ax.set_xticks(x)
ax.set_xticklabels(overall_keys)
ax.set_ylabel('Score')
ax.set_ylim([0, 1.0])
ax.set_title('nnU-Net 2D: 3-Fold Cross-Validation Metrics (TCGA-LGG)',
             fontweight='bold')
ax.legend(ncol=4, fontsize=9)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('figures/final_3fold_summary.png', dpi=150)
plt.close()
print("Saved figures/final_3fold_summary.png")

# ── Figure 2: per-institution equity ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
colors = ['#4CAF50' if d > 0.87 else '#FF9800' if d > 0.82 else '#F44336'
          for d in inst_summary['dice_mean']]
ax.bar(inst_summary['institution'], inst_summary['dice_mean'],
       yerr=inst_summary['dice_std'].fillna(0), capsize=4, color=colors, alpha=0.85)
ax.axhspan(0.82, 0.92, alpha=0.08, color='green')
ax.axhline(tumor_dice['mean'], color='blue', linestyle='--', lw=1.2,
           label=f"Overall tumor-slice {tumor_dice['mean']:.3f}")
ax.set_ylabel('Dice (tumor slices, 3-fold mean)')
ax.set_xlabel('Institution (TCIA center)')
ax.set_title('Cross-Institution Equity: 3-Fold Mean Dice\n'
             '(error bars = variation across folds)', fontweight='bold')
ax.set_ylim([0, 1.0])
ax.legend()
ax.grid(axis='y', alpha=0.3)
for i, (_, r) in enumerate(inst_summary.iterrows()):
    ax.text(i, 0.05, f"n={int(r['total_cases'])}", ha='center', fontsize=8, color='white',
            fontweight='bold')
plt.tight_layout()
plt.savefig('figures/final_institution_3fold.png', dpi=150)
plt.close()
print("Saved figures/final_institution_3fold.png")

# ── Figure 3: tumor size ─────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(size_summary['size_bucket'], size_summary['dice_mean'],
       yerr=size_summary['dice_std'].fillna(0), capsize=4, color='#9C27B0', alpha=0.8)
ax.axhspan(0.82, 0.92, alpha=0.08, color='green')
ax.set_ylabel('Dice (3-fold mean)')
ax.set_xlabel('Tumor size (ground-truth pixels)')
ax.set_title('Performance vs Tumor Size (3-Fold Mean)\n'
             'Small tumors = hardest, highest-stakes cases', fontweight='bold')
ax.set_ylim([0, 1.0])
ax.grid(axis='y', alpha=0.3)
for i, (_, r) in enumerate(size_summary.iterrows()):
    ax.text(i, r['dice_mean'] + 0.03, f"n={int(r['total_cases'])}",
            ha='center', fontsize=8)
plt.tight_layout()
plt.savefig('figures/final_tumorsize_3fold.png', dpi=150)
plt.close()
print("Saved figures/final_tumorsize_3fold.png")

print("\nDone. Final 3-fold aggregation complete.")
