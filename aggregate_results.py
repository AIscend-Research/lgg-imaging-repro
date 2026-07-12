"""
Aggregates nnU-Net cross-validation results across 3 folds.

Usage (after all 3 folds finish training on Kaggle):
    python aggregate_results.py \
        --results_dir nnunet_results/Dataset001_TCGALGG/nnUNetTrainer__nnUNetPlans__2d

Outputs:
    results/baseline_metrics.json    -- mean +/- std across folds
    figures/training_curves.png      -- loss curves per fold
    figures/fold_comparison.png      -- Dice per fold bar chart
"""

import os
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.makedirs('results', exist_ok=True)
os.makedirs('figures', exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument('--results_dir',
    default='nnunet_results/Dataset001_TCGALGG/nnUNetTrainer__nnUNetPlans__2d',
    type=str)
args = parser.parse_args()

# ── Load per-fold summary files ───────────────────────────────────────────────
# nnU-Net writes validation_raw/summary.json inside each fold_X directory
fold_metrics = {}
for fold in range(3):
    summary_path = os.path.join(args.results_dir, f'fold_{fold}',
                                'validation_raw', 'summary.json')
    if not os.path.exists(summary_path):
        print(f"WARNING: fold {fold} summary not found at {summary_path}")
        continue
    with open(summary_path) as f:
        data = json.load(f)
    # nnU-Net summary format: data['mean']['1'] has per-class metrics
    metrics_1 = data.get('mean', {}).get('1', {})
    fold_metrics[fold] = {
        'Dice':       metrics_1.get('Dice', None),
        'IoU':        metrics_1.get('IoU', None),
        'HD95':       metrics_1.get('HD95', None),
        'Precision':  metrics_1.get('Precision', None),
        'Recall':     metrics_1.get('Recall', None),
    }
    hd = fold_metrics[fold]['HD95']
    hd_str = f"{hd:.2f}" if hd is not None else "N/A"
    print(f"Fold {fold}: Dice={fold_metrics[fold]['Dice']:.4f}  "
          f"IoU={fold_metrics[fold]['IoU']:.4f}  "
          f"HD95={hd_str}")

if not fold_metrics:
    print("No fold results found. Train all 3 folds first, then run this script.")
    exit(0)

# ── Aggregate ─────────────────────────────────────────────────────────────────
metric_names = ['Dice', 'IoU', 'HD95', 'Precision', 'Recall']
agg = {}
for m in metric_names:
    vals = [fold_metrics[f][m] for f in fold_metrics if fold_metrics[f][m] is not None]
    if vals:
        agg[m] = {'mean': float(np.mean(vals)), 'std': float(np.std(vals)),
                  'min': float(np.min(vals)),   'max': float(np.max(vals))}

print("\n--- Aggregated Results (3-fold CV) ---")
for m, s in agg.items():
    print(f"  {m:<12}: {s['mean']:.4f} +/- {s['std']:.4f}  "
          f"[{s['min']:.4f} -- {s['max']:.4f}]")

# Compare to target
target_dice_min, target_dice_max = 0.82, 0.92
dice_mean = agg.get('Dice', {}).get('mean', None)
if dice_mean:
    status = "WITHIN" if target_dice_min <= dice_mean <= target_dice_max else "OUTSIDE"
    print(f"\n  Target Dice range: {target_dice_min}--{target_dice_max}  "
          f"-> Our result {dice_mean:.4f} is {status} target range.")

with open('results/baseline_metrics.json', 'w') as f:
    json.dump({'per_fold': fold_metrics, 'aggregated': agg}, f, indent=2)
print("\nSaved results/baseline_metrics.json")

# ── Figure: fold comparison bar chart ─────────────────────────────────────────
metric_plot = ['Dice', 'IoU', 'HD95']
available_metrics = [m for m in metric_plot if any(
    fold_metrics[f][m] is not None for f in fold_metrics)]
colors_all = {'Dice': '#2196F3', 'IoU': '#4CAF50', 'HD95': '#FF9800'}

fig, axes = plt.subplots(1, len(available_metrics), figsize=(5 * len(available_metrics), 4))
if len(available_metrics) == 1:
    axes = [axes]

for ax, metric in zip(axes, available_metrics):
    color = colors_all[metric]
    folds = sorted(fold_metrics.keys())
    fold_labels = [f"Fold {f}" for f in folds if fold_metrics[f][metric] is not None]
    vals  = [fold_metrics[f][metric] for f in folds if fold_metrics[f][metric] is not None]
    bars  = ax.bar(fold_labels, vals, color=color, alpha=0.85)
    if metric in agg:
        ax.axhline(agg[metric]['mean'], color='red', linestyle='--', lw=1.2,
                   label=f"Mean={agg[metric]['mean']:.3f}")
    ax.set_title(metric, fontsize=12, fontweight='bold')
    ax.set_ylabel(metric)
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.005,
                f'{v:.3f}', ha='center', fontsize=9)
    if metric == 'Dice':
        ax.axhspan(0.82, 0.92, alpha=0.08, color='green', label='Target range')
        ax.set_ylim([0.7, 1.0])

plt.suptitle('nnU-Net 2D Baseline: 3-Fold Cross-Validation Results',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/fold_comparison.png', dpi=150)
plt.close()
print("Saved figures/fold_comparison.png")
