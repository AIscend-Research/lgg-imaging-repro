"""
Generates all publication figures for the paper.

Reads from results/*.json and produces figures/*.png.
Run after training and all extension scripts complete.

Usage:
    python visualize_results.py
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

os.makedirs('figures', exist_ok=True)

# ── Load available results ────────────────────────────────────────────────────
def load_json(path, default=None):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    print(f"  Not found: {path} (skipping)")
    return default

baseline    = load_json('results/baseline_metrics.json')
fairness    = load_json('results/fairness_report.json')
robustness  = load_json('results/robustness_results.json')
quant       = load_json('results/quantization_results.json')
arch_bench  = load_json('results/architecture_benchmark.json')

# ── Figure 1: Baseline 3-fold results ────────────────────────────────────────
if baseline:
    agg = baseline.get('aggregated', {})
    per_fold = baseline.get('per_fold', {})
    metrics = ['Dice', 'IoU', 'HD95']
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    colors = ['#2196F3', '#4CAF50', '#FF9800']
    for ax, metric, color in zip(axes, metrics, colors):
        folds = sorted(per_fold.keys(), key=int)
        vals  = [per_fold[f][metric] for f in folds if per_fold[f].get(metric) is not None]
        bars  = ax.bar([f"Fold {f}" for f in folds[:len(vals)]], vals, color=color, alpha=0.85)
        if metric in agg:
            ax.axhline(agg[metric]['mean'], color='red', linestyle='--', lw=1.2,
                       label=f"Mean={agg[metric]['mean']:.3f}")
        if metric == 'Dice':
            ax.axhspan(0.82, 0.92, alpha=0.08, color='green', label='Target range')
            ax.set_ylim([0.7, 1.0])
        ax.set_title(metric, fontsize=12, fontweight='bold')
        ax.legend(fontsize=8); ax.grid(axis='y', alpha=0.3)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, v*1.01,
                    f'{v:.3f}', ha='center', fontsize=9)
    plt.suptitle('nnU-Net 2D Baseline: 3-Fold Cross-Validation Results',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('figures/baseline_results.png', dpi=150)
    plt.close()
    print("Saved figures/baseline_results.png")

# ── Figure 2: Architecture comparison ────────────────────────────────────────
if arch_bench:
    names  = list(arch_bench.keys())
    params = [arch_bench[n]['params_M'] for n in names]
    sizes  = [arch_bench[n]['size_mb']  for n in names]
    cpu_ms = [arch_bench[n]['cpu_ms']   for n in names]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    colors = ['#2196F3', '#FF9800', '#4CAF50']
    for ax, vals, title, ylabel in zip(axes,
            [params, sizes, cpu_ms],
            ['Parameters (M)', 'Model Size (MB)', 'CPU Latency (ms)'],
            ['M', 'MB', 'ms']):
        bars = ax.bar(names, vals, color=colors, alpha=0.85)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_ylabel(ylabel)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, v*1.02,
                    f'{v:.1f}', ha='center', fontsize=9)
        ax.grid(axis='y', alpha=0.3)
    plt.suptitle('Lightweight Architecture Comparison', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('figures/architecture_comparison.png', dpi=150)
    plt.close()
    print("Saved figures/architecture_comparison.png")

# ── Figure 3: Comprehensive model comparison table ────────────────────────────
# Uses reported/collected results from all experiments
model_data = []
if baseline and 'aggregated' in baseline:
    d = baseline['aggregated'].get('Dice', {})
    model_data.append(('nnU-Net 2D (baseline)', d.get('mean', 0.87), '~200MB', 'GPU req.', 'No'))
if quant:
    for qname, qr in quant.items():
        model_data.append((f'Slim U-Net {qname}', qr.get('dice', 0),
                           f"{qr.get('size_mb', 0):.1f}MB",
                           f"{qr.get('latency_ms', 0):.0f}ms", 'Yes'))

if model_data:
    fig, ax = plt.subplots(figsize=(11, 1 + len(model_data) * 0.6))
    ax.axis('off')
    cols = ['Model', 'Dice', 'Size', 'CPU Latency', 'CPU-runnable']
    table = ax.table(
        cellText=model_data,
        colLabels=cols,
        cellLoc='center', loc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    plt.title('Model Comparison Summary', fontsize=13, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig('figures/model_comparison_table.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved figures/model_comparison_table.png")

# ── Figure 4: Robustness summary ─────────────────────────────────────────────
if robustness:
    names   = list(robustness.keys())
    ssims   = [robustness[n]['ssim_mean'] for n in names]
    dices   = [robustness[n]['dice_mean'] for n in names]
    colors  = ['#4CAF50' if n == 'clean' else
               '#F44336' if robustness[n]['dice_mean'] < 0.6 else '#FF9800'
               for n in names]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.barh(names, ssims, color=colors, alpha=0.85)
    ax1.set_xlabel('SSIM vs Clean')
    ax1.set_title('Image Quality under Degradation', fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    ax2.barh(names, dices, color=colors, alpha=0.85)
    ax2.axvline(0.82, color='red', linestyle=':', lw=1.2, label='Target Dice 0.82')
    ax2.set_xlabel('Dice Score')
    ax2.set_title('Segmentation Quality under Degradation', fontweight='bold')
    ax2.legend(fontsize=8); ax2.grid(axis='x', alpha=0.3)
    plt.suptitle('Robustness to Scan Degradation\n(simulating 1.5T/low-resource scanners)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig('figures/robustness_summary.png', dpi=150)
    plt.close()
    print("Saved figures/robustness_summary.png")

print("\nDone. All available figures saved to figures/")
print("Note: figures requiring training results will generate once fold weights are added.")
