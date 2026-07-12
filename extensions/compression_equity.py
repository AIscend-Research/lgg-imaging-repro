"""
Equity-of-compression analysis: do lightweight / distilled models degrade
DISPROPORTIONATELY on the hardest cases (small tumors)?

Loads every trained checkpoint in models/ and evaluates per-slice Dice on the
shared validation split, bucketed by tumor size. This tests whether model
compression trades away accuracy unevenly across the difficulty spectrum --
a key responsible-AI question for low-resource deployment.

Usage (after lightweight_unet.py + distill.py have produced checkpoints):
    python extensions/compression_equity.py --datapath data/raw/kaggle_3m

Outputs:
    results/compression_equity.json
    figures/compression_equity.png
"""

import os
import re
import json
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from lightweight_unet import (UNet, LGGDataset, collect_pairs, CHANNEL_CONFIGS)

os.makedirs('results', exist_ok=True)
os.makedirs('figures', exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument('--datapath', default='data/raw/kaggle_3m', type=str)
parser.add_argument('--seed',     default=42, type=int)
args = parser.parse_args()
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Map checkpoint filename -> (display label, channel variant)
CANDIDATES = [
    ('models/unet_full_best.pth',            'Full',            'full'),
    ('models/unet_slim_best.pth',            'Slim',            'slim'),
    ('models/unet_micro_best.pth',           'Micro',           'micro'),
    ('models/unet_micro_distilled_best.pth', 'Micro (distill)', 'micro'),
    ('models/unet_slim_distilled_best.pth',  'Slim (distill)',  'slim'),
]

SIZE_BINS = [0, 200, 500, 1000, 2000, np.inf]
SIZE_LABELS = ['XS (<200)', 'S (200-500)', 'M (500-1k)', 'L (1k-2k)', 'XL (>2k)']

def per_slice_dice(logits, mask, smooth=1.0):
    pred = (torch.sigmoid(logits) > 0.5).float()
    inter = (pred * mask).sum().item()
    denom = pred.sum().item() + mask.sum().item()
    return (2 * inter + smooth) / (denom + smooth)

# ── Shared validation split (identical to the training scripts) ───────────────
pairs = collect_pairs(args.datapath)
np.random.seed(args.seed)
np.random.shuffle(pairs)
val_pairs = pairs[int(0.85 * len(pairs)):]
val_ds = LGGDataset(val_pairs)
loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)
print(f"Validation slices: {len(val_ds)}")

# Precompute tumor size (gt pixels) per slice once
sizes = []
for _, mask in loader:
    sizes.append(float(mask.sum().item()))
sizes = np.array(sizes)
buckets = np.digitize(sizes, SIZE_BINS[1:-1])  # 0..4

results = {}
for ckpt, label, variant in CANDIDATES:
    if not os.path.exists(ckpt):
        continue
    model = UNet(CHANNEL_CONFIGS[variant]).to(DEVICE)
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=False))
    model.eval()

    dices = []
    with torch.no_grad():
        for imgs, masks in loader:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            dices.append(per_slice_dice(model(imgs), masks))
    dices = np.array(dices)

    # Only tumor-bearing slices for the size-stratified Dice
    tumor_mask = sizes > 0
    by_bucket = {}
    for b, blabel in enumerate(SIZE_LABELS):
        sel = (buckets == b) & tumor_mask
        if sel.sum() > 0:
            by_bucket[blabel] = {'dice': float(dices[sel].mean()),
                                 'n': int(sel.sum())}
    results[label] = {
        'overall_tumor_dice': float(dices[tumor_mask].mean()),
        'by_size': by_bucket,
    }
    print(f"{label:<16}: tumor Dice={results[label]['overall_tumor_dice']:.4f}  "
          + "  ".join(f"{k}={v['dice']:.3f}" for k, v in by_bucket.items()))

if not results:
    print("No checkpoints found in models/. Train lightweight_unet.py + distill.py first.")
    raise SystemExit(0)

with open('results/compression_equity.json', 'w') as f:
    json.dump(results, f, indent=2)
print("\nSaved results/compression_equity.json")

# ── Figure: Dice vs tumor size, one line per model ───────────────────────────
fig, ax = plt.subplots(figsize=(9, 5.5))
present_labels = [l for l in SIZE_LABELS
                  if any(l in r['by_size'] for r in results.values())]
for label, r in results.items():
    ys = [r['by_size'].get(l, {}).get('dice', np.nan) for l in present_labels]
    ax.plot(present_labels, ys, 'o-', lw=2, markersize=6, label=label)
ax.axhspan(0.82, 0.92, alpha=0.08, color='green')
ax.set_xlabel('Tumor size (ground-truth pixels)')
ax.set_ylabel('Dice (validation, tumor slices)')
ax.set_title('Equity of Compression: Dice vs Tumor Size by Model\n'
             'Do smaller/distilled models lose more on hard (small-tumor) cases?',
             fontweight='bold')
ax.set_ylim([0, 1.0])
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('figures/compression_equity.png', dpi=150)
plt.close()
print("Saved figures/compression_equity.png")
