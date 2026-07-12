"""
Post-training quantization (FP16 and INT8) of the lightweight U-Net.

Usage (run after training lightweight_unet.py):
    python extensions/quantize_model.py \
        --model_path models/unet_slim_best.pth \
        --variant    slim \
        --datapath   data/raw/kaggle_3m

Outputs:
    models/unet_slim_fp16.pth
    models/unet_slim_int8.pth
    deploy/unet_slim.onnx
    deploy/unet_slim_int8.onnx
    results/quantization_results.json
    figures/quantization_comparison.png
"""

import os
import json
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from lightweight_unet import UNet, LGGDataset, collect_pairs, dice_score, CHANNEL_CONFIGS

os.makedirs('models',  exist_ok=True)
os.makedirs('deploy',  exist_ok=True)
os.makedirs('results', exist_ok=True)
os.makedirs('figures', exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', default='models/unet_slim_best.pth', type=str)
parser.add_argument('--variant',    default='slim', choices=['full', 'slim', 'micro'])
parser.add_argument('--datapath',   default='data/raw/kaggle_3m', type=str)
args = parser.parse_args()

DEVICE = torch.device('cpu')

def model_file_size(path):
    return os.path.getsize(path) / 1024 / 1024

def measure_latency(model, n=50, half=False, device='cpu'):
    model.eval()
    dummy = torch.randn(1, 1, 256, 256).to(device)
    if half: dummy = dummy.half()
    with torch.no_grad():
        for _ in range(5): model(dummy)
        if device == 'cuda': torch.cuda.synchronize()
        t = time.time()
        for _ in range(n): model(dummy)
        if device == 'cuda': torch.cuda.synchronize()
    return (time.time() - t) / n * 1000

# ── Load model and test data ──────────────────────────────────────────────────
if not os.path.exists(args.model_path):
    print(f"Model not found: {args.model_path}. Run lightweight_unet.py --mode train first.")
    exit(1)

print("Loading model...")
model_fp32 = UNet(CHANNEL_CONFIGS[args.variant])
model_fp32.load_state_dict(torch.load(args.model_path, map_location='cpu', weights_only=False))
model_fp32.eval()

pairs = collect_pairs(args.datapath)
np.random.seed(42)
np.random.shuffle(pairs)  # match lightweight_unet.py / distill.py split exactly
val_pairs = pairs[int(0.85*len(pairs)):]
val_ds = LGGDataset(val_pairs)
val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=0)

def evaluate(model, loader, half=False, device='cpu'):
    model.eval()
    total_dice = 0
    with torch.no_grad():
        for imgs, masks in loader:
            imgs = imgs.to(device)
            if half: imgs = imgs.half()
            preds = model(imgs).float().cpu()
            total_dice += dice_score(preds, masks)
    return total_dice / len(loader)

torch.save(model_fp32.state_dict(), f'models/unet_{args.variant}_fp32.pth')
fp32_size   = model_file_size(f'models/unet_{args.variant}_fp32.pth')
fp32_lat    = measure_latency(model_fp32)
fp32_dice   = evaluate(model_fp32, val_loader)
print(f"FP32: Dice={fp32_dice:.4f}  Size={fp32_size:.2f} MB  Latency={fp32_lat:.0f} ms")

# ── FP16 ──────────────────────────────────────────────────────────────────────
# FP16 conv/batchnorm are not implemented on CPU in PyTorch, so FP16 dice/latency
# are measured on GPU when available; size (the reliable win) is always reported.
model_fp16 = UNet(CHANNEL_CONFIGS[args.variant])
model_fp16.load_state_dict(torch.load(args.model_path, map_location='cpu', weights_only=False))
model_fp16 = model_fp16.half().eval()
torch.save(model_fp16.state_dict(), f'models/unet_{args.variant}_fp16.pth')
fp16_size = model_file_size(f'models/unet_{args.variant}_fp16.pth')
if torch.cuda.is_available():
    model_fp16 = model_fp16.to('cuda')
    fp16_lat  = measure_latency(model_fp16, half=True, device='cuda')
    fp16_dice = evaluate(model_fp16, val_loader, half=True, device='cuda')
    print(f"FP16: Dice={fp16_dice:.4f}  Size={fp16_size:.2f} MB  Latency={fp16_lat:.0f} ms (GPU)")
else:
    fp16_lat, fp16_dice = None, None
    print(f"FP16: Size={fp16_size:.2f} MB  (dice/latency skipped — FP16 needs GPU)")

# ── INT8 Dynamic Quantization ─────────────────────────────────────────────────
# NOTE: dynamic quantization only supports Linear/LSTM (not Conv2d). A conv-heavy
# U-Net therefore sees little size change here; FP16 is the reliable size win, and
# genuine INT8 conv compression would require static (calibrated) quantization.
model_int8 = torch.quantization.quantize_dynamic(
    model_fp32, {nn.Linear}, dtype=torch.qint8
)
torch.save(model_int8.state_dict(), f'models/unet_{args.variant}_int8.pth')
int8_size   = model_file_size(f'models/unet_{args.variant}_int8.pth')
int8_lat    = measure_latency(model_int8)
int8_dice   = evaluate(model_int8, val_loader)
print(f"INT8: Dice={int8_dice:.4f}  Size={int8_size:.2f} MB  Latency={int8_lat:.0f} ms")

# ── ONNX Export ───────────────────────────────────────────────────────────────
dummy = torch.randn(1, 1, 256, 256)
onnx_path = f'deploy/unet_{args.variant}.onnx'
torch.onnx.export(model_fp32, dummy, onnx_path,
                  input_names=['image'], output_names=['logits'],
                  opset_version=17, dynamo=False)
onnx_size = model_file_size(onnx_path)
print(f"ONNX: Size={onnx_size:.2f} MB  (Android deployment)")

# ── Results ───────────────────────────────────────────────────────────────────
results = {
    'FP32':  {'dice': fp32_dice, 'size_mb': fp32_size, 'latency_ms': fp32_lat},
    'FP16':  {'dice': fp16_dice, 'size_mb': fp16_size, 'latency_ms': fp16_lat},
    'INT8':  {'dice': int8_dice, 'size_mb': int8_size, 'latency_ms': int8_lat},
    'ONNX':  {'size_mb': onnx_size},
}
with open(f'results/quantization_{args.variant}_results.json', 'w') as f:
    json.dump(results, f, indent=2)

# ── Figure ────────────────────────────────────────────────────────────────────
labels = ['FP32', 'FP16', 'INT8']
nan = float('nan')
def safe(v): return nan if v is None else v
dices  = [safe(fp32_dice), safe(fp16_dice), safe(int8_dice)]
sizes  = [safe(fp32_size), safe(fp16_size), safe(int8_size)]
lats   = [safe(fp32_lat),  safe(fp16_lat),  safe(int8_lat)]

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
colors = ['#2196F3', '#E91E63', '#FF9800']
for ax, vals, title, ylabel in zip(axes,
        [dices, sizes, lats],
        ['Dice Score', 'Model Size (MB)', 'CPU/GPU Latency (ms)'],
        ['Dice', 'MB', 'ms']):
    bars = ax.bar(labels, vals, color=colors, alpha=0.85)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_ylabel(ylabel)
    for bar, v in zip(bars, vals):
        if v != v:  # NaN
            continue
        ax.text(bar.get_x() + bar.get_width()/2, v * 1.02,
                f'{v:.3f}' if ylabel == 'Dice' else f'{v:.1f}',
                ha='center', fontsize=9)
    ax.grid(axis='y', alpha=0.3)
axes[0].axhline(0.82, color='red', linestyle=':', lw=1, label='Target 0.82')
axes[0].legend(fontsize=8)
plt.suptitle(f'Quantization Results: {args.variant.capitalize()} U-Net', fontweight='bold')
plt.tight_layout()
plt.savefig(f'figures/quantization_{args.variant}_comparison.png', dpi=150)
plt.close()
print(f"Saved figures/quantization_{args.variant}_comparison.png")
print(f"Saved results/quantization_{args.variant}_results.json")
