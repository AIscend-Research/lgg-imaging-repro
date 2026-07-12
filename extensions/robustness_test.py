"""
Simulates low-resource scan degradation and tests model robustness.

Degradation types (mimicking older 1.5T scanners and field conditions):
  - Gaussian noise (SNR reduction)
  - Gaussian blur (motion / low-field blur)
  - Downsampling + upsampling (reduced resolution, e.g. 0.5x)
  - Rician noise (MRI-specific noise model)
  - Missing/corrupted slices (zeroed out)
  - Brightness/contrast shift (scanner calibration drift)

Usage:
    python robustness_test.py \
        --datapath lgg-mri-segmentation/kaggle_3m

Outputs:
    results/robustness_results.json
    figures/robustness_comparison.png
"""

import os
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import gaussian_filter
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

os.makedirs('results', exist_ok=True)
os.makedirs('figures', exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument('--datapath',  default='lgg-mri-segmentation/kaggle_3m', type=str)
parser.add_argument('--n_samples', default=100, type=int, help='Slices to test per condition')
parser.add_argument('--seed',      default=42, type=int)
args = parser.parse_args()
np.random.seed(args.seed)

# ── Degradation functions ─────────────────────────────────────────────────────
def add_gaussian_noise(img, sigma=25):
    noise = np.random.normal(0, sigma, img.shape)
    return np.clip(img.astype(float) + noise, 0, 255).astype(np.uint8)

def add_blur(img, sigma=2.0):
    return np.clip(gaussian_filter(img.astype(float), sigma=sigma), 0, 255).astype(np.uint8)

def downsample_upsample(img, factor=0.5):
    h, w = img.shape
    small = np.array(Image.fromarray(img).resize(
        (int(w * factor), int(h * factor)), Image.BILINEAR))
    return np.array(Image.fromarray(small).resize((w, h), Image.BILINEAR))

def add_rician_noise(img, sigma=20):
    img_f = img.astype(float)
    noise1 = np.random.normal(0, sigma, img_f.shape)
    noise2 = np.random.normal(0, sigma, img_f.shape)
    return np.clip(np.sqrt((img_f + noise1)**2 + noise2**2), 0, 255).astype(np.uint8)

def zero_slice(img):
    return np.zeros_like(img)

def brightness_shift(img, shift=40):
    return np.clip(img.astype(float) + shift, 0, 255).astype(np.uint8)

def contrast_reduce(img, factor=0.5):
    mean = img.mean()
    return np.clip(mean + (img.astype(float) - mean) * factor, 0, 255).astype(np.uint8)

DEGRADATIONS = {
    'clean':            lambda x: x,
    'gaussian_noise':   lambda x: add_gaussian_noise(x, sigma=25),
    'heavy_noise':      lambda x: add_gaussian_noise(x, sigma=50),
    'blur_mild':        lambda x: add_blur(x, sigma=1.5),
    'blur_heavy':       lambda x: add_blur(x, sigma=3.0),
    'downsample_0.5x':  lambda x: downsample_upsample(x, factor=0.5),
    'downsample_0.25x': lambda x: downsample_upsample(x, factor=0.25),
    'rician_noise':     lambda x: add_rician_noise(x, sigma=20),
    'missing_slice':    lambda x: zero_slice(x),
    'brightness_shift': lambda x: brightness_shift(x, shift=50),
    'contrast_reduce':  lambda x: contrast_reduce(x, factor=0.5),
}

# ── Load sample tumor slices ──────────────────────────────────────────────────
def load_pairs(datapath, n=100):
    pairs = []
    for patient in sorted(os.listdir(datapath)):
        ppath = os.path.join(datapath, patient)
        if not os.path.isdir(ppath):
            continue
        for f in sorted(os.listdir(ppath)):
            if f.endswith('.tif') and '_mask' not in f:
                mask_f = f.replace('.tif', '_mask.tif')
                mask_path = os.path.join(ppath, mask_f)
                if os.path.exists(mask_path):
                    mask = np.array(Image.open(mask_path).convert('L'))
                    if mask.max() > 0:
                        pairs.append((os.path.join(ppath, f), mask_path))
        if len(pairs) >= n:
            break
    return pairs[:n]

def dice_from_arrays(pred, gt, threshold=128):
    pred_bin = (pred > threshold).astype(int)
    gt_bin   = (gt > 0).astype(int)
    intersection = (pred_bin * gt_bin).sum()
    return 2 * intersection / (pred_bin.sum() + gt_bin.sum() + 1e-8)

# Proxy segmentation (Otsu threshold) — replaced by real nnU-Net once weights available
def mock_segment(img):
    from skimage.filters import threshold_otsu
    try:
        thresh = threshold_otsu(img)
    except Exception:
        thresh = 128
    return (img > thresh).astype(np.uint8) * 255

print("Loading tumor slices...")
pairs = load_pairs(args.datapath, n=args.n_samples)
print(f"Loaded {len(pairs)} tumor slices.")

# ── Run degradation experiments ───────────────────────────────────────────────
results = {}
for deg_name, deg_fn in DEGRADATIONS.items():
    ssim_scores, dice_scores, psnr_scores = [], [], []
    for flair_path, mask_path in tqdm(pairs, desc=deg_name, leave=False):
        clean   = np.array(Image.open(flair_path).convert('L'))
        mask    = np.array(Image.open(mask_path).convert('L'))
        degraded = deg_fn(clean)

        s   = ssim(clean, degraded, data_range=255)
        mse = np.mean((clean.astype(float) - degraded.astype(float))**2)
        psnr = 10 * np.log10(255**2 / (mse + 1e-8))
        pred = mock_segment(degraded)
        d   = dice_from_arrays(pred, mask)

        ssim_scores.append(s)
        psnr_scores.append(psnr)
        dice_scores.append(d)

    results[deg_name] = {
        'ssim_mean': float(np.mean(ssim_scores)),
        'ssim_std':  float(np.std(ssim_scores)),
        'psnr_mean': float(np.mean(psnr_scores)),
        'dice_mean': float(np.mean(dice_scores)),
        'dice_std':  float(np.std(dice_scores)),
    }
    print(f"  {deg_name:<22}: SSIM={results[deg_name]['ssim_mean']:.3f}  "
          f"PSNR={results[deg_name]['psnr_mean']:.1f}dB  "
          f"Dice(proxy)={results[deg_name]['dice_mean']:.3f}")

with open('results/robustness_results.json', 'w') as f:
    json.dump(results, f, indent=2)

# ── Figure: Robustness comparison ─────────────────────────────────────────────
names    = list(results.keys())
ssims    = [results[n]['ssim_mean'] for n in names]
psnrs    = [results[n]['psnr_mean'] for n in names]
dices    = [results[n]['dice_mean'] for n in names]
dice_std = [results[n]['dice_std']  for n in names]

colors = ['#4CAF50' if n == 'clean' else '#F44336' if d < 0.3 else '#FF9800'
          for n, d in zip(names, dices)]

fig, axes = plt.subplots(1, 3, figsize=(17, 5))

axes[0].barh(names, ssims, color=colors, alpha=0.85)
axes[0].axvline(1.0, color='grey', linestyle='--', lw=0.8)
axes[0].set_xlabel('SSIM (vs clean)', fontsize=10)
axes[0].set_title('Image Quality: SSIM', fontsize=11, fontweight='bold')
axes[0].grid(axis='x', alpha=0.3)

axes[1].barh(names, psnrs, color=colors, alpha=0.85)
axes[1].set_xlabel('PSNR (dB)', fontsize=10)
axes[1].set_title('Image Quality: PSNR', fontsize=11, fontweight='bold')
axes[1].grid(axis='x', alpha=0.3)

axes[2].barh(names, dices, xerr=dice_std, color=colors, alpha=0.85, capsize=3)
axes[2].axvline(0.82, color='green', linestyle=':', lw=1.2, label='Target Dice 0.82')
axes[2].set_xlabel('Dice Score (proxy segmentation)', fontsize=10)
axes[2].set_title('Segmentation Quality under Degradation\n(proxy Otsu; replace with nnU-Net)',
                  fontsize=11, fontweight='bold')
axes[2].legend(fontsize=8)
axes[2].grid(axis='x', alpha=0.3)

plt.suptitle('Robustness to Scan Degradation — Simulating 1.5T / Low-Resource Scanners',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/robustness_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

print("\nSaved results/robustness_results.json")
print("Saved figures/robustness_comparison.png")
print("\nNOTE: Replace mock_segment() with actual nnU-Net inference once weights are available.")
