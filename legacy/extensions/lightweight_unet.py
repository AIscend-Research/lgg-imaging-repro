"""
Lightweight U-Net variants for CPU-deployable brain tumor segmentation.

Three variants:
  1. Full U-Net (baseline reference)
  2. Slim U-Net (halved channels)
  3. Micro U-Net (quarter channels, CPU-runnable)

Usage:
    python extensions/lightweight_unet.py \
        --mode    train \
        --variant slim \
        --datapath data/raw/kaggle_3m \
        --epochs  30

    python extensions/lightweight_unet.py --mode benchmark

This module is import-safe: classes/functions are defined at module level and
the training/benchmark code only runs under `if __name__ == '__main__'`, so
distill.py / quantize_model.py / compression_equity.py can import from it
without triggering a training run.
"""

import os
import time
import json
import tempfile
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Architecture ──────────────────────────────────────────────────────────────
CHANNEL_CONFIGS = {
    'full':  [64, 128, 256, 512],
    'slim':  [32,  64, 128, 256],
    'micro': [16,  32,  64, 128],
}

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.block(x)

class UNet(nn.Module):
    def __init__(self, channels, in_ch=1, num_classes=1):
        super().__init__()
        c = channels
        self.enc1 = ConvBlock(in_ch, c[0])
        self.enc2 = ConvBlock(c[0], c[1])
        self.enc3 = ConvBlock(c[1], c[2])
        self.enc4 = ConvBlock(c[2], c[3])
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(c[3], c[3]*2)
        self.up4 = nn.ConvTranspose2d(c[3]*2, c[3], 2, stride=2)
        self.dec4 = ConvBlock(c[3]*2, c[3])
        self.up3 = nn.ConvTranspose2d(c[3], c[2], 2, stride=2)
        self.dec3 = ConvBlock(c[2]*2, c[2])
        self.up2 = nn.ConvTranspose2d(c[2], c[1], 2, stride=2)
        self.dec2 = ConvBlock(c[1]*2, c[1])
        self.up1 = nn.ConvTranspose2d(c[1], c[0], 2, stride=2)
        self.dec1 = ConvBlock(c[0]*2, c[0])
        self.out = nn.Conv2d(c[0], num_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b  = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(b),  e4], 1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        return self.out(d1)

# ── Dataset ───────────────────────────────────────────────────────────────────
class LGGDataset(Dataset):
    def __init__(self, pairs, size=256):
        self.pairs = pairs
        self.img_t = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        self.msk_t = transforms.Compose([
            transforms.Resize((size, size), interpolation=Image.NEAREST),
            transforms.ToTensor()
        ])

    def __len__(self): return len(self.pairs)

    def __getitem__(self, idx):
        fp, mp = self.pairs[idx]
        img  = Image.open(fp).convert('L')
        mask = Image.open(mp).convert('L')
        return self.img_t(img), (self.msk_t(mask) > 0).float()

def collect_pairs(datapath, tumor_only=False):
    pairs = []
    for patient in sorted(os.listdir(datapath)):
        ppath = os.path.join(datapath, patient)
        if not os.path.isdir(ppath): continue
        for f in sorted(os.listdir(ppath)):
            if f.endswith('.tif') and '_mask' not in f:
                mp = os.path.join(ppath, f.replace('.tif', '_mask.tif'))
                if os.path.exists(mp):
                    if tumor_only:
                        m = np.array(Image.open(mp).convert('L'))
                        if m.max() == 0: continue
                    pairs.append((os.path.join(ppath, f), mp))
    return pairs

# ── Loss / metric ─────────────────────────────────────────────────────────────
def dice_loss(pred, target, smooth=1):
    pred   = torch.sigmoid(pred)
    inter  = (pred * target).sum(dim=(2, 3))
    union  = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    return 1 - (2 * inter + smooth) / (union + smooth)

def combined_loss(pred, target):
    return dice_loss(pred, target).mean() + F.binary_cross_entropy_with_logits(pred, target)

def dice_score(pred_logits, target):
    pred = (torch.sigmoid(pred_logits) > 0.5).float()
    inter = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    return ((2 * inter + 1) / (union + 1)).mean().item()

# ── Size / latency helpers ────────────────────────────────────────────────────
def model_size_mb(model):
    with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as tmp:
        path = tmp.name
    torch.save(model.state_dict(), path)
    size = os.path.getsize(path) / 1024 / 1024
    os.remove(path)
    return size

def inference_time_ms(model, device, n=50):
    """Benchmark on `device`, moving the model there and restoring it after."""
    orig = next(model.parameters()).device
    model = model.to(device)
    model.eval()
    dummy = torch.randn(1, 1, 256, 256, device=device)
    with torch.no_grad():
        for _ in range(5): model(dummy)  # warmup
        if device.type == 'cuda': torch.cuda.synchronize()
        start = time.time()
        for _ in range(n): model(dummy)
        if device.type == 'cuda': torch.cuda.synchronize()
        elapsed = (time.time() - start) / n * 1000
    model.to(orig)
    return elapsed


# ── CLI (only runs when executed directly, not on import) ─────────────────────
if __name__ == '__main__':
    import argparse

    os.makedirs('models',  exist_ok=True)
    os.makedirs('results', exist_ok=True)
    os.makedirs('figures', exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',      default='train', choices=['train', 'benchmark'])
    parser.add_argument('--variant',   default='slim',  choices=['full', 'slim', 'micro'])
    parser.add_argument('--datapath',  default='data/raw/kaggle_3m', type=str)
    parser.add_argument('--epochs',    default=30, type=int)
    parser.add_argument('--lr',        default=1e-4, type=float)
    parser.add_argument('--batch',     default=16, type=int)
    parser.add_argument('--seed',      default=42, type=int)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {DEVICE}")

    # ── Benchmark mode ────────────────────────────────────────────────────────
    if args.mode == 'benchmark':
        print("\n--- Architecture Benchmark ---")
        bench = {}
        for name, channels in CHANNEL_CONFIGS.items():
            m = UNet(channels).to(DEVICE)
            n_params = sum(p.numel() for p in m.parameters()) / 1e6
            size_mb  = model_size_mb(m)
            lat_ms   = inference_time_ms(m, DEVICE)
            cpu_lat  = inference_time_ms(m, torch.device('cpu'))
            bench[name] = {'params_M': n_params, 'size_mb': size_mb,
                           'gpu_ms': lat_ms, 'cpu_ms': cpu_lat}
            print(f"  {name:<8}: {n_params:.2f}M params  {size_mb:.1f} MB  "
                  f"GPU: {lat_ms:.1f}ms  CPU: {cpu_lat:.0f}ms")
        with open('results/architecture_benchmark.json', 'w') as f:
            json.dump(bench, f, indent=2)
        print("Saved results/architecture_benchmark.json")
        raise SystemExit(0)

    # ── Training mode ─────────────────────────────────────────────────────────
    print(f"\nTraining {args.variant} U-Net for {args.epochs} epochs...")
    all_pairs = collect_pairs(args.datapath)
    np.random.seed(args.seed)
    np.random.shuffle(all_pairs)
    split = int(0.85 * len(all_pairs))
    train_ds = LGGDataset(all_pairs[:split])
    val_ds   = LGGDataset(all_pairs[split:])
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False, num_workers=0)
    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}")

    model     = UNet(CHANNEL_CONFIGS[args.variant]).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    n_params  = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Parameters: {n_params:.2f}M")

    train_losses, val_dices = [], []
    best_dice = 0
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for imgs, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            optimizer.zero_grad()
            loss = combined_loss(model(imgs), masks)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        model.eval()
        val_dice_total = 0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
                val_dice_total += dice_score(model(imgs), masks)
        vd = val_dice_total / len(val_loader)
        tl = total_loss / len(train_loader)
        train_losses.append(tl)
        val_dices.append(vd)
        print(f"Epoch {epoch+1}/{args.epochs}  Loss: {tl:.4f}  Val Dice: {vd:.4f}")
        if vd > best_dice:
            best_dice = vd
            torch.save(model.state_dict(), f'models/unet_{args.variant}_best.pth')

    print(f"\nBest Val Dice: {best_dice:.4f}")
    size_mb = model_size_mb(model)
    lat_cpu = inference_time_ms(model, torch.device('cpu'))
    print(f"Model size: {size_mb:.2f} MB  CPU latency: {lat_cpu:.0f} ms/slice")

    with open(f'results/unet_{args.variant}_metrics.json', 'w') as f:
        json.dump({'variant': args.variant, 'best_dice': best_dice,
                   'size_mb': size_mb, 'cpu_latency_ms': lat_cpu,
                   'params_M': n_params}, f, indent=2)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(train_losses, color='#F44336', lw=2)
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Combined Loss')
    ax1.set_title(f'{args.variant.capitalize()} U-Net Training Loss')
    ax1.grid(alpha=0.3)
    ax2.plot(val_dices, color='#4CAF50', lw=2)
    ax2.axhline(0.82, color='red', linestyle=':', lw=1, label='Target 0.82')
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Dice Score')
    ax2.set_title('Validation Dice')
    ax2.legend(); ax2.grid(alpha=0.3)
    plt.suptitle(f'{args.variant.capitalize()} U-Net Training Curves', fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'figures/unet_{args.variant}_training.png', dpi=150)
    plt.close()
    print(f"Saved figures/unet_{args.variant}_training.png")
