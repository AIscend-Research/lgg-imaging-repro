"""
Knowledge distillation: Full U-Net (teacher) -> Micro U-Net (student).

Trains a Micro U-Net to mimic a trained Full U-Net's soft predictions while
also matching ground-truth masks. The goal is a CPU-deployable student that
recovers more accuracy than training Micro from scratch.

Usage (run after lightweight_unet.py has trained the Full teacher):
    python extensions/distill.py \
        --teacher_path models/unet_full_best.pth \
        --student      micro \
        --datapath     data/raw/kaggle_3m \
        --epochs       30

Outputs:
    models/unet_<student>_distilled_best.pth
    results/distill_<student>_metrics.json
    figures/distill_<student>_training.png
"""

import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from lightweight_unet import (UNet, LGGDataset, collect_pairs, dice_score,
                              combined_loss, model_size_mb, inference_time_ms,
                              CHANNEL_CONFIGS)

os.makedirs('models',  exist_ok=True)
os.makedirs('results', exist_ok=True)
os.makedirs('figures', exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument('--teacher_path', default='models/unet_full_best.pth', type=str)
parser.add_argument('--student',      default='micro', choices=['slim', 'micro'])
parser.add_argument('--datapath',     default='data/raw/kaggle_3m', type=str)
parser.add_argument('--epochs',       default=30, type=int)
parser.add_argument('--lr',           default=1e-4, type=float)
parser.add_argument('--batch',        default=16, type=int)
parser.add_argument('--alpha',        default=0.5, type=float,
                    help='weight on distillation loss vs ground-truth loss')
parser.add_argument('--temperature',  default=2.0, type=float)
parser.add_argument('--seed',         default=42, type=int)
args = parser.parse_args()

torch.manual_seed(args.seed)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")

if not os.path.exists(args.teacher_path):
    print(f"Teacher not found: {args.teacher_path}. "
          f"Run lightweight_unet.py --mode train --variant full first.")
    raise SystemExit(1)

# ── Teacher (frozen) ──────────────────────────────────────────────────────────
teacher = UNet(CHANNEL_CONFIGS['full']).to(DEVICE)
teacher.load_state_dict(torch.load(args.teacher_path, map_location=DEVICE, weights_only=False))
teacher.eval()
for p in teacher.parameters():
    p.requires_grad = False
print(f"Loaded teacher from {args.teacher_path}")

# ── Student ───────────────────────────────────────────────────────────────────
student = UNet(CHANNEL_CONFIGS[args.student]).to(DEVICE)
n_params = sum(p.numel() for p in student.parameters()) / 1e6
print(f"Student ({args.student}): {n_params:.2f}M params")

# ── Data (same split convention as lightweight_unet.py) ──────────────────────
all_pairs = collect_pairs(args.datapath)
np.random.seed(args.seed)
np.random.shuffle(all_pairs)
split = int(0.85 * len(all_pairs))
train_loader = DataLoader(LGGDataset(all_pairs[:split]), batch_size=args.batch,
                          shuffle=True, num_workers=0)
val_loader   = DataLoader(LGGDataset(all_pairs[split:]), batch_size=args.batch,
                          shuffle=False, num_workers=0)
print(f"Train: {split}  Val: {len(all_pairs) - split}")

# ── Distillation loss ─────────────────────────────────────────────────────────
def distillation_loss(student_logits, teacher_logits, masks, alpha, T):
    # Soft-target term: student matches teacher's temperature-scaled probabilities
    soft_teacher = torch.sigmoid(teacher_logits / T)
    soft_student = torch.sigmoid(student_logits / T)
    distill = F.binary_cross_entropy(soft_student, soft_teacher) * (T * T)
    # Hard-target term: student matches ground truth
    hard = combined_loss(student_logits, masks)
    return alpha * distill + (1 - alpha) * hard

# ── Train ─────────────────────────────────────────────────────────────────────
optimizer = torch.optim.Adam(student.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

train_losses, val_dices = [], []
best_dice = 0.0
for epoch in range(args.epochs):
    student.train()
    total = 0.0
    for imgs, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
        with torch.no_grad():
            t_logits = teacher(imgs)
        optimizer.zero_grad()
        s_logits = student(imgs)
        loss = distillation_loss(s_logits, t_logits, masks, args.alpha, args.temperature)
        loss.backward()
        optimizer.step()
        total += loss.item()
    scheduler.step()

    student.eval()
    vd = 0.0
    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            vd += dice_score(student(imgs), masks)
    vd /= len(val_loader)
    tl = total / len(train_loader)
    train_losses.append(tl)
    val_dices.append(vd)
    print(f"Epoch {epoch+1}/{args.epochs}  Loss: {tl:.4f}  Val Dice: {vd:.4f}")
    if vd > best_dice:
        best_dice = vd
        torch.save(student.state_dict(), f'models/unet_{args.student}_distilled_best.pth')

print(f"\nBest distilled Val Dice: {best_dice:.4f}")
size_mb = model_size_mb(student)
cpu_lat = inference_time_ms(student, torch.device('cpu'))
print(f"Size: {size_mb:.2f} MB  CPU latency: {cpu_lat:.0f} ms/slice")

with open(f'results/distill_{args.student}_metrics.json', 'w') as f:
    json.dump({'student': args.student, 'best_dice': best_dice,
               'size_mb': size_mb, 'cpu_latency_ms': cpu_lat,
               'params_M': n_params, 'alpha': args.alpha,
               'temperature': args.temperature,
               'teacher': args.teacher_path}, f, indent=2)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
ax1.plot(train_losses, color='#9C27B0', lw=2)
ax1.set_xlabel('Epoch'); ax1.set_ylabel('Distillation Loss')
ax1.set_title(f'{args.student.capitalize()} Distillation Loss'); ax1.grid(alpha=0.3)
ax2.plot(val_dices, color='#4CAF50', lw=2)
ax2.axhline(0.82, color='red', linestyle=':', lw=1, label='Target 0.82')
ax2.set_xlabel('Epoch'); ax2.set_ylabel('Dice'); ax2.set_title('Validation Dice')
ax2.legend(); ax2.grid(alpha=0.3)
plt.suptitle(f'Knowledge Distillation: Full -> {args.student.capitalize()}',
             fontweight='bold')
plt.tight_layout()
plt.savefig(f'figures/distill_{args.student}_training.png', dpi=150)
plt.close()
print(f"Saved figures/distill_{args.student}_training.png")
print(f"Saved results/distill_{args.student}_metrics.json")
