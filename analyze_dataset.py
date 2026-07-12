"""
Exploratory analysis of the TCGA-LGG dataset before nnU-Net conversion.

Usage:
    python analyze_dataset.py --datapath lgg-mri-segmentation/kaggle_3m

Outputs:
    figures/dataset_overview.png
    figures/institution_breakdown.png
    figures/tumor_examples.png
    results/dataset_stats.json
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

os.makedirs('results', exist_ok=True)
os.makedirs('figures', exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument('--datapath', default='lgg-mri-segmentation/kaggle_3m', type=str)
args = parser.parse_args()

# ── Scan all patient directories ──────────────────────────────────────────────
records = []
patient_dirs = sorted([
    d for d in os.listdir(args.datapath)
    if os.path.isdir(os.path.join(args.datapath, d))
])
print(f"Found {len(patient_dirs)} patient directories.")

for patient in tqdm(patient_dirs, desc="Scanning patients"):
    ppath = os.path.join(args.datapath, patient)
    flair_files = sorted([f for f in os.listdir(ppath)
                          if f.endswith('.tif') and '_mask' not in f])
    institution = patient.split('_')[1] if '_' in patient else 'UNK'
    for flair_file in flair_files:
        mask_file = flair_file.replace('.tif', '_mask.tif')
        mask_path = os.path.join(ppath, mask_file)
        if not os.path.exists(mask_path):
            continue
        mask_arr = np.array(Image.open(mask_path).convert('L'))
        has_tumor = int(mask_arr.max() > 0)
        tumor_pixels = int((mask_arr > 0).sum())
        img_arr = np.array(Image.open(os.path.join(ppath, flair_file)).convert('L'))
        records.append({
            'patient': patient,
            'institution': institution,
            'slice': flair_file,
            'has_tumor': has_tumor,
            'tumor_pixels': tumor_pixels,
            'img_mean': float(img_arr.mean()),
            'img_std': float(img_arr.std()),
        })

df = pd.DataFrame(records)
print(f"Total slices: {len(df)}  |  With tumor: {df['has_tumor'].sum()}  |  No tumor: {(~df['has_tumor'].astype(bool)).sum()}")

# ── Per-patient summary ───────────────────────────────────────────────────────
pat_df = df.groupby('patient').agg(
    institution=('institution', 'first'),
    n_slices=('slice', 'count'),
    n_tumor_slices=('has_tumor', 'sum'),
    mean_tumor_pixels=('tumor_pixels', 'mean'),
).reset_index()
pat_df['tumor_prevalence'] = pat_df['n_tumor_slices'] / pat_df['n_slices']

inst_df = pat_df.groupby('institution').agg(
    n_patients=('patient', 'count'),
    n_slices=('n_slices', 'sum'),
    mean_tumor_prevalence=('tumor_prevalence', 'mean'),
).reset_index().sort_values('n_patients', ascending=False)

# ── Save stats ────────────────────────────────────────────────────────────────
stats = {
    'total_patients': int(len(patient_dirs)),
    'total_slices': int(len(df)),
    'tumor_slices': int(df['has_tumor'].sum()),
    'no_tumor_slices': int((~df['has_tumor'].astype(bool)).sum()),
    'institutions': inst_df.to_dict(orient='records'),
    'slices_per_patient_mean': float(pat_df['n_slices'].mean()),
    'slices_per_patient_std': float(pat_df['n_slices'].std()),
}
with open('results/dataset_stats.json', 'w') as f:
    json.dump(stats, f, indent=2)
print("Saved results/dataset_stats.json")

# ── Figure 1: Dataset overview ────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Panel A: patients per institution
axes[0].bar(inst_df['institution'], inst_df['n_patients'], color='#2196F3', alpha=0.85)
axes[0].set_title('Patients per Institution', fontweight='bold')
axes[0].set_xlabel('Institution Code')
axes[0].set_ylabel('Number of Patients')
axes[0].grid(axis='y', alpha=0.3)
for i, (inst, n) in enumerate(zip(inst_df['institution'], inst_df['n_patients'])):
    axes[0].text(i, n + 0.3, str(n), ha='center', fontsize=9)

# Panel B: class balance
counts = [df['has_tumor'].sum(), (~df['has_tumor'].astype(bool)).sum()]
axes[1].bar(['With Tumor', 'No Tumor'], counts, color=['#F44336', '#4CAF50'], alpha=0.85)
axes[1].set_title('Slice Class Balance', fontweight='bold')
axes[1].set_ylabel('Number of Slices')
axes[1].grid(axis='y', alpha=0.3)
for i, c in enumerate(counts):
    axes[1].text(i, c + 20, str(c), ha='center', fontsize=10)

# Panel C: tumor size distribution (non-zero only)
tumor_pixels = df[df['has_tumor'] == 1]['tumor_pixels']
axes[2].hist(tumor_pixels, bins=40, color='#FF9800', alpha=0.85, edgecolor='white')
axes[2].set_title('Tumor Size Distribution\n(tumor slices only)', fontweight='bold')
axes[2].set_xlabel('Tumor Pixels per Slice')
axes[2].set_ylabel('Count')
axes[2].grid(axis='y', alpha=0.3)

plt.suptitle('TCGA-LGG Dataset Overview', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/dataset_overview.png', dpi=150)
plt.close()
print("Saved figures/dataset_overview.png")

# ── Figure 2: Institution breakdown ──────────────────────────────────────────
fig, ax1 = plt.subplots(figsize=(9, 4))
x = range(len(inst_df))
bars = ax1.bar(x, inst_df['n_patients'], color='#2196F3', alpha=0.75, label='Patients')
ax1.set_xticks(list(x))
ax1.set_xticklabels(inst_df['institution'])
ax1.set_ylabel('Number of Patients', color='#2196F3')
ax1.tick_params(axis='y', labelcolor='#2196F3')

ax2 = ax1.twinx()
ax2.plot(list(x), inst_df['mean_tumor_prevalence'] * 100, 'o-',
         color='#F44336', linewidth=2, markersize=6, label='Tumor Prevalence %')
ax2.set_ylabel('Mean Tumor Prevalence (%)', color='#F44336')
ax2.tick_params(axis='y', labelcolor='#F44336')

ax1.set_title('Institution Breakdown: Patient Count vs Tumor Prevalence',
              fontweight='bold')
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=9)
ax1.grid(axis='y', alpha=0.2)
plt.tight_layout()
plt.savefig('figures/institution_breakdown.png', dpi=150)
plt.close()
print("Saved figures/institution_breakdown.png")

# ── Figure 3: Example tumor slices ───────────────────────────────────────────
tumor_rows = df[df['has_tumor'] == 1].sample(6, random_state=42)
fig, axes = plt.subplots(2, 3, figsize=(12, 7))
for ax, (_, row) in zip(axes.flat, tumor_rows.iterrows()):
    ppath = os.path.join(args.datapath, row['patient'])
    img = np.array(Image.open(os.path.join(ppath, row['slice'])).convert('L'))
    mask = np.array(Image.open(
        os.path.join(ppath, row['slice'].replace('.tif', '_mask.tif'))).convert('L'))
    ax.imshow(img, cmap='gray')
    overlay = np.zeros((*img.shape, 4))
    overlay[mask > 0] = [1, 0, 0, 0.45]
    ax.imshow(overlay)
    ax.set_title(f"{row['patient'][:15]}\n{row['tumor_pixels']} px", fontsize=7)
    ax.axis('off')

plt.suptitle('Sample Tumor Slices (FLAIR + mask overlay)', fontweight='bold', fontsize=12)
plt.tight_layout()
plt.savefig('figures/tumor_examples.png', dpi=150)
plt.close()
print("Saved figures/tumor_examples.png")

print("\n=== Dataset Summary ===")
print(f"  Patients:      {stats['total_patients']}")
print(f"  Total slices:  {stats['total_slices']}")
print(f"  Tumor slices:  {stats['tumor_slices']} ({100*stats['tumor_slices']/stats['total_slices']:.1f}%)")
print(f"  Institutions:  {len(inst_df)}")
for _, row in inst_df.iterrows():
    print(f"    {row['institution']}: {row['n_patients']} patients, "
          f"{row['mean_tumor_prevalence']*100:.1f}% tumor prevalence")
