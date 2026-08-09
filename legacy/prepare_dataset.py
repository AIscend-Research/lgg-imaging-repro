"""
Converts the Kaggle TCGA-LGG dataset (kaggle_3m/) to nnU-Net v2 format.

The Kaggle dataset structure:
    kaggle_3m/
        TCGA_CS_4941_19960909/
            TCGA_CS_4941_19960909_1.tif       <- FLAIR slice
            TCGA_CS_4941_19960909_1_mask.tif  <- binary mask
            ...

nnU-Net output structure:
    Dataset001_TCGALGG/
        imagesTr/   TCGALGG_001_0000.nii.gz  (FLAIR, channel 0)
        labelsTr/   TCGALGG_001.nii.gz       (binary mask)
        dataset.json

Usage:
    python prepare_dataset.py --datapath data/raw/kaggle_3m \
                               --outputpath nnunet_data/Dataset001_TCGALGG
"""

import os
import argparse
import json
import glob
import numpy as np
import nibabel as nib
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()
parser.add_argument('--datapath',   default='data/raw/kaggle_3m', type=str)
parser.add_argument('--outputpath', default='nnunet_data/Dataset001_TCGALGG', type=str)
parser.add_argument('--test_split', default=0.1, type=float, help='Fraction held out as test set')
parser.add_argument('--seed',       default=42, type=int)
args = parser.parse_args()

images_tr = os.path.join(args.outputpath, 'imagesTr')
images_ts = os.path.join(args.outputpath, 'imagesTs')
labels_tr = os.path.join(args.outputpath, 'labelsTr')
os.makedirs(images_tr, exist_ok=True)
os.makedirs(images_ts, exist_ok=True)
os.makedirs(labels_tr, exist_ok=True)

# ── Collect all patient folders ───────────────────────────────────────────────
patient_dirs = sorted([
    d for d in os.listdir(args.datapath)
    if os.path.isdir(os.path.join(args.datapath, d))
])
print(f"Found {len(patient_dirs)} patient directories.")

# ── Collect all (flair, mask) pairs ──────────────────────────────────────────
all_pairs = []
for patient in patient_dirs:
    patient_path = os.path.join(args.datapath, patient)
    # FLAIR files: those NOT ending in _mask
    flair_files = sorted([
        f for f in os.listdir(patient_path)
        if f.endswith('.tif') and '_mask' not in f
    ])
    for flair_file in flair_files:
        mask_file = flair_file.replace('.tif', '_mask.tif')
        mask_path = os.path.join(patient_path, mask_file)
        if os.path.exists(mask_path):
            all_pairs.append({
                'patient':    patient,
                'flair_path': os.path.join(patient_path, flair_file),
                'mask_path':  mask_path,
                'slice_name': flair_file.replace('.tif', '')
            })

print(f"Found {len(all_pairs)} (FLAIR, mask) pairs total.")

# ── Separate tumor vs no-tumor for class-balanced split ──────────────────────
def has_tumor(mask_path):
    mask = np.array(Image.open(mask_path).convert('L'))
    return mask.max() > 0

print("Checking tumor presence (this takes ~1 min)...")
tumor_pairs   = [p for p in tqdm(all_pairs) if has_tumor(p['mask_path'])]
notumor_pairs = [p for p in all_pairs if p not in tumor_pairs]
print(f"  With tumor: {len(tumor_pairs)}")
print(f"  No tumor:   {len(notumor_pairs)}")

# ── Train/test split (stratified by tumor presence) ──────────────────────────
# Split at patient level to avoid data leakage
all_patients = list(set(p['patient'] for p in all_pairs))
train_patients, test_patients = train_test_split(
    all_patients, test_size=args.test_split, random_state=args.seed
)
train_set = [p for p in all_pairs if p['patient'] in train_patients]
test_set  = [p for p in all_pairs if p['patient'] in test_patients]
print(f"Train slices: {len(train_set)}  |  Test slices: {len(test_set)}")

# ── Convert and save as NIfTI ────────────────────────────────────────────────
def tif_to_nifti(image_path, out_path):
    img_arr = np.array(Image.open(image_path).convert('L')).astype(np.float32)
    # Add channel and depth dimensions: (H, W) -> (W, H, 1)
    img_arr = img_arr.T[..., np.newaxis]
    nib.save(nib.Nifti1Image(img_arr, np.eye(4)), out_path)

def mask_to_nifti(mask_path, out_path):
    mask_arr = np.array(Image.open(mask_path).convert('L'))
    mask_arr = (mask_arr > 0).astype(np.uint8)
    mask_arr = mask_arr.T[..., np.newaxis]
    nib.save(nib.Nifti1Image(mask_arr, np.eye(4)), out_path)

print("Converting training set to NIfTI...")
training_cases = []
for idx, pair in enumerate(tqdm(train_set)):
    case_id = f"TCGALGG_{idx+1:04d}"
    flair_out = os.path.join(images_tr, f"{case_id}_0000.nii.gz")
    mask_out  = os.path.join(labels_tr, f"{case_id}.nii.gz")
    tif_to_nifti(pair['flair_path'], flair_out)
    mask_to_nifti(pair['mask_path'],  mask_out)
    training_cases.append({
        'case_id':  case_id,
        'patient':  pair['patient'],
        'original': pair['slice_name'],
        'has_tumor': has_tumor(pair['mask_path'])
    })

print("Converting test set to NIfTI...")
for idx, pair in enumerate(tqdm(test_set)):
    case_id = f"TCGALGG_TEST_{idx+1:04d}"
    tif_to_nifti(pair['flair_path'], os.path.join(images_ts, f"{case_id}_0000.nii.gz"))

# ── Write dataset.json ────────────────────────────────────────────────────────
dataset_json = {
    "channel_names": {"0": "FLAIR"},
    "labels": {"background": 0, "tumor": 1},
    "numTraining": len(train_set),
    "file_ending": ".nii.gz",
    "name": "Dataset001_TCGALGG",
    "description": "TCGA-LGG FLAIR segmentation, 2D slices from kaggle_3m",
    "reference": "https://www.kaggle.com/mateuszbuda/lgg-mri-segmentation",
    "licence": "CC BY 4.0",
    "release": "1.0",
    "overwrite_image_reader_writer": "SimpleITKIO"
}
with open(os.path.join(args.outputpath, 'dataset.json'), 'w') as f:
    json.dump(dataset_json, f, indent=2)

# ── Write case metadata (needed for fairness analysis later) ──────────────────
import pandas as pd
meta_df = pd.DataFrame(training_cases)
meta_df['institution'] = meta_df['patient'].str.extract(r'TCGA_([A-Z]+)_')
meta_df.to_csv(os.path.join(args.outputpath, 'case_metadata.csv'), index=False)

print(f"\nDone. Dataset written to {args.outputpath}")
print(f"  Training cases:  {len(train_set)}")
print(f"  Test cases:      {len(test_set)}")
print(f"  dataset.json:    written")
print(f"  case_metadata.csv: written (includes institution labels)")
print(f"\nNext step: nnUNetv2_plan_and_preprocess -d 1 --verify_dataset_integrity")
