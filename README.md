# [Re] Association of genomic subtypes of lower-grade gliomas with shape features (TCGA-LGG)

A **ReScience C** **full replication** of:

> Buda, M., Saha, A., & Mazurowski, M. A. (2019). *Association of genomic subtypes of
> lower-grade gliomas with shape features automatically extracted by a deep learning
> algorithm.* Computers in Biology and Medicine, 109, 218–225.
> doi:[10.1016/j.compbiomed.2019.05.002](https://doi.org/10.1016/j.compbiomed.2019.05.002)

We reimplement the paper's plain 4-level U-Net from the architecture description (this is an
**independent implementation**, not the authors' code) and reproduce **both** parts of the
paper:

1. **Segmentation** — automatic FLAIR-abnormality segmentation, **mean Dice 0.82, median
   0.85, computed per patient under patient-level cross-validation**.
2. **Radiogenomics (the paper's actual headline claim)** — shape features of the segmented
   tumor (bounding ellipsoid volume ratio, angular standard deviation, margin fluctuation)
   are associated with genomic subtype, and the deep-learning masks preserve those
   associations about as well as the manual masks. This is CPU-only statistics on masks we
   already predict, so it adds ~0 GPU-hours; it is what makes this a *full* replication.

> **Scope.** **Full replication.** The core pipeline (`prepare-data → make-splits → train →
> evaluate → shape-features → radiogenomics → figures → report`) reproduces the segmentation
> number *and* the radiogenomics associations, for a total of ~3–5 GPU-hours. GPU-heavy
> extras (robustness, quantization, distillation, cross-institution, nnU-Net) are **not
> required for acceptance** and live in an optional appendix built on the *same* leakage-safe
> splits. See **Deviations** below.

---

## Dataset provenance (please read)

We use Mateusz Buda's **LGG MRI Segmentation** dataset
(`mateuszbuda/lgg-mri-segmentation` on Kaggle): 110 lower-grade-glioma patients, 3,929
slices at 256×256, 3-channel images whose channels are **pre-contrast / FLAIR /
post-contrast**, with a single binary FLAIR-abnormality mask per slice (1,373 tumor slices,
2,556 empty). Institutions (2-letter code after `TCGA_`): DU=45, HT=34, CS=16, FG=14, EZ=1.

**This is a US National Cancer Institute / TCIA collection from five US institutions. It is
not an African dataset and must not be described as one.** (The first annotated African
brain-MRI dataset is BraTS-Africa, Adewole et al. — a different collection.) Note also that
≈108 of these 110 patients also appear in the mainline BraTS training data, which is a
leakage hazard: the reproduction therefore **trains from random initialization and uses no
BraTS-derived weights** (see R4).

The dataset also ships **`data.csv`** — one row per patient with the genomic subtype labels
(`RNASeqCluster`, `MethylationCluster`, `CNCluster`, …) and US-population demographics
(`race`, `ethnicity`, `neoplasm_histologic_grade`, …) that the radiogenomics reproduction
(Section 7) and the fairness figure (F7) consume. Blank cells are handled by dropping only the
affected patient from that specific test.

Download:
```bash
kaggle datasets download -d mateuszbuda/lgg-mri-segmentation
unzip lgg-mri-segmentation.zip -d data/lgg-mri-segmentation
# images live under data/lgg-mri-segmentation/kaggle_3m/TCGA_*/ ; data.csv sits alongside
```

---

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Everything is driven by [`config.yaml`](config.yaml) (paths, channel selection, fold preset,
epochs, seed, batch size). Flags override individual fields.

---

## Run (exact commands)

```bash
# 0. Leakage test MUST pass before any training (R1).
PYTHONPATH=src python -m pytest tests/test_no_leakage.py -v

# 1. Build the per-slice manifest from the Kaggle folder.
PYTHONPATH=src python -m lgg.cli prepare-data

# 2. Patient-level folds -> splits/splits.json (kfold5 default; buda22 for paper-exact).
PYTHONPATH=src python -m lgg.cli make-splits            # or: make-splits --preset buda22

# 3. Train all folds from scratch (checkpoints every epoch; --resume continues a run).
PYTHONPATH=src python -m lgg.cli train                  # or: train --fold 0 --resume

# 4. Per-patient (volume-aggregated) Dice/IoU/HD95 — the segmentation headline (R2).
PYTHONPATH=src python -m lgg.cli evaluate

# 5. Radiogenomics (Section 7) — the full-replication piece, CPU-only, ~0 GPU.
#    shape-features reconstructs each patient's 3D mask (predicted AND ground truth),
#    extracts BEVR / angular-std / margin-fluctuation, and joins data.csv labels;
#    radiogenomics runs Bonferroni-corrected Fisher tests + discrimination AUC for both.
PYTHONPATH=src python -m lgg.cli shape-features
PYTHONPATH=src python -m lgg.cli radiogenomics

# 6. Figures F1–F7 (each also writes its underlying CSV) + reports.
PYTHONPATH=src python -m lgg.cli figures
PYTHONPATH=src python -m lgg.cli report      # metrics.json + comparison.md (FULL replication)

# Cheap wiring check end-to-end (Section 5), < ~5 min on CPU:
PYTHONPATH=src python -m lgg.cli smoke

# --- Optional Section-12 GPU extras (NOT required for acceptance) ------------
PYTHONPATH=src python -m lgg.cli run-robustness
PYTHONPATH=src python -m lgg.cli quantize
PYTHONPATH=src python -m lgg.cli distill
PYTHONPATH=src python -m lgg.cli cross-institution
PYTHONPATH=src python -m lgg.cli report      # re-run to fold the extras into metrics.json
```

**On Kaggle, run [`kaggle_reproduction.ipynb`](kaggle_reproduction.ipynb)** — a single
notebook that executes the full-replication core (steps 0–6) with checkpoint/resume across
the 12-hour session cap and writes every metric and figure needed to write the paper. The
Section-12 GPU extras are in an optional appendix cell, default **off**
(`RUN_OPTIONAL_EXTRAS`). See the notebook header for the few knobs (fold preset, epochs).

---

## What compares to the paper

`reports/comparison.md` puts our **per-patient** mean/median Dice next to the target
**0.82 / 0.85** with signed deltas. The headline is Dice aggregated over each patient's
whole slice set (sum intersections and sums, then Dice), then mean/median over the 110
patients (R2). Per-slice Dice is reported separately and is *never* the headline — with the
2,556 empty slices it is a different, rosier number. IoU and HD95 are additions (the paper
reports neither) and are kept visually distinct (R3). Empty-mask convention: Dice = 1.0 if
prediction and ground truth are both empty, 0.0 if exactly one is empty.

---

## Reproduction requirements (how each is met)

| # | Requirement | Where |
|---|---|---|
| R1 | Patient-level splits, never slice-level | `data/splits.py`, `tests/test_no_leakage.py` (written first) |
| R2 | Per-patient volume-aggregated Dice as headline | `metrics.py`, `evaluate.py` |
| R3 | Paper metric (Dice) first; IoU/HD95 distinct additions | `metrics.py`, `report.py` |
| R4 | Train from scratch, no BraTS weights | `train.py`; `config.train.pretrained_encoder` guarded off |
| R5 | Keep all slices; Dice+BCE loss (+ optional fg sampling) | `metrics.dice_bce_loss`, `train.foreground_bias` |
| R6 | Patient-level k-fold, `buda22` & `kfold5`, seed, saved split | `data/splits.py`, `splits/splits.json` |
| R7 | Seeds, deterministic cuDNN, pinned deps, single config | `utils.set_seed`, `requirements.txt`, `config.yaml` |
| R8 | Params, size, VRAM, GPU/CPU latency per volume | `quantize.benchmark_model` |
| R9 | Explicit channels (3ch pre/FLAIR/post default; FLAIR-only opt) | `data/dataset.py`, `config.data.in_channels` |
| R10 | ReScience packaging, provenance, full-replication statement, deviations, metrics.json + comparison.md | this README, `report.py` |
| R11 | Figures F1–F9 (F1–F7 required, incl. the automatic-vs-manual discrimination ROC; each with underlying CSV), all from existing outputs | `figures.py` |
| §7 | Radiogenomics: shape features + Fisher tests + AUC, predicted & GT masks | `shape_features.py`, `radiogenomics.py` |

---

## Repository layout

```
src/lgg/
  data/{prepare,splits,dataset}.py   models/unet.py
  metrics.py train.py evaluate.py                       # segmentation core
  shape_features.py radiogenomics.py                    # Section 7 (full replication)
  figures.py report.py cli.py                           # figures F1–F7 + reports
  robustness.py quantize.py distill.py cross_institution.py   # optional Section-12 extras
tests/test_no_leakage.py             # written before the training loop (R1)
config.yaml  requirements.txt  splits/  reports/  figures/
nnunet/                              # optional comparison module (Section 12)
kaggle_reproduction.ipynb           # single Kaggle notebook, full-replication core + optional appendix
```

---

## Model & training

Plain 4-level U-Net (two conv+ReLU blocks per level, 2×2 max-pool, transpose-conv decoder
with skip connections, 1×1 output). Batch norm is **on by default** to match the public
reference repo `mateuszbuda/brain-segmentation-pytorch`; `config.train.batchnorm: false`
gives the strictly-paper net. Loss = soft-Dice + BCE. Optimizer: Adam (lr 1e-3),
cosine-annealed. AMP on by default. Default input is 3-channel (pre/FLAIR/post); FLAIR-only
single-channel is a documented option (R9). Checkpoints every epoch; `--resume` continues
across a Kaggle session cap; an ETA to full training is printed after epoch 1.

---

## Deviations (recorded per ReScience, R10)

1. **Compute-friendly default preset.** The default `kfold5` (patient-level 5-fold) replaces
   the paper's exact `buda22` (22 subsets of 5) for turnaround; `make-splits --preset buda22`
   reproduces the paper's protocol exactly. Both are leakage-safe (tests enforce it).
2. **Batch norm on by default.** The paper's own U-Net omitted BN; the widely used public
   repo adds it. We default to the public-repo variant and expose a flag for the paper-exact
   net.
3. **Epoch budget.** Kaggle's 12-hour session cap limits full training; `config.train.epochs`
   is the knob and the notebook resumes across sessions. Any near-miss vs 0.82 is documented,
   not tuned away (Section 5).
4. **3-channel input by default.** The paper used all available sequences; the public
   benchmark ships 3-channel pre/FLAIR/post, which we use for the headline. FLAIR-only is a
   legitimate documented alternative (`config.data.in_channels: 1`).
5. **HD95** requires `medpy`; if it is unavailable the pipeline still runs and reports HD95 as
   not computed.
6. **Shape-feature definitions.** The paper's three shape features trace to its reference [6];
   the exact formulas are not restated in Buda 2019, so we implement the most faithful reading
   (recorded in [`shape_features.py`](src/lgg/shape_features.py)) and target the **direction and
   significance** of the genomic associations, not bit-identical feature values:
   - *Bounding ellipsoid volume ratio (BEVR)* — tumor volume ÷ volume of a PCA-based minimum
     enclosing ellipsoid (scaled to enclose every voxel, so the ratio stays in (0, 1]).
   - *Angular standard deviation* — normalized std of the boundary's radial signature about the
     tumor centroid, on the largest-tumor slice.
   - *Margin fluctuation* — normalized std of the high-frequency residual of that signature
     (after a low-order Fourier smooth).
7. **Isotropic voxels.** The Kaggle dataset ships no per-slice spacing, so voxels are treated as
   isotropic unit cubes. Every shape feature used is a dimensionless ratio or normalized
   dispersion, hence scale-invariant to an isotropic voxel size.
8. **Fisher one-vs-rest + Bonferroni.** Multi-cluster subtypes are tested one cluster-value
   against the rest (the paper reports exactly this "cluster R2 vs rest" style of contrast), and
   the full family of (feature × subtype × cluster) tests is Bonferroni-corrected.

Hardware we ran on: _fill in after your Kaggle run_ (e.g. Kaggle P100 16 GB / T4×2, 12-hour
sessions). The notebook prints per-epoch wall-clock and a measured ETA so real compute is
recorded rather than guessed.

---

## Citation

```bibtex
@article{buda2019association,
  title={Association of genomic subtypes of lower-grade gliomas with shape features
         automatically extracted by a deep learning algorithm},
  author={Buda, Mateusz and Saha, Ashirbani and Mazurowski, Maciej A},
  journal={Computers in Biology and Medicine}, volume={109}, pages={218--225}, year={2019}
}
```
