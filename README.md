# [Re] Automatic segmentation of lower-grade gliomas (TCGA-LGG)

A **ReScience C** partial replication of the **segmentation** result of:

> Buda, M., Saha, A., & Mazurowski, M. A. (2019). *Association of genomic subtypes of
> lower-grade gliomas with shape features automatically extracted by a deep learning
> algorithm.* Computers in Biology and Medicine, 109, 218–225.
> doi:[10.1016/j.compbiomed.2019.05.002](https://doi.org/10.1016/j.compbiomed.2019.05.002)

We reimplement the paper's plain 4-level U-Net from the architecture description (this is an
**independent implementation**, not the authors' code) and reproduce its automatic
FLAIR-abnormality segmentation performance: **mean Dice 0.82, median Dice 0.85, computed
per patient under patient-level cross-validation**. On top of the faithful reproduction we
scaffold low-resource-deployment extensions (robustness, quantization, distillation,
cross-institution) built on the *same* leakage-safe splits.

> **Scope.** This is a **partial replication**: it covers the segmentation result only. The
> paper's headline *radiogenomics* claim (shape features predicting genomic subtype) is out
> of scope unless the optional module in [`src/lgg`](src/lgg) Section 9 is run. See
> **Deviations** below.

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

Download:
```bash
kaggle datasets download -d mateuszbuda/lgg-mri-segmentation
unzip lgg-mri-segmentation.zip -d data/lgg-mri-segmentation
# images live under data/lgg-mri-segmentation/kaggle_3m/TCGA_*/
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

# 4. Per-patient (volume-aggregated) Dice/IoU/HD95 — the headline (R2).
PYTHONPATH=src python -m lgg.cli evaluate

# 5. Extensions (all reuse the same leakage-safe splits).
PYTHONPATH=src python -m lgg.cli run-robustness
PYTHONPATH=src python -m lgg.cli quantize
PYTHONPATH=src python -m lgg.cli distill
PYTHONPATH=src python -m lgg.cli cross-institution

# 6. Write reports/metrics.json + reports/comparison.md + figures.
PYTHONPATH=src python -m lgg.cli report

# Cheap wiring check end-to-end (Section 5), < ~5 min on CPU:
PYTHONPATH=src python -m lgg.cli smoke
```

**On Kaggle, run [`kaggle_reproduction.ipynb`](kaggle_reproduction.ipynb)** — a single
notebook that executes steps 0–6 with checkpoint/resume across the 12-hour session cap and
writes every metric and figure needed to write the paper. See the notebook header for the
few knobs (fold preset, epochs, which extensions to run).

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
| R10 | ReScience packaging, provenance, deviations, metrics.json + comparison.md | this README, `report.py` |

---

## Repository layout

```
src/lgg/
  data/{prepare,splits,dataset}.py   models/unet.py
  metrics.py train.py evaluate.py
  robustness.py quantize.py distill.py cross_institution.py report.py cli.py
tests/test_no_leakage.py             # written before the training loop (R1)
config.yaml  requirements.txt  splits/  reports/  figures/
nnunet/                              # optional comparison module (Section 8)
kaggle_reproduction.ipynb           # single Kaggle notebook, runs everything
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
6. **Radiogenomics associations (paper's main claim) are out of scope** — this is a partial
   replication of the segmentation result. Marked clearly here and in `comparison.md`.

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
