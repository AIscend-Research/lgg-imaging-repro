# Reproducing and Extending nnU-Net on TCGA-LGG: Lightweight Models for Brain Tumor Segmentation in Low-Resource African Healthcare Settings

**Authors:** [Team names]
**Date:** June 2026
**Repository:** https://github.com/adithyabalakumar007-ai/mlrc-tracks-TCGA-LGG-comm

---

## Abstract

Brain tumors disproportionately burden populations in low- and middle-income countries, yet the most capable AI diagnostic models are designed for well-resourced hospital infrastructure. In this reproducibility study, we reproduce the nnU-Net 2D full-resolution baseline on the TCGA-LGG MRI Segmentation Dataset — 110 lower-grade glioma (LGG) patients across 5 institutions from The Cancer Genome Atlas, totalling 3,929 FLAIR slices. Across 3-fold cross-validation we achieve a mean Dice of **0.840 ± 0.011** on tumor-bearing slices (**0.796 ± 0.021** over all validation slices), landing **within** the 0.82–0.92 range reported in prior literature and confirming a successful reproduction. We then conduct an equity analysis that is the core contribution of this work: while the model is **demographically equitable** (Dice varies by <0.02 across gender, age, and histologic grade), it exhibits large and consistent disparities by **tumor size** (a 0.37 Dice gap between the smallest and largest tumors), **institution** (a 0.09 gap, with the under-represented TCGA_CS center worst at 0.777), and **genomic subtype** (a 0.077 gap, RNASeq cluster 3 worst at 0.782). We additionally characterize image-quality degradation under simulated 1.5T scanner conditions (noise, blur, downsampling, missing slices). Finally, we show that a **Micro U-Net (1.94M parameters, 7.5 MB)** matches the Full U-Net's validation Dice (0.864 vs 0.868) at **16× fewer parameters and 8.5× faster CPU inference (67 ms/slice)**, and that **FP16 quantization halves model size with no accuracy loss** — making CPU-only, smartphone-scale deployment feasible — while crucially *not* worsening performance on the hardest small-tumor cases. All code, metrics, and figures are released openly.

**Keywords:** Brain tumor segmentation, nnU-Net, TCGA-LGG, fairness, equity, low-resource healthcare, Africa

---

## 1. Introduction

### 1.1 Brain Tumors in Low-Resource Settings

Brain tumors represent a disproportionate burden in sub-Saharan Africa and South Asia, where neurosurgical infrastructure is severely limited. The WHO estimates fewer than 1 neurosurgeon per 1 million people across much of Africa, compared to 2–5 per 100,000 in high-income countries. Accurate, early-stage tumor segmentation from MRI is critical for treatment planning — it determines surgical approach, radiotherapy target volumes, and chemotherapy eligibility. Without automated tools, this relies entirely on scarce neuroradiology expertise.

MRI availability itself is a barrier: while high-income countries operate 3T scanners with DICOM-standard workflows, many African hospitals rely on 1.5T machines that are poorly calibrated, producing noisier images with lower resolution and signal-to-noise ratio. Any deployed AI system must be robust to this image quality degradation.

### 1.2 nnU-Net as a Generalizable Baseline

nnU-Net (Isensee et al., 2021) is a self-configuring deep learning segmentation framework that automatically adapts its architecture, preprocessing, and training to any given dataset's properties. It has achieved state-of-the-art results across more than 20 medical image segmentation benchmarks without task-specific hyperparameter tuning. Its 2D configuration, applied to individual FLAIR slices, is the appropriate choice for the TCGA-LGG dataset, which is distributed as 2D slices rather than volumetric acquisitions.

### 1.3 The TCGA-LGG Dataset

The Kaggle TCGA-LGG dataset (Buda et al., 2019) provides 3,929 FLAIR MRI slices from 110 LGG patients at 5 TCIA institutions: **TCGA_CS, TCGA_DU, TCGA_FG, TCGA_HT, and TCGA_EZ**. Of these slices, 1,373 (34.9%) contain visible tumor regions (binary masks provided). Each patient additionally carries genomic and clinical annotations (RNASeq/methylation/miRNA clusters, histologic grade, gender, age, race), enabling subgroup equity analysis. The institution labels serve as a proxy for geographic and resource diversity — institutions contributing fewer patients represent under-resourced settings in our equity analysis.

### 1.4 Contributions

1. **Reproducibility** — We reproduce nnU-Net 2D on TCGA-LGG via 3-fold cross-validation with fully documented hyperparameters, preprocessing, and deviations (notably a reduced 200-epoch budget imposed by Kaggle's 9-hour GPU session limit).
2. **Equity analysis (primary contribution)** — We measure segmentation performance across institution, tumor size, genomic subtype, histologic grade, gender, and age, separating *demographic* equity from *clinical/structural* disparity.
3. **Robustness characterization** — Image-quality degradation (SSIM/PSNR) under 11 simulated low-field scanner conditions.
4. **Responsible-AI discussion** — Ethical implications of deploying AI segmentation without radiologist oversight.
5. **Lightweight deployment study** — Slim/Micro U-Net variants, FP16/INT8 quantization, knowledge distillation, and an equity-of-compression analysis, showing a 1.94M-parameter Micro model matches the Full model's Dice at 16× the efficiency without harming hard cases.

---

## 2. Related Work

### 2.1 Medical Image Segmentation

Deep learning-based medical image segmentation has been dominated by the U-Net architecture (Ronneberger et al., 2015), which uses skip connections between encoder and decoder to preserve spatial information. Variants including Attention U-Net, TransUNet, and Swin-UNetR have progressively improved benchmark performance, but require significant GPU memory and are impractical for CPU-only deployment.

nnU-Net (Isensee et al., 2021) represents a paradigm shift: instead of novel architectural innovations, it automates the engineering decisions (patch size, batch size, normalization, augmentation) that typically require expert tuning. On the Medical Segmentation Decathlon, nnU-Net outperformed all specialized methods across 10 diverse tasks.

### 2.2 Brain Tumor Segmentation

The BraTS challenge has driven advances in glioma segmentation, with top methods exceeding 0.90 Dice on whole-tumor regions. For lower-grade gliomas specifically, Buda et al. (2019) established the TCGA-LGG benchmark and showed CNNs can segment LGG with Dice > 0.88. Prior nnU-Net work on TCGA-LGG reports Dice in the 0.82–0.92 range depending on fold configuration and preprocessing.

### 2.3 Fairness in Medical AI

Obermeyer et al. (2019) demonstrated that commercial healthcare algorithms systematically underserved Black patients due to biased training proxies. In medical imaging, disparities in representation across scanning sites, patient demographics, and disease subtypes translate directly to performance disparities. Achieving equitable AI-assisted diagnosis requires explicit measurement and mitigation of these gaps — the focus of this work.

---

## 3. Methodology

### 3.1 Dataset

We use the TCGA-LGG dataset as released on Kaggle (`mateuszbuda/lgg-mri-segmentation`): 3,929 FLAIR slices from 110 patients at 5 institutions. We apply a patient-level 90/10 train/test split (seed=42) to avoid leakage across slices from the same patient, yielding **3,495 training slices** and **434 held-out test slices**. All reproduction metrics below are computed on nnU-Net's internal cross-validation splits of the 3,495-slice training partition (see §3.3); the 434-slice held-out set is reserved for future final-model evaluation.

**Class balance:** 1,373 of 3,929 slices (34.9%) contain tumor.

**Institution breakdown** (from `analyze_dataset.py`):

| Institution | Patients | Slices | Mean tumor prevalence |
|---|---|---|---|
| TCGA_DU | 45 | 1,878 | 33.1% |
| TCGA_HT | 34 | 1,029 | 40.6% |
| TCGA_CS | 16 | 358 | 33.9% |
| TCGA_FG | 14 | 640 | 34.3% |
| TCGA_EZ | 1 | 24 | 25.0% |

The dataset is heavily imbalanced across institutions: TCGA_DU contributes 45 patients while TCGA_EZ contributes a single patient. This imbalance underpins the equity analysis in §4.4.

### 3.2 Preprocessing

Images are converted from TIF (256×256) to NIfTI (`.nii.gz`) for nnU-Net compatibility via `prepare_dataset.py`. Pixel values are treated as single-channel grayscale (FLAIR). Masks are binarized (pixel > 0 → tumor). nnU-Net's automated planning (`nnUNetv2_plan_and_preprocess`) determined: patch size 256×256, batch size 49, Z-score normalization, and a 7-stage PlainConvUNet with features (32, 64, 128, 256, 512, 512, 512).

### 3.3 nnU-Net 2D Baseline

We use nnU-Net v2 in 2D full-resolution mode with planner-determined hyperparameters:
- Architecture: 2D U-Net, 7 resolution stages
- Loss: Dice + Cross-entropy (batch-Dice enabled)
- Optimizer: SGD with Nesterov momentum (lr = 0.01, weight decay = 3e-5), polynomial LR decay
- Augmentation: nnU-Net default (rotation, scaling, elastic deformation, mirroring, Gaussian noise/blur, brightness/contrast)
- **Training budget: 200 epochs** (see deviation note below), ~137 s/epoch on a Kaggle Tesla T4
- Cross-validation: folds 0, 1, 2 of nnU-Net's internal 5-fold split (each fold: ~2,800 train / ~700 validation slices)

**Documented deviations from the canonical nnU-Net recipe:**
1. **200 epochs instead of 1,000.** The full 1,000-epoch schedule (~38 h) exceeds Kaggle's 9-hour GPU session limit. We used a custom `nnUNetTrainer_200epochs` subclass. Validation pseudo-Dice had already reached ~0.81 by epoch 3 and plateaued well before epoch 200, so the reduction is not expected to materially change the headline result, but it is a genuine deviation and a likely source of any small gap below the upper end of the target range.
2. **`torch.compile` disabled / accelerator switch.** Initial runs on a Tesla P100 (CUDA capability 6.0) failed because the installed PyTorch's Triton backend requires capability ≥ 7.0; we switched to a Tesla T4.
3. **Data-augmentation worker count** reduced (`nnUNet_n_proc_DA=2`) to fit Kaggle RAM.

### 3.4 Subgroup Equity Analysis

We pool per-case validation Dice from folds 0–2 (covering 2,093 of 3,495 training slices, i.e. 3 of 5 CV folds), map each slice to its patient, and join the per-patient clinical/genomic metadata (`data.csv`). To prevent patients with many slices from dominating, we compute a per-patient mean Dice (tumor slices only) before grouping. We then report Dice by institution, tumor size, genomic subtype (RNASeqCluster), histologic grade, gender, and age group (`subgroup_analysis.py`).

### 3.5 Robustness Testing

We simulate 11 degradation conditions on FLAIR slices (`robustness_test.py`): clean; Gaussian noise (σ=25, 50); blur (σ=1.5, 3.0); downsample 0.5×/0.25×; Rician noise (σ=20); missing/zeroed slice; brightness shift (+50); contrast reduction (50%). For each we report SSIM and PSNR versus the clean image. **Note:** segmentation Dice under degradation currently uses an Otsu-threshold *proxy* segmenter, not nnU-Net inference, so only the SSIM/PSNR image-quality results are reported as final; real degraded-input inference is listed as future work (§6).

### 3.6 Lightweight, Quantization, and Distillation Extensions

We implement three plain U-Net variants (`extensions/lightweight_unet.py`) with progressively halved channel widths — Full [64,128,256,512], Slim [32,64,128,256], Micro [16,32,64,128] — trained with Adam (lr=1e-4), cosine-annealed, combined Dice+BCE loss, 20 epochs, on a slice-level 85/15 split (seed=42). **Note:** unlike the patient-level nnU-Net split, this slice-level split and the shorter schedule make the lightweight Dice numbers an internal comparison across variants rather than a like-for-like comparison with the nnU-Net baseline. Knowledge distillation (`extensions/distill.py`) trains Micro as a student of the Full teacher (soft-target BCE at temperature 2.0, α=0.5). Post-training quantization (`extensions/quantize_model.py`) applies FP16 casting and INT8 dynamic quantization. Compression equity (`extensions/compression_equity.py`) re-evaluates every checkpoint stratified by tumor size.

---

## 4. Results

### 4.1 Baseline Reproduction

3-fold cross-validation (folds 0/1/2), foreground tumor class, from `aggregate_folds.py`:

| Metric | Target (literature) | Fold 0 | Fold 1 | Fold 2 | Mean ± Std |
|---|---|---|---|---|---|
| Dice (tumor slices) | 0.82–0.92 | 0.839 | 0.827 | 0.853 | **0.840 ± 0.011** |
| Dice (all val slices) | — | 0.810 | 0.766 | 0.811 | 0.796 ± 0.021 |
| IoU | 0.70–0.85 | 0.738 | 0.694 | 0.734 | 0.722 ± 0.020 |
| Precision | — | 0.916 | 0.920 | 0.907 | 0.915 ± 0.006 |
| Recall | — | 0.904 | 0.896 | 0.905 | 0.902 ± 0.004 |

**The tumor-slice Dice of 0.840 ± 0.011 lies within the 0.82–0.92 target range — a successful reproduction.** The tight cross-fold standard deviation (±0.011) indicates a stable result despite the reduced 200-epoch budget.

**HD95 / Hausdorff distance** is not reported: nnU-Net's default `summary.json` provides Dice, IoU, and confusion-matrix counts but not boundary distances. Computing HD95 would require re-running evaluation with boundary metrics enabled and is listed in future work.

**Compute:** ~137 s/epoch on a Kaggle Tesla T4; ~7.6 h per fold for 200 epochs. Per-slice inference latency and CPU/GPU memory were not separately benchmarked (future work).

### 4.2 Cross-Institution Fairness

Per-institution Dice on tumor slices, averaged across the 3 folds (`final_3fold_metrics.json`):

| Institution | Tumor cases (pooled) | Dice (3-fold mean) | Δ vs best |
|---|---|---|---|
| TCGA_HT | 195 | 0.868 | — |
| TCGA_FG | 126 | 0.840 | −0.028 |
| TCGA_DU | 310 | 0.839 | −0.029 |
| TCGA_EZ | 5 | 0.823 | −0.045 (n=5, unreliable) |
| TCGA_CS | 77 | 0.777 | **−0.091** |

**Equity gap: 0.091 Dice points** between the best (TCGA_HT) and worst reliable (TCGA_CS) institution. The gap is consistent across folds (cross-fold SD ≤ 0.045 per institution).

### 4.3 Performance by Tumor Size

Dice by ground-truth tumor area, 3-fold mean:

| Tumor size (px) | Cases (pooled) | Dice (3-fold mean) | Cross-fold SD |
|---|---|---|---|
| XS (<200) | 38 | **0.555** | 0.033 |
| S (200–500) | 92 | 0.687 | 0.057 |
| M (500–1k) | 123 | 0.786 | 0.025 |
| L (1k–2k) | 189 | 0.888 | 0.008 |
| XL (>2k) | 271 | 0.925 | 0.003 |

**A 0.370 Dice gap separates the smallest from the largest tumors** — the single largest disparity in the study, and remarkably consistent across folds (SD ≤ 0.057).

### 4.4 Genomic and Demographic Equity

Per-patient Dice by subgroup (`subgroup_metrics.json`):

| Dimension | Best group | Worst group | Gap |
|---|---|---|---|
| Genomic subtype (RNASeqCluster) | RNASeq-1: 0.859 (n=23) | RNASeq-3: 0.782 (n=10) | **0.077** |
| Histologic grade | grade=1: 0.842 (n=46) | grade=2: 0.822 (n=52) | 0.020 |
| Gender | group 2: 0.840 (n=48) | group 1: 0.822 (n=50) | 0.018 |
| Age group | 55–70: 0.841 (n=28) | 40–55: 0.823 (n=30) | 0.017 |

**Key result:** the model is **demographically equitable** — gender, age, and histologic grade each vary by <0.02 Dice — but shows a meaningful **0.077 gap across genomic subtypes**, with RNASeq cluster 3 underserved. (Grade/gender codes follow the source `data.csv`, which ships no published codebook, so they are reported as raw codes.)

**Summary of disparities (largest to smallest):** tumor size (0.370) ≫ institution (0.091) > genomic subtype (0.077) ≫ grade (0.020) ≈ gender (0.018) ≈ age (0.017).

### 4.5 Lightweight U-Net Variants

Three plain U-Net variants trained 20 epochs on the slice-level split
(`unet_*_metrics.json`, `architecture_benchmark.json`). "Val Dice (all)" is the
mean over all validation slices (tumour + empty); "Tumor Dice" is the mean over
tumour-bearing slices only (`compression_equity.json`), the metric comparable to
the nnU-Net baseline:

| Model | Params | Size (MB) | Val Dice (all) | Tumor Dice | CPU ms/slice | GPU ms/slice |
|---|---|---|---|---|---|---|
| Full  | 31.0M | 118.5 | 0.868 | 0.766 | 571 | 20.8 |
| Slim  | 7.76M | 29.7  | 0.866 | 0.752 | 166 | 7.6 |
| Micro | 1.94M | 7.5   | 0.864 | 0.787 | 67  | 4.0 |

**Micro U-Net matches Full's validation Dice (0.864 vs 0.868) with 16× fewer
parameters, a 15.9× smaller footprint, and 8.5× faster CPU inference (67 ms vs
571 ms/slice).** On tumour slices Micro is actually the strongest of the three
(0.787) — at 20 epochs the larger models show no advantage on this 2D task.
All three sit ~5–9 Dice points below the nnU-Net baseline (0.840 tumour-slice),
as expected given the shorter schedule, lighter augmentation, and plain
architecture; the gap would narrow with longer training.

### 4.6 Quantization (Micro U-Net)

Post-training quantization of the Micro model (`quantization_micro_results.json`).
FP16 dice/latency are measured on GPU (FP16 conv is unsupported on CPU):

| Precision | Val Dice (all) | Size (MB) | Latency (ms) |
|---|---|---|---|
| FP32 | 0.864 | 7.46 | 65.6 (CPU) |
| FP16 | 0.866 | **3.75** | 3.0 (GPU) |
| INT8 (dynamic) | 0.864 | 7.46 | 53.6 (CPU) |

**FP16 halves model size (7.46 → 3.75 MB) with no accuracy loss** (Dice +0.002,
within noise). INT8 *dynamic* quantization produces no size reduction — it only
quantizes Linear/LSTM layers, and a U-Net is all-convolutional; genuine INT8 conv
compression requires static (calibrated) quantization (future work). A 3.75 MB
FP16 Micro U-Net is small enough to ship inside a mobile app. (Slim quantization
ran but its outputs were overwritten by the Micro run in this batch; the script
now writes variant-specific filenames for future runs.)

### 4.7 Equity of Compression

Does shrinking the model disproportionately hurt the hardest (small-tumour)
cases? Tumour-slice Dice by size, per model (`compression_equity.json`):

| Tumor size | Full | Slim | Micro | Micro (distill) |
|---|---|---|---|---|
| XS (<200) | 0.314 | 0.221 | **0.435** | 0.251 |
| S (200–500) | 0.492 | 0.471 | 0.515 | 0.378 |
| M (500–1k) | 0.662 | 0.633 | 0.726 | 0.562 |
| L (1k–2k) | 0.822 | 0.815 | 0.835 | 0.731 |
| XL (>2k) | 0.909 | 0.909 | 0.907 | 0.878 |

**Compression does not disproportionately harm small tumours** — the Micro model
is in fact the *best* of all variants on the smallest (XS) lesions (0.435 vs
Full's 0.314). The dominant determinant of accuracy remains tumour size, not
model size, across every variant. This is a reassuring result for low-resource
deployment: the cheap, CPU-runnable model is no worse on hard cases than the
heavy one.

### 4.8 Knowledge Distillation

Distilling the Full teacher into a Micro student (`distill_micro_metrics.json`)
raised the all-slice validation Dice to **0.871** (vs 0.864 for Micro trained
from scratch), but *lowered* tumour-slice Dice to 0.697 (vs 0.787). The
distillation objective (α=0.5 soft targets) pulled the student toward the
teacher's confident background predictions on the majority-empty slices,
improving the aggregate metric while sacrificing tumour delineation. We report
this as a negative/nuanced result: naive distillation is not beneficial here and
would need a tumour-weighted objective to help.

### 4.9 Robustness: Image-Quality Degradation

SSIM/PSNR versus clean FLAIR across degradation conditions (`robustness_results.json`, 100 tumor slices):

| Degradation | SSIM | PSNR (dB) |
|---|---|---|
| Clean | 1.000 | ∞ (128) |
| Downsample 0.5× | 0.955 | 35.3 |
| Mild blur (σ=1.5) | 0.935 | 33.6 |
| Downsample 0.25× | 0.875 | 29.8 |
| Heavy blur (σ=3.0) | 0.837 | 28.3 |
| Contrast −50% | 0.591 | 23.6 |
| Brightness +50 | 0.465 | 14.2 |
| Missing slice | 0.284 | 15.1 |
| Rician noise (σ=20) | 0.236 | 20.8 |
| Gaussian noise (σ=25) | 0.224 | 21.7 |
| Heavy noise (σ=50) | 0.100 | 16.1 |

Noise (Gaussian/Rician) is by far the most destructive to image fidelity (SSIM 0.10–0.24), whereas mild blur and 0.5× downsampling are comparatively benign (SSIM > 0.93). Segmentation Dice under these conditions (with the trained nnU-Net rather than the current Otsu proxy) is future work.

---

## 5. Analysis & Discussion

### 5.1 Reproduction Fidelity

Our tumor-slice Dice (0.840 ± 0.011) sits comfortably inside the 0.82–0.92 literature range, validating both the pipeline and the dataset conversion. The most likely reason we land mid-range rather than at the top is the reduced 200-epoch budget (§3.3); validation pseudo-Dice plateaued early, so we expect the full 1,000-epoch schedule would add at most a few points. The low cross-fold variance is reassuring evidence that the result is not a lucky split.

### 5.2 Tumor Size Is the Dominant Equity Axis

The 0.370 Dice gap by tumor size dwarfs every other disparity. Small tumors (<200 px, Dice 0.555) are precisely the early-stage lesions where segmentation is most clinically valuable — earlier detection means more treatment options. A model that excels on large, late-stage tumors but fails on small ones is misaligned with clinical need. This is the study's central responsible-AI finding: aggregate Dice (0.840) masks a severe weakness on the highest-stakes cases.

### 5.3 Cross-Institution Equity

The 0.091 gap between TCGA_HT (0.868) and TCGA_CS (0.777) tracks representation: TCGA_CS contributes only 358 slices and shows both the lowest Dice and the highest variance. This mirrors the real-world equity gap — hospitals in well-resourced settings contribute more training data, and their patients receive more accurate predictions, while patients at under-resourced institutions, who arguably need AI assistance most, receive the worst performance. Federated learning or representation-aware augmentation could help close this gap.

### 5.4 Demographic Equity vs Clinical Disparity

A notable positive result: the model does **not** show meaningful bias by gender, age, or histologic grade (all <0.02 Dice). The disparities that exist are *clinical/structural* (tumor size, institution, genomic subtype) rather than demographic. The 0.077 genomic-subtype gap (RNASeq-3 worst) warrants follow-up — molecular subtype correlates with tumor morphology, and under-representation of a subtype in training may explain it.

### 5.5 Lightweight Models Make CPU-Only Deployment Viable

The Micro U-Net is the clearest practical win of the study: at 1.94M parameters
and 7.5 MB it runs in 67 ms/slice on CPU — fast enough for interactive use on a
commodity laptop with no GPU — while matching the Full model's validation Dice.
FP16 quantization shrinks it further to 3.75 MB, smartphone-app scale, with no
accuracy loss. For a rural clinic where a GPU workstation is out of budget, this
is the difference between "deployable" and "not". Two honest caveats temper the
result: (1) all lightweight variants trail the nnU-Net baseline by ~5–9 tumour-
slice Dice points at this training budget, so there is an accuracy cost to pay
for portability; and (2) naive knowledge distillation did not help — it improved
the aggregate metric while degrading tumour delineation (§4.8), a reminder that
compression techniques must be validated on the clinically relevant metric, not
just aggregate Dice. Crucially, the equity-of-compression analysis (§4.7) shows
the cheap model is *no worse* on the hardest small-tumour cases than the heavy
one, so portability does not come at the cost of fairness.

### 5.6 Ethical Implications of Deployment Without Radiologist Oversight

In high-income settings, AI segmentation is decision support: a radiologist reviews and overrides as needed. In rural settings with no radiologist, the model's prediction may become the de facto diagnosis, removing the human error-correction layer. Given our findings, responsible deployment requires:
1. **Size-aware confidence** — the model should flag small-lesion cases where Dice is expectedly low.
2. **Input quality control** — a zeroed/corrupted slice yields Dice 0 silently; slice-level QC must reject such inputs.
3. **Subgroup auditing** — periodic monitoring by institution and genomic subtype.
4. **Regulatory clearance** — no model here is cleared for clinical diagnosis.

### 5.7 Limitations

1. **200-epoch budget** for nnU-Net (vs canonical 1,000) and **20 epochs** for the lightweight variants (Kaggle GPU limits); both likely understate achievable Dice.
2. **3 of 5 CV folds** evaluated; the held-out 434-slice test set is not yet scored.
3. **Two Dice conventions** — the lightweight/quantization tables report all-slice validation Dice, while the baseline and equity analyses report tumour-slice Dice; we label each but they are not directly interchangeable, and the lightweight models use a slice-level (not patient-level) split.
4. **2D only / FLAIR only** — no 3D context, no T1/T1Gd.
5. **No HD95 / boundary metrics** — nnU-Net default summary omits them.
6. **Robustness Dice is a proxy** (Otsu), not nnU-Net inference; only SSIM/PSNR are final.
7. **INT8 dynamic quantization is ineffective** for conv U-Nets (no size reduction); static quantization is needed for a genuine INT8 result.
8. **No African validation set** — TCGA-LGG is US/China-sourced; generalization to African scanners and demographics is untested.
9. **Coded clinical labels** — `data.csv` ships no codebook for grade/gender/race.

---

## 6. Future Work

1. **Static (calibrated) INT8 quantization** for genuine 4× conv compression, and **tumour-weighted distillation** to fix the §4.8 regression.
2. **Real degraded-input inference** — replace the Otsu proxy with nnU-Net on degraded slices.
3. **Full 1,000-epoch run** and all 5 folds; score the held-out test set; add HD95; longer lightweight schedules with patient-level splits for a like-for-like comparison.
4. **Volumetric & multi-sequence** extension (3D nnU-Net, T1/T1Gd fusion).
5. **African cohort validation** and **federated learning** for cross-institution equity.
6. **Size-aware calibration** to communicate uncertainty on small lesions.

---

## 7. Conclusion

We reproduced the nnU-Net 2D baseline on TCGA-LGG, achieving **0.840 ± 0.011** tumor-slice Dice across 3-fold cross-validation — within the 0.82–0.92 target range. Our equity analysis shows the model is demographically fair (gender/age/grade gaps <0.02) but exhibits large, consistent disparities by tumor size (0.370), institution (0.091), and genomic subtype (0.077). The tumor-size disparity is the headline finding: aggregate accuracy hides systematic failure on the small, early-stage lesions where segmentation matters most. On the deployment side, a **Micro U-Net (1.94M params, 7.5 MB, 67 ms/slice CPU) matches the Full model's Dice at 16× the efficiency**, FP16 halves its size to 3.75 MB with no accuracy loss, and — importantly — this compression does *not* worsen the hardest small-tumour cases. We argue that equitable deployment in low-resource settings requires size-aware confidence, input quality control, and subgroup auditing — not just a high average Dice — and that cheap, CPU-runnable models can deliver this without sacrificing fairness. All code, metrics, and figures are released openly to support replication.

---

## References

1. Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021). nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. *Nature Methods, 18*(2), 203-211.
2. Buda, M., Saha, A., & Mazurowski, M. A. (2019). Association of genomic subtypes of lower-grade gliomas with shape features automatically extracted by a deep learning algorithm. *Computers in Biology and Medicine, 109*, 218-225.
3. Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional networks for biomedical image segmentation. *MICCAI 2015*.
4. Hinton, G., Vinyals, O., & Dean, J. (2015). Distilling the knowledge in a neural network. *arXiv:1503.02531*.
5. Jacob, B., et al. (2018). Quantization and training of neural networks for efficient integer-arithmetic-only inference. *CVPR 2018*.
6. Obermeyer, Z., Powers, B., Vogeli, C., & Mullainathan, S. (2019). Dissecting racial bias in an algorithm used to manage the health of populations. *Science, 366*(6464), 447-453.
