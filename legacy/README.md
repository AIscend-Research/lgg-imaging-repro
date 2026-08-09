# legacy/ — archived earlier study (not part of the ReScience submission)

Everything in this folder belongs to an **earlier, separate project** that used the same
Kaggle TCGA-LGG dataset but a different method, framing, and write-up. It is kept for
reference and is **not** run, cited, or required by the reproduction in the repository root.

**Do not mix the two.** In particular, this study frames TCGA-LGG as a proxy for
low-resource African healthcare settings. That framing is incorrect and is explicitly
disavowed by the current work: TCGA-LGG is a US National Cancer Institute / TCIA collection
from five US institutions. See the "Dataset provenance" section of the root
[`README.md`](../README.md).

## Contents

| Path | What it is |
|---|---|
| `paper.md` | The archived manuscript (nnU-Net reproduction + equity / robustness / compression study) |
| `prepare_dataset.py` | TIF → NIfTI conversion for nnU-Net |
| `analyze_dataset.py`, `analyze_fold.py` | Dataset and per-fold summaries |
| `aggregate_folds.py`, `aggregate_results.py` | Fold/metric aggregation |
| `subgroup_analysis.py` | Dice by institution / tumor size / subtype / demographics |
| `visualize_results.py` | Figure generation for the archived paper |
| `extensions/` | Lightweight U-Net variants, distillation, quantization, robustness, fairness, compression equity |
| `extensions_kaggle_notebook.ipynb` | Kaggle notebook driving `extensions/` |
| `results/` | Archived metric JSON/CSV outputs |
| `figures/` | Archived figures (PNG) |
| `nnunet_results/` | Archived nnU-Net validation summaries (folds 0–2, reduced-epoch run) |

## Relationship to the current work

The current work is a ReScience C replication of Buda et al. (2019), driven entirely by
[`src/lgg/`](../src/lgg) and [`config.yaml`](../config.yaml). It re-derives its own splits,
metrics, figures (F1–F9), and reports from scratch and shares **no code** with this folder.

These scripts were written against the repository root and will not run in place without
path fixes. Recover any file's original location from git history:

```bash
git log --follow -- legacy/<file>
```
