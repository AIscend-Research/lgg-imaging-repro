# Optional nnU-Net comparison module (Section 8)

This is a **comparison baseline**, not the reproduction. The reproduction is the plain
4-level U-Net in [`../src/lgg`](../src/lgg). Run this only if GPU quota allows.

To keep the comparison apples-to-apples and leakage-free:

1. **One case = one patient.** Convert each patient's slices into a single stacked case so
   nnU-Net's case-level splitting operates on patients, not loose slices.
2. **Override nnU-Net's automatic splits.** Build `splits_final.json` directly from
   [`../splits/splits.json`](../splits/splits.json) so nnU-Net uses the exact same
   patient-level folds and cannot leak slices across folds. Copy it into
   `nnUNet_preprocessed/Dataset001_TCGALGG/`.
3. **Reduced-epoch trainer.** Use `nnUNetTrainer_250epochs` (or `_100epochs` for a quick
   sanity pass). The default 1000-epoch schedule does **not** fit Kaggle's session/quota
   limits (~15–30 h *per fold* on a P100). Record the reduced budget as a deviation.
4. **Same per-patient metrics.** Evaluate with the R2 per-patient volume-aggregated Dice
   (reuse `lgg.metrics`) so the number is directly comparable to the U-Net reproduction.

An earlier reduced-epoch nnU-Net run is archived under
[`../legacy/nnunet_results/`](../legacy/nnunet_results) for reference only — it predates
these steps and is **not** leakage-audited against `../splits/splits.json`. Regenerate it
through the steps above if you want a comparable number.

**Warning:** never run default (1000-epoch) nnU-Net at full length on Kaggle — five folds is
~100–150 h and blows the weekly quota.
