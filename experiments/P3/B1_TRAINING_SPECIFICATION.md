# B1 Training Specification

**Status:** preregistered configuration; no training has run. This specification implements `B1_CONTRACT.md` and its scene-grouped manifest.

## Data assignment and sampling

Use `B1_SPLIT_MANIFEST.csv` field `split_frozen_scene_grouped` only:

- Train: 96 SIDD-CC instances, 192 indexed RAW/target pairs, scene numbers 001, 002, 004–008.
- Validation: 12 instances / 24 pairs, scene 003.
- Internal paired comparison test: 7 instances / 14 pairs, scenes 009 and 010; never use for optimization, early stopping, or hyperparameter/model selection.
- Official test: 29 scene groups × 32 blocks = 928 blocks; held until the final declared evaluation.

Each group, both pair indices, and all crops stay in one split. Sample crops only after split assignment. The unit of statistical aggregation is scene group, not crop count. The official test's scene metadata remains unknown.

### Split correction record

The prior 98/17 split randomized 115 instances within camera strata. The SIDD-CC README explicitly defines the second identifier field as scene number. Audit found each of the ten scene numbers on both sides of the old train/validation split; e.g. scene 001 had 11 train and 7 validation instances. That is scene leakage for this task. The old assignments are retained as `split_existing_instance_random`; the frozen assignment groups complete scene numbers. Scene 003 is validation; 009 and 010 form a small same-scene-group internal B0/B1 comparison. This allocation preserves a 96/12/7 partition and leaves official test intact. It is deterministic and documented, not presented as a camera-held-out split.

## Preprocessing and augmentation

1. Read only the verified `data` float32 RAW variable and paired `GT_SRGB` RGB PNG using the instance/index mapping.
2. Validate finite RAW, 2D/even dimensions, RGB PNG, aligned dimensions and manifest membership. Preserve negative/overrange RAW.
3. Choose 256×256 sensor-pixel crops with top-left coordinates aligned to even CFA lattice coordinates. Apply the exact same crop to RAW and target.
4. Apply a uniformly sampled member of the dihedral transforms (0/90/180/270 degree rotations, horizontal/vertical reflections), identically to mosaic and RGB target, then pack RAW. Repacking after transforms is mandatory; never retain an assumed CFA label.
5. Convert target uint8 to float32 `/255`; do not apply per-frame normalization, color correction, gamma conversion, or target clipping beyond its native uint8 range.
6. No mixup, mosaic cut-and-paste, arbitrary scaling, color jitter, or independently sampled RAW/RGB geometry.

Augmentation policy and seed must be fixed in the run config and recorded. Validation and test receive no stochastic augmentation.

## Model and objective

Use frozen 357,276-parameter compact packed Bayer U-Net from `B1_CONTRACT.md`. Input patch tensor is `N×4×128×128`; output is `N×3×256×256` in `[0,1]` sRGB.

**Primary loss:** mean squared error over all RGB pixels and channels, weight 1.0. It is appropriate to bounded sRGB regression and aligns with the PSNR family, though not with perceptual/colorimetric accuracy. No additional losses in the primary run. Charbonnier/L1, SSIM, LPIPS, color-space losses, saturation weighting or RAW auxiliary loss are separate experiments requiring a new run ID and no test tuning.

## Optimizer and schedule

- Precision: float32; autocast/mixed precision disabled for the registered run.
- Optimizer: AdamW, learning rate `3e-4`, betas `(0.9, 0.999)`, epsilon `1e-8`, weight decay `1e-4`.
- Batch size: 8 crops.
- Epoch cap: 100.
- Scheduler: cosine annealing from `3e-4` to `1e-6` over 100 epochs; no restart or warmup.
- Early stopping: stop after 15 consecutive completed epochs with no improvement in validation scene-macro MSE; `min_delta=0` in float64 comparison of recorded scalar.
- Checkpoint: select the lowest validation scene-macro MSE; ties select earliest epoch. Freeze once before internal/official test evaluation.
- No gradient accumulation unless required by hardware; if used, preserve effective batch size 8 and record it. An OOM-driven change to crop size, batch size, or architecture is a protocol deviation with a new config hash/run ID.

## Reproducibility

Set Python, NumPy, framework, loader and augmentation seeds to `20261106`; deterministic kernels/algorithms enabled where supported. Record any nondeterministic operator warning, framework/CUDA/Metal version, device model, OS, thread counts, package lock, source and split hashes, config hash, checkpoint SHA-256, and exact command. Do not claim bitwise determinism if backend kernels cannot provide it. Save epoch-level train loss and validation scene-macro loss/PSNR. Keep test data unavailable to training code.

## Resource plan

Start on a named device in FP32. Planning estimate from architecture proposal is 0.5–2 GiB for batch 8 plus runtime/workspace; this is not a measurement. Measure training peak and inference peak. Tiling is only for inference memory, not a way to multiply correlated samples. Report model-only and full pipeline costs separately. No GPU requirement and no real-time/mobile claim.
