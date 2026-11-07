# B1 Architecture Proposal: Metadata-Free Bayer RAW-to-sRGB

> **Frozen implementation contract:** see `B1_CONTRACT.md`. This proposal's compact two-level U-Net is selected, but its earlier split references are superseded by `B1_SPLIT_MANIFEST.csv` and `B1_TRAINING_SPECIFICATION.md`.

**Status:** proposal only; no B1 model code or checkpoint exists. The expanded SIDD-CC audit now supports supervised Bayer-to-sRGB training. B0's official-test comparison remains separately blocked by missing test metadata.

## Task contract

- **Input:** normalized, black-subtracted, unclipped single-plane float32 Bayer mosaic, `H×W`. Keep negative and overrange values. Do not re-subtract BlackLevel or clip.
- **Input representation:** pack every 2×2 cell in fixed spatial order `[top-left, top-right, bottom-left, bottom-right]`, producing `4×H/2×W/2`. Do not assign color labels at inference; test CFA metadata is absent. Training includes all three CFA phases so the network learns the phase/camera mapping from the packed samples.
- **Target/output:** paired `GT_SRGB` as full-resolution RGB, normalized from uint8 to `[0,1]` and represented as display-encoded sRGB. Output shape `3×H×W`; output activation maps explicitly to `[0,1]` because the target is 8-bit sRGB. This is an end-to-end learned ISP (demosaic + denoising + rendering) against the SIDD-CC camera pipeline.
- **Metadata:** no per-pixel conditioning in the first model. This makes the model runnable on official test blocks, whose CFA and camera metadata are not provided. Metadata-conditioned B1 is an ablation only after test metadata is obtained and verified.

## Compact two-level U-Net with pixel shuffle

Use a stride-2 encoder in packed Bayer space and a skip-connected decoder. Bilinear upsampling is followed by 1×1 channel reduction; the decoder then concatenates the corresponding skip feature and applies two 3×3 convolutions. A 3×3 head predicts 12 low-resolution channels; PixelShuffle(2) produces 3 full-resolution sRGB channels.

1. Stem: 3×3 conv `4→24`, ReLU.
2. Encoder level 1: two 3×3 `24→24` convs, ReLU.
3. Downsample: 3×3 stride-2 conv `24→48`, ReLU; level 2: two 3×3 `48→48` convs, ReLU.
4. Downsample: 3×3 stride-2 conv `48→96`, ReLU; bottleneck: two 3×3 `96→96` convs, ReLU.
5. Decoder level 2: bilinear ×2; 1×1 conv `96→48`; concatenate 48-channel skip; 3×3 convs `96→48→48`, ReLU.
6. Decoder level 1: bilinear ×2; 1×1 conv `48→24`; concatenate 24-channel skip; 3×3 convs `48→24→24`, ReLU.
7. Head: 3×3 conv `24→12`; PixelShuffle(2); sigmoid output.

With convolution biases, no BatchNorm, and no trainable upsampling parameters, the formula-based count is **357,276 parameters**. Float32 weights occupy about **1.43 MB** decimal before serialization overhead. These are design estimates, not measured model results.

## Objective and data handling

Use RGB MSE on the paired full-resolution `GT_SRGB` target as the primary loss, matching the declared sRGB PSNR metric. Do not normalize each image independently. Report per-channel error and scene-level aggregation. A Charbonnier/L1 objective may be an ablation; no perceptual or color-chart loss is justified as the primary objective. Optional RAW-denoising pretraining is a separate experiment and must not use test data.

Training crops are 256×256 sensor pixels, corresponding to 128×128 packed input. Apply geometric transforms to the full mosaic and target identically, then repack; this preserves Bayer phase changes instead of silently treating CFA metadata as unchanged. Extract patches only after the instance-level split is fixed.

## Memory and inference estimate

At 128×128 packed resolution, hidden 24/48/96-channel feature maps are small (largest individual map about 1.5 MiB float32). With skips and backward buffers, plan roughly 0.5–2 GiB for batch 8 plus framework/workspace overhead; this is only a planning estimate and must be measured on the selected device. Avoid full-resolution training batches. Full-frame inference should use overlapping sensor-domain tiles with aligned even origins; merge RGB tile outputs with a fixed overlap/crop rule and measure seams against single-tile inference on a manageable crop.

Start with float32. CPU and one named accelerator backend may be measured separately. Use mixed precision only after numerical comparison against float32. Parameter count alone does not establish full-frame speed or low memory.

## Why this design and what it risks

Fixed-position packing halves each spatial axis and exposes the Bayer periodic structure without an external CFA map. The model has under 0.4M parameters and produces the requested full-resolution RGB output. Because it receives no CFA or camera metadata, it must infer phase and rendering differences from image samples. This avoids inventing test metadata but may fail on ambiguous/neutral regions or under camera/lighting shift. The model could exploit scene/camera correlations in this small five-camera dataset; report per-camera and per-scene outcomes and do not claim unseen-camera generalization.

The output learns the SIDD-CC camera rendering, not a device-independent linear RGB reconstruction. It cannot be judged for absolute color accuracy without measured chart references. The model's full sRGB output is a standalone B1; a selected-tile refiner alone does not satisfy this role.
