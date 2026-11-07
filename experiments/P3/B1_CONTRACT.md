# B1 Frozen Task and Model Contract

**Contract ID:** `PRISM-P3-B1-CONTRACT-20261106-001`  
**Status:** Frozen for implementation. No B1 code/checkpoint exists.  
**Decision:** Approved with documented limitations; see `B1_CONTRACT_FREEZE_REPORT.md`.

## Evidence labels

- **Verified:** the pinned SIDD-CC files, READMEs, P1/P2 reports, frozen B0 artifacts, and manifests support the stated fact.
- **Decision:** an experimental choice fixed here before implementation.
- **Limitation:** a known scope or evidence constraint that must remain in result reports.

## Task choice

**Primary task:** full-image Bayer RAW → dataset camera-pipeline sRGB reconstruction, supervised against `GT_SRGB`. This is the only primary task among the candidates supported by a directly paired clean target. It jointly learns demosaicing, denoising, color rendering, tone mapping and display encoding as represented by the SIDD-CC target pipeline. It does not produce an independently defined linear RGB intermediate.

- Bayer → linear RGB has no supplied linear RGB target.
- Bayer denoising → separate fixed renderer changes the primary task and lacks a validated matching renderer.
- `NOISY_RAW → NOISY_SRGB` is a rendering-only ablation, not the clean-output objective.

**Verified target semantics:** training `GT_SRGB` is an 8-bit RGB PNG paired by instance ID and index with the RAW. Upstream SRGB documentation describes the images as rendered from corresponding RAW through the camera pipeline. The dataset does not sufficiently document every internal ISP operation, its profile parameters, or a universal colorimetric reference. Treat the target as the SIDD-CC camera-pipeline rendering / estimated clean target, not chart-calibrated truth. It is lossy 8-bit display-referred sRGB. The noisy sRGB companion is not the primary target.

## Input and output contract

| Property | Frozen definition |
|---|---|
| Source input | `NOISY_RAW_{idx}.MAT`, variable `data`, verified 2-D float32 Bayer mosaic. |
| Original input dimensions | `H×W`; full-resolution training and 256×256 official test blocks; even dimensions verified. |
| Model input | NCHW float32 tensor `N×4×(H/2)×(W/2)` after 2×2 spatial packing. Channel order is `[top-left, top-right, bottom-left, bottom-right]`; no CFA-color reordering. |
| Input values | Preserve stored values directly. Dataset values are already black-level subtracted and normalized by upstream convention; do not black-subtract again, divide by levels again, per-image normalize, clip negatives, or clip overrange values. Reject NaN/Inf and malformed dimensions. |
| White/saturation | Original RAW has no second white-level transform. Preserve values at/above nominal white; do not infer or apply a saturation mask in primary B1. Any saturation-aware loss is a separately preregistered ablation. |
| CFA | No metadata input and no one-hot CFA flag. Fixed spatial packing exposes phase positions; training includes BGGR, GRBG and RGGB captures. The model must learn phase/camera response from examples. This is a known identifiability and generalization risk, especially on neutral/ambiguous content. |
| Metadata | None. Specifically no camera, ISO, AsShotNeutral, ColorMatrix2, exposure, or values inferred from filenames. |
| Output | NCHW float32 `N×3×H×W`, RGB channel order, full sensor-pixel resolution; sigmoid-bounded `[0,1]` display-encoded sRGB. |
| Target tensor | `GT_SRGB` PNG converted RGB uint8 → float32 by `/255`; fixed `[0,1]` range, no image-specific normalization. |
| Color space | Output and target are display-referred standard sRGB as supplied/documented. B1 is not claimed to recover camera-linear RGB, XYZ, or physical reflectance. No extra gamma/OETF is applied after model output. |

For packed spatial dimensions not divisible by four, pad only the packed tensor at right/bottom to the network's stride requirement using reflection where valid (replicate for a dimension too short to reflect), then crop the RGB output back to the original size. Pair crops must use identical sensor coordinates, with RAW crop origins even in both dimensions. Reject odd sensor dimensions rather than shifting CFA phase implicitly.

## Frozen architecture

Compact two-level skip U-Net in packed space with PixelShuffle(2) RGB head, exactly as enumerated in `B1_ARCHITECTURE_PROPOSAL.md`: widths 24/48/96, ReLU, bilinear upsampling + learned 1×1 reduction, concatenative skips, no normalization layers, 12-channel low-resolution head, PixelShuffle(2), sigmoid. Parameter count is **357,276 by formula**, not an observed model result. No residual input addition because RAW and RGB do not share channel/domain semantics. FP32 is initial execution precision.

Full-frame inference is primary. If tiling is required by memory, tile on even sensor origins, use a fixed overlap/crop blending rule, and demonstrate seam error against full-frame inference on a frame that fits. Tiling does not redefine B1.

## Camera/CFA scope

SIDD-CC training covers five camera families and three Bayer patterns, with ISO 50–10,000. The frozen model receives neither camera nor CFA metadata and therefore may learn dataset-specific correlations. The official test lacks per-capture metadata but the four-plane input is runnable. Results support only within-dataset evidence over supplied SIDD-CC scenes; no unseen-camera claim is authorized. Metadata-conditioned and CFA-conditioned variants are out of scope unless a verified test-time source is separately obtained and a new contract is registered.

## Source of truth

Data revision: `5eb2ab6375d7f99f36b59bb5aacb46af18409412`. Pair evidence: `data/manifests/sidd_cc_expanded_pair_inventory.csv`. Split truth: `B1_SPLIT_MANIFEST.csv`; SHA-256 `9e93f75071fb3fc4bf5e2c0e0ab127cc90104caafe231dda83f2a2d388ec441c` at freeze. The earlier 98/17 instance-randomized split is retained in a separate column for traceability and is **not** the frozen experimental assignment because it splits documented scene numbers across train/validation.
