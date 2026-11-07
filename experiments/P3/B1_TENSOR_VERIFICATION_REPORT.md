# B1.1 Tensor Verification Report

## Frozen tensor contract

For a Bayer input `(H,W)`, the loader returns input float32 `[4,H/2,W/2]` in `[TL,TR,BL,BR]` order and target float32 `[3,H,W]` in RGB order. Batches add the leading batch dimension. RAW values are copied only as needed for contiguous layout; no normalization, clipping, gamma or color transform is applied. PNG uint8 values are divided by 255 exactly once.

## Four-split representative samples

| Split | Stable sample ID | Input tensor | Target tensor | Source RAW range | Result |
|---|---|---:|---:|---:|---|
| Train (registered 256 crop) | `0001_001_S6_00100_00060_3200_L:034` | `[4,128,128]` | `[3,256,256]` | 0.00782014 to 0.143695 | exact unpack PASS |
| Validation (full frame) | `0054_003_N6_00100_00160_5500_N:035` | `[4,1560,2104]` | `[3,3120,4208]` | −0.00834202 to 0.673618 | exact unpack PASS |
| Internal comparison (full frame) | `0192_009_IP_00100_00200_3200_N:076` | `[4,1512,2016]` | `[3,3024,4032]` | −0.0137370 to 1.0 | exact unpack PASS |
| Official test (block 0) | `0020:block00` | `[4,128,128]` | `[3,256,256]` | 0.0169403 to 1.0 | exact unpack PASS |

All outputs are contiguous float32. Representative target tensors were finite in `[0,1]`; the original uint8 target channel values round-trip through the `/255` representation exactly at their source quantization. Full per-sample statistics (shape, dtype, min, max, mean, standard deviation, finite, NaN and infinity counts) are recorded in `B1_TENSOR_VERIFICATION_RESULTS.json`.

## Complete-split verification

`PYTHONPATH=src python3 scripts/verify_b1_data_pipeline.py --all-samples` completed successfully:

- 192/192 training pairs, loaded at full source dimensions for the complete pass.
- 24/24 validation pairs.
- 14/14 internal comparison pairs.
- 928/928 official test blocks.
- Every sample passed input/target shape checks, finite checks, target range checks and exact Bayer pack/unpack reconstruction.
- The paired files and source hashes were verified; no RAW/target file was resized or silently cropped during the full-resolution pass.

Aggregate RAW source observations, without clipping: train minimum −0.0667362, maximum 1.0, 100,038,619 negative samples; validation minimum −0.0667362, maximum 1.0, 18,033,820 negatives; internal comparison minimum −0.0508210, maximum 1.0, 11,748,265 negatives; official test minimum −0.0667362, maximum 1.0, 3,898,638 negatives. No value greater than 1 occurred in this acquired revision. Exact `value_count` values are in the JSON result artifact.

Synthetic tests additionally verified negative and >1 float32 preservation, all three source CFA configurations without channel reordering, every D4 transformation, transformed CFA phase, odd-size rejection, malformed MAT rejection and target channel/scaling behavior.

## Visual artifacts

Four inspection panels and their data statistics are linked in `B1_VISUAL_INSPECTION.md`. For display only, each grayscale RAW panel is independently min/max mapped to 8-bit; the stored dataset and training tensors are not changed. Visual similarity is not evidence of colorimetric accuracy.

**Decision:** Tensor and data correspondence checks PASS for this pipeline milestone. This does not constitute model or ISP-quality evaluation.
