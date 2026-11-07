# B1 Learned ISP Feasibility Report

**Updated:** 2026-11-06, after expanded-data audit `PRISM-P1-20261106-002`  
**Decision:** A supervised standalone **Bayer RAW-to-sRGB B1 is now supported** by the acquired SIDD-CC train and official test data. B1 has not been implemented or trained. A final official-test B0/B1 comparison is not yet possible because B0 requires test CFA/color metadata that the released test blocks omit.

## Evidence and acquisition

The dataset was acquired from `MhdMohammadi/SIDD-CC` at pinned revision `5eb2ab6375d7f99f36b59bb5aacb46af18409412`, into `data/external/sidd_cc_expanded/`. The existing P1 subset was left untouched. The acquired copy contains 115 full-resolution train instances with two indexed RAW/RGB pairs each, plus 29 official test instances with 32 paired RAW/sRGB blocks each. Published modality sizes were 24,593,736,118 bytes RAW and 8,668,424,920 bytes SRGB. The local 1,163-file SHA-256 manifest and per-sample inventory are referenced in [SIDD_CC_EXPANDED_DATA_AUDIT.md](SIDD_CC_EXPANDED_DATA_AUDIT.md).

The SIDD-CC card identifies Bayer float32 RAW and 8-bit camera-pipeline sRGB modalities, and requires citations to SIDD-CC and SIDD. The SRGB format README specifies matching instance/sample indices and says the train PNGs are rendered from corresponding RAW arrays. The official SIDD page linked by the card currently states MIT License; project documentation records this source clarification. See [SIDD-CC card](https://huggingface.co/datasets/MhdMohammadi/SIDD-CC/blob/5eb2ab6375d7f99f36b59bb5aacb46af18409412/README.md), [RAW README](https://huggingface.co/datasets/MhdMohammadi/SIDD-CC/blob/5eb2ab6375d7f99f36b59bb5aacb46af18409412/Raw/README.md), [SRGB README](https://huggingface.co/datasets/MhdMohammadi/SIDD-CC/blob/5eb2ab6375d7f99f36b59bb5aacb46af18409412/SRGB/README.md), and [SIDD license/source page](https://abdokamel.github.io/sidd/index.html).

## Pair and alignment verification

All 115 RAW and SRGB train instance lists match. The audit matched the exact instance prefix and index for **230/230** noisy RAW → clean sRGB examples, verified the paired noisy/GT files exist, and checked all RAW MAT arrays are finite float32 Bayer mosaics with matching dimensions and corresponding RGB PNGs are 8-bit RGB at the same dimensions.

For every train pair, the frozen B0 transform was run on an even-origin center RAW crop. The generated edge map and same-index noisy sRGB target edge map were compared over ±3 integer pixels. Best correlation was at zero offset for **230/230** pairs; correlation values ranged 0.3311–0.9923 (median 0.8506). This tests central-crop integer translation, not subpixel accuracy, local warp or colorimetric fidelity. All 29 official test scene-order records and all 32 RAW/SRGB block coordinates match; block arrays have the documented dimensions and dtypes.

## Tasks supported

| B1 task | Input → target | Feasibility now |
|---|---|---|
| RAW denoising | Noisy Bayer float32 → GT Bayer float32 | Supported; use as a separate sensor-domain baseline if desired. |
| RAW-to-RGB reconstruction | Noisy Bayer → linear RGB | No linear RGB ground truth supplied; train a model to this intermediate target only if a trusted rendering/linearization contract is independently built. |
| Full supervised RAW-to-sRGB ISP | Noisy Bayer → clean camera-pipeline sRGB | **Supported.** Use `NOISY_RAW_{idx}` input and paired `GT_SRGB_{idx}` output. The target includes denoising and the camera rendering pipeline. |

Do not train noisy RAW → noisy sRGB as the primary reconstruction target if the intended baseline should recover clean output; that pairing is useful as a separate rendering-only ablation. The sRGB targets represent the SIDD-CC camera pipeline. They are not chart-calibrated colorimetric truth and do not establish fidelity to a universal reference rendering.

## Metadata and scope limitations

For all training captures, each indexed metadata MAT includes `cfa_str`, AsShotNeutral, ColorMatrix2, and original BlackLevel/WhiteLevel. Stored RAW is already normalized and black-subtracted, and negative values must remain. Five cameras and all three verified CFAs appear in train: S6/GRBG, GP/N6/G4/BGGR, and iPhone/RGGB.

Test blocks omit per-capture CFA, camera, ISO, AsShotNeutral, ColorMatrix2 and black/white metadata. The split manifest retains these as unknown. Do not fill them from the train set or infer them from scene IDs. A full learned ISP that consumes only the Bayer array can still run on the official test blocks; therefore the proposed B1 input is metadata-free and represents each 2×2 Bayer cell in fixed spatial order. This asks the network to infer CFA phase and camera rendering from the paired training examples, a material limitation to test explicitly. A CFA-/metadata-conditioned variant cannot be evaluated on the official test until an authorized, verified mapping is obtained.

## Locked splits and adequacy

The published train group of 115 instances was deterministically divided at instance level into **98 train / 17 validation**, stratified by camera with seed `20261106`; both sample indices and every crop stay in the same instance group. The official **29-instance / 928-block test** is untouched. Exact IDs and source image records are in [sidd_cc_expanded_split_manifest.csv](../../data/manifests/sidd_cc_expanded_split_manifest.csv). Camera counts in validation: S6=2, GP=4, N6=3, G4=1, IP=7. Every camera is also represented in training, so this is not a held-out-camera split. It supports a defensible within-dataset held-out-instance result, not broad camera generalization.

The official test set has independent capture groups and paired targets, so it is suitable for final within-dataset evaluation. Its small size, five-camera coverage, class/content bias and missing capture metadata limit the breadth of conclusions. No random patch split is used.

## Metrics: possible and blocked

- **RAW-domain RMSE/PSNR:** possible for the noisy RAW baseline and a separately trained RAW denoiser, using original normalized float32 data, fixed nominal `data_range=1.0`, and no clipping. They do not evaluate a model whose output is RGB.
- **RGB sRGB PSNR/SSIM:** possible on the official paired RGB target, fixed range 1.0 after uint8-to-float conversion, with exact same crop and no per-image normalization. Report each test scene and aggregate across 29 instances, not across patches as if independent.
- **LPIPS:** possible on rendered RGB sRGB predictions/targets if implementation, backbone and normalization are fixed. It measures perceptual similarity to this dataset rendering, not physical truth.
- **Delta E 2000:** computable as target-relative sRGB-to-Lab image difference with a declared D65 conversion, but not a colorimetric-accuracy claim because there is no color chart/reference measurement or verified embedded camera profile.
- **Runtime/memory/parameters/checkpoint size:** not available until B1 exists; use the same-device protocol in [B1_EXPERIMENTAL_PROTOCOL.md](B1_EXPERIMENTAL_PROTOCOL.md).
- **B0 vs B1 on the official test:** blocked until B0's required test CFA/AsShotNeutral/ColorMatrix2 metadata are verified. B0 can be compared on validation captures with metadata, clearly labeled as validation only.

## Decision and next implementation gate

Full supervised Bayer-RAW-to-sRGB B1 is now data-feasible. The recommended first B1 is a metadata-free, standalone, compact Bayer-to-sRGB model, with a separate metadata-conditioned ablation only after test metadata is available. Before training, freeze the output/input policy, verify deterministic preprocessing, and ensure the split manifest and source hashes remain unchanged. Implement B1 as a full-frame reconstruction model; a selected-tile model alone is not B1. Do not train until the architecture and protocol are accepted in the next engineering milestone.

No quality scores, B1 runtime, parameter count of an implemented model, or generalization claim is reported here.


## B1 split-integrity correction (2026-11-06)

The earlier 98/17 assignment in this report is superseded for B1. The dataset README defines the second component of each instance name as scene number; auditing the prior assignment found scene numbers in both train and validation. The frozen B1 assignment groups by documented scene number: train 96 instances (scenes 001,002,004–008), validation 12 (scene 003), internal paired B0/B1 comparison 7 (scenes 009–010), and official test 29 unchanged. The old split is retained in the manifest for provenance, not used for B1. See `B1_SPLIT_INTEGRITY_REPORT.md` and checksum-linked `B1_SPLIT_MANIFEST.csv`.
