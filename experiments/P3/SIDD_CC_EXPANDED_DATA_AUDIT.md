# SIDD-CC Expanded RAW/SRGB Acquisition and Audit

**Run:** `PRISM-P1-20261106-002`  
**Acquired:** 2026-11-06  
**Dataset:** `MhdMohammadi/SIDD-CC`  
**Pinned revision:** `5eb2ab6375d7f99f36b59bb5aacb46af18409412`  
**Local copy:** `data/external/sidd_cc_expanded/`  
**Original P1 subset:** `data/external/sidd_cc/` — untouched; inventory SHA-256 remains `21ec28480a6029c02b7679e4242a9a1eedd1ed206b5ea70b18399d5a149222df`.

## Acquisition and license

The dataset card documents separate RAW and SRGB modalities and gives download examples using the Hugging Face CLI. The revision-pinned acquisition fetched 1,163 requested files into a new directory. The published RAW modality is 24,593,736,118 bytes and SRGB is 8,668,424,920 bytes; the resulting SHA-256 file manifest covers 1,163 files totaling 33,262,165,122 bytes. There was 94 GiB free before acquisition and 63 GiB after.

The SIDD-CC card labels its license `sidd-terms-of-use`, links the official SIDD page, and requires citing both SIDD-CC and the original SIDD. The linked official SIDD page currently says the dataset and associated code are under the MIT License. The project records that linked source and has corrected an earlier unsupported “academic/non-commercial” characterization; preserve the license notice and required attribution. Sources: [SIDD-CC dataset card](https://huggingface.co/datasets/MhdMohammadi/SIDD-CC/blob/5eb2ab6375d7f99f36b59bb5aacb46af18409412/README.md), [official SIDD terms page](https://abdokamel.github.io/sidd/index.html).

The acquisition script is [acquire_siddcc_expanded.py](../../scripts/acquire_siddcc_expanded.py); the read-only audit and split generator is [audit_siddcc_expanded.py](../../scripts/audit_siddcc_expanded.py). Alignment is checked by [verify_siddcc_raw_srgb_alignment.py](../../scripts/verify_siddcc_raw_srgb_alignment.py). No dataset files were added to Git, and this workspace is not a Git repository.

## Formats and available data

The upstream [RAW format README](https://huggingface.co/datasets/MhdMohammadi/SIDD-CC/blob/5eb2ab6375d7f99f36b59bb5aacb46af18409412/Raw/README.md) describes single-plane Bayer mosaics stored as float32 MATLAB `data` arrays. Values are already black-subtracted, normalized by original white-minus-black level, and unclipped; negative samples are meaningful. The [SRGB format README](https://huggingface.co/datasets/MhdMohammadi/SIDD-CC/blob/5eb2ab6375d7f99f36b59bb5aacb46af18409412/SRGB/README.md) describes full-resolution, 8-bit RGB PNGs rendered from corresponding RAW arrays through the camera pipeline. Files share instance prefixes and sample indices.

| Split | Groups | Samples / blocks | RAW representation | RGB target |
|---|---:|---:|---|---|
| Official train, full resolution | 115 instances | 2 indexed noisy/GT pairs per instance; 230 pair indices | Bayer float32 MAT | Noisy and GT 8-bit RGB PNG |
| Locked training subset | 98 instances | 196 indexed pairs | same | same |
| Locked validation subset | 17 instances | 34 indexed pairs | same | same |
| Official test | 29 instances | 32 paired blocks per instance; 928 blocks | `(29,32,256,256)` float32 blocks | `(29,32,256,256,3)` uint8 RGB blocks |

The inventory verified all 230 train pair indices. Each has matching RAW noisy/GT MAT shape and dtype, finite values, and a metadata MAT containing CFA, AsShotNeutral, ColorMatrix2, and original BlackLevel/WhiteLevel. All sRGB noisy/GT PNG pairs decode as RGB and match the RAW dimensions. The 230 indices cover 5 cameras and 3 CFA patterns: GRBG 26 pairs (S6), BGGR 106 (GP/N6/G4), RGGB 98 (iPhone). ISO values range from 50 through 10,000. RAW values remain in their existing normalized, unclipped representation; this audit did not re-normalize or clip them.

The pair inventory is [sidd_cc_expanded_pair_inventory.csv](../../data/manifests/sidd_cc_expanded_pair_inventory.csv), the split file is [sidd_cc_expanded_split_manifest.csv](../../data/manifests/sidd_cc_expanded_split_manifest.csv), the alignment result is [sidd_cc_expanded_alignment_results.csv](../../data/manifests/sidd_cc_expanded_alignment_results.csv), and file-level SHA-256 records are [sidd_cc_expanded_file_manifest.csv](../../data/manifests/sidd_cc_expanded_file_manifest.csv).

## Pair mapping and alignment

For train data, mapping is by exact instance ID and image suffix: e.g. `{instance}_NOISY_RAW_{idx}.MAT` → `{instance}_GT_SRGB_{idx}.PNG`. The audit requires two unique sample indices per instance, exactly two noisy and two GT sRGB files, and matching full-resolution dimensions. It found 115 identical RAW/SRGB train instance lists and **230/230** complete cross-modality pair indices.

To check spatial registration empirically, the frozen B0 transform was applied to an even-origin center crop from every noisy Bayer RAW sample. The resulting edge map was compared with the same crop of the corresponding noisy sRGB target over integer offsets −3…+3 pixels. The maximum edge correlation was at offset `(0,0)` for **230/230** samples; the zero-offset edge correlations ranged 0.3311–0.9923 (median 0.8506). This is a registration diagnostic, not an image-quality score. The method tests integer translation on a center crop; it does not establish subpixel registration, eliminate all local geometric distortion, or validate color accuracy.

The official test `test_scenes_info.txt` records match between RAW and SRGB modalities. The `BenchmarkBlocks32` coordinate arrays are byte-data-equivalent after MAT parsing and contain the same 32 coordinates. Test scene order matches. All four test block tensors were loaded and checked for documented shapes/dtypes; both float32 RAW tensors are finite, and the RGB tensors are uint8. Per the [SRGB README](https://huggingface.co/datasets/MhdMohammadi/SIDD-CC/blob/5eb2ab6375d7f99f36b59bb5aacb46af18409412/SRGB/README.md), block `[i][j]` in each modality denotes the same source region.

## Metadata limitations

Training metadata is available per indexed RAW capture: `cfa_str`, AsShotNeutral, ColorMatrix2, and original black/white levels. The official test blocks do **not** ship per-capture metadata. `test_scenes_info.txt` gives scene ID, source image index and dimensions; camera/CFA, ISO, AsShotNeutral and ColorMatrix2 are not present in the downloaded test files. They remain **unknown** in the split manifest. They were not inferred from scene IDs or neighboring train captures.

This gap has two consequences:

1. A learned Bayer-to-sRGB model can be trained and evaluated without metadata conditioning by consuming the four spatial positions of the mosaic as input planes. It must learn the CFA phase/camera response implicitly from the data; this is an explicit research risk.
2. Frozen B0 requires CFA and color metadata and therefore cannot be run on the official test blocks from the acquired files alone. B0/B1 comparison on the official test set is blocked until an authorized, verified test metadata mapping is available. B0/B1 comparisons can use the 17-scene validation split, where capture metadata exists, but that set is for selection and is not a final held-out comparison.

## Split lock

Before model work, the 115 official train IDs and 29 test scene IDs were checked for overlap; none was found. The 115 train instances were split at instance level. A deterministic seed `20261106` and largest-remainder 15% validation allocation stratified by camera produced **98 train / 17 validation** instances. Both indexed pairs and all crops from an instance stay in one split. The official 29-instance test split remains test, with 32 blocks per instance. Validation camera counts are S6=2, GP=4, N6=3, G4=1, IP=7; exact IDs are in the split manifest. This does not create a held-out camera split: all five cameras occur in training and validation. Report scene/camera generalization accordingly.

## Gate result

- Dataset access/license provenance recorded; source revision pinned.
- RAW/SRGB formats, train pair indices, dimensions, metadata availability and integer-offset registration audited.
- Official test block shapes, order and common coordinates verified.
- Instance-level 98/17/29 train/validation/test groups written and hashed in manifests.
- P1 source data and inventory remain unchanged.
- **B1 learned Bayer-RAW-to-sRGB supervised training is now data-feasible.** B1 has not been implemented or trained.
- **Official-test B0/B1 comparison remains blocked** by missing test capture metadata required by B0.


## B1 split-integrity correction (2026-11-06)

The earlier 98/17 assignment in this report is superseded for B1. The dataset README defines the second component of each instance name as scene number; auditing the prior assignment found scene numbers in both train and validation. The frozen B1 assignment groups by documented scene number: train 96 instances (scenes 001,002,004–008), validation 12 (scene 003), internal paired B0/B1 comparison 7 (scenes 009–010), and official test 29 unchanged. The old split is retained in the manifest for provenance, not used for B1. See `B1_SPLIT_INTEGRITY_REPORT.md` and checksum-linked `B1_SPLIT_MANIFEST.csv`.
