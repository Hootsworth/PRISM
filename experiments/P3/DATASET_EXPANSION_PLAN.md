# Dataset Expansion Plan for B1

**Status:** The planned SIDD-CC expansion is complete. No model was trained. See [SIDD_CC_EXPANDED_DATA_AUDIT.md](SIDD_CC_EXPANDED_DATA_AUDIT.md) for the full audit.

## Completed: SIDD-CC RAW + SRGB

Acquired `MhdMohammadi/SIDD-CC` revision `5eb2ab6375d7f99f36b59bb5aacb46af18409412` into `data/external/sidd_cc_expanded/`, keeping the original P1 five-scene directory untouched. The copy contains 115 train instances with two indexed pairs each, plus 29 official test instances and 32 paired blocks per instance. The published modalities total 24,593,736,118 bytes RAW and 8,668,424,920 bytes SRGB. All 230 train pair indices, image sizes, RAW formats, required training metadata, and common test block coordinates/order were verified. An integer center-crop edge registration check found zero best offset for all 230 train pairs within a ±3-pixel search.

The split is locked by instance: 98 train / 17 validation from the official train set, and all 29 official test instances untouched. Seed `20261106`, camera-stratified validation allocation. Pair and split CSVs plus 1,163 file SHA-256 records are in `data/manifests/`. The test metadata gap remains: official test blocks do not include CFA/camera/ISO/AsShotNeutral/ColorMatrix2. This does not block a metadata-free B1, but blocks running frozen metadata-dependent B0 on the official test set.

The SIDD-CC card labels its license `sidd-terms-of-use`, links the original SIDD page, and requires both SIDD-CC and SIDD citations. The linked SIDD page currently states MIT License. Preserve notice and attribution; the project has corrected the earlier unsupported non-commercial characterization. Sources: [SIDD-CC card](https://huggingface.co/datasets/MhdMohammadi/SIDD-CC/blob/5eb2ab6375d7f99f36b59bb5aacb46af18409412/README.md), [RAW README](https://huggingface.co/datasets/MhdMohammadi/SIDD-CC/blob/5eb2ab6375d7f99f36b59bb5aacb46af18409412/Raw/README.md), [SRGB README](https://huggingface.co/datasets/MhdMohammadi/SIDD-CC/blob/5eb2ab6375d7f99f36b59bb5aacb46af18409412/SRGB/README.md), [official SIDD page](https://abdokamel.github.io/sidd/index.html).

## Remaining data work

1. Resolve the official-test metadata gap before claiming a same-split B0/B1 comparison. Locate an authorized source that maps the 29 test scenes to CFA and capture ColorMatrix2/AsShotNeutral; verify the mapping, do not infer values from train captures. If this source cannot be found, keep B0/B1 comparison on metadata-complete validation only and report the limitation.
2. Consider a camera-held-out experiment only after increasing camera diversity; current 5-camera data cannot support claims beyond those devices and conditions.
3. Do not add another dataset by default. If future research requires broader camera generalization, evaluate a separate licensed dataset before acquisition and preserve an independent source-level test set.

## Other documented candidates (not acquired)

- **Original SIDD Medium:** official documentation describes 320 paired examples from 160 instances, noisy/GT three-channel Raw-RGB MATs, noisy/GT sRGB PNGs, DNG archives corresponding to noisy Raw-RGB samples, and metadata. Raw-RGB MAT is not a Bayer mosaic. The official page states MIT License. It could add training samples if DNG-to-sRGB mapping is verified, but is unnecessary for the present supervised SIDD-CC B1 and has a different split/target lineage. [Official SIDD data page](https://abdokamel.github.io/sidd/index.html).
- **Modular Neural ISP S24:** repository documentation describes 16-bit PNG demosaiced three-channel RAW representations, metadata, denoised RAW and several sRGB targets. This is not single-plane Bayer input, and the linked documentation notes possible raw/sRGB pixel shift. Its dataset access/license and direct PRISM compatibility would need separate verification before use. [MNI dataset documentation](https://github.com/SamsungLabs/modular_neural_isp/blob/main/datasets/README.md).
- **Zurich / MIT-Adobe FiveK:** MNI docs require preprocessing; their denoised RAW targets are pseudo-targets, and FiveK expert edits are not camera-native sRGB truth. Verify rights, DNG decoding, metadata, alignment and target semantics before any use.

No further download is needed to begin a metadata-free SIDD-CC B1 implementation. The next gate is to freeze the input/output model contract and either acquire verified test metadata for B0 comparison or explicitly scope that comparison to validation.


## B1 split-integrity correction (2026-11-06)

The earlier 98/17 assignment in this report is superseded for B1. The dataset README defines the second component of each instance name as scene number; auditing the prior assignment found scene numbers in both train and validation. The frozen B1 assignment groups by documented scene number: train 96 instances (scenes 001,002,004–008), validation 12 (scene 003), internal paired B0/B1 comparison 7 (scenes 009–010), and official test 29 unchanged. The old split is retained in the manifest for provenance, not used for B1. See `B1_SPLIT_INTEGRITY_REPORT.md` and checksum-linked `B1_SPLIT_MANIFEST.csv`.
