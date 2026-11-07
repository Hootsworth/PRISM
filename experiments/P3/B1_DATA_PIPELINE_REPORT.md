# B1.1 Dataset Pipeline Report

**Run:** `PRISM-P3-20261106-004`  
**Status:** Data pipeline implemented and verified; no B1 network or training code.  
**Dataset:** SIDD-CC revision `5eb2ab6375d7f99f36b59bb5aacb46af18409412`.

## Implementation

- `src/prism/data/b1_dataset.py` provides strict MAT/PNG loading, manifest/hash checks, paired train/evaluation samples, official block access, split-aware datasets, tensor conversion, and PyTorch DataLoader helpers.
- `src/prism/data/b1_packing.py` implements exact fixed-position `[TL,TR,BL,BR]` pack/unpack and synchronized D4 transforms for Bayer mosaics and RGB targets.
- `scripts/verify_b1_data_pipeline.py` performs split smoke checks, all-pair/block verification, exact reconstruction, statistics, and figures.
- `scripts/profile_b1_data_pipeline.py` profiles file decoding, packing, tensor conversion, full-resolution samples and a training batch.
- `tests/test_b1_data_pipeline.py` covers loader validation, transforms, split separation and checksum failure behavior.

No new runtime dependency was added. PyTorch remains in the existing `learned` optional dependency group; SciPy and Pillow are already declared by the project.

## Data behavior and integrity

The loader accepts only MAT variable `data` as a finite 2D float32 RAW array with even dimensions. It makes no black-level subtraction, white-level division, clipping, gamma operation, denoising, demosaicing or color conversion. The model input is a contiguous float32 tensor with shape `[4,H/2,W/2]`; spatial slots are never CFA-reordered. `unpack_bayer(pack_bayer(raw))` is bit-exact.

The target loader requires a PNG in RGB mode with uint8 samples, checks the exact paired dimensions, keeps RGB order, and converts to contiguous CHW float32 by `/255`. It does not resize or apply another transfer function.

The loader requires the frozen 144-row B1 split SHA-256 `9e93f75071fb3fc4bf5e2c0e0ab127cc90104caafe231dda83f2a2d388ec441c`, source file manifest SHA-256 `42c5c6db7519946e3a73c7060c8f1e84e1f6ac181d47f970ee7052a5367500a8`, pair inventory SHA-256 `8090fa23d8c62f827a5d8b79a08b81c12066d20afef2cb300bc772aa0d8baa01`, and audited group split hash. It verifies each pair's five referenced file hashes on first use, the group checksum, dimensions, formats, all split labels, and duplicate/cross-split references. Official test rows map by audited scene order and block index to the shared RAW/RGB block tensors; test CFA/color metadata is never inferred.

Every source-manifest entry was SHA-256 checked: 1,163/1,163 files (33,262,165,122 bytes), command `PYTHONPATH=src python3 scripts/verify_b1_source_hashes.py`, 13.121 s. Every train/validation/internal pair loaded: 230/230. Every official test block loaded: 928/928. The artifact audit found no missing or unreferenced MAT/PNG payload files and no dataset file absent from the source file manifest. Hugging Face `.cache` lock/metadata files are treated as download-client state, not dataset payload. The dataset directory was not modified by the pipeline.

## Crops, augmentation and loading

Full-resolution loading is supported for validation, internal comparison, official test and audits. The frozen training view uses paired 256×256 sensor-pixel crops with even top-left offsets, then a seeded D4 transform applied identically to RAW and RGB; RAW is packed only afterward. CFA phase is transformed by spatial geometry and is deliberately not sent as metadata. Unsupported transform names, odd dimensions, non-finite data, shape mismatches and path traversal fail with a sample/file-specific error.

`make_b1_dataloader` exposes batch size, worker count, pinned memory, persistent workers, shuffle and seed. Defaults are batch 8 for train (use `make_training_dataset`, 256 crops) and batch 1 for evaluation, 0 workers, and pinned memory only when CUDA is available. This avoids multiplying the full test MAT arrays across worker processes and keeps full-resolution evaluation batches bounded. Worker seeds initialize Python, NumPy and each dataset's augmentation generator.

## Verification outcome

The 13 dedicated B1 data-pipeline tests pass. The all-sample verification command checked 192 train pairs, 24 validation pairs, 14 internal comparison pairs and 928 official test blocks. All loaded RAW arrays were finite; all pack/unpack checks were exact; all target conversions had the expected shape and `[0,1]` range. The captured RAW values included negatives; these remained intact. No source value exceeded 1 in this audited expanded copy (maximum observed was 1.0), so preservation of a value above 1 is covered by synthetic unit fixtures rather than falsely claimed as a dataset observation.

Commands and measured memory/timing are recorded in `B1_DATA_PIPELINE_PERFORMANCE.md`. Tensor values and representative per-split figures are in `B1_TENSOR_VERIFICATION_REPORT.md` and `B1_VISUAL_INSPECTION.md`.

## Boundary

This completes dataset/tensor integration only. It does not implement a U-Net, create a checkpoint, compute image-quality metrics, or train a model. The official target remains the dataset camera-pipeline rendering; visual inspection does not establish colorimetric accuracy.
