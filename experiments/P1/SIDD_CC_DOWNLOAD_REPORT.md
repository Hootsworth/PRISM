# SIDD-CC Selective RAW Download Report

**Project:** PRISM (Physics-guided Reconstruction Inspection and Sensor Monitoring)  
**Phase:** P1 (Dataset Ingestion and RAW Inventory)  
**Date:** 2026-11-06  
**Operator:** Codex / Antigravity Agent  
**Repository Source:** Hugging Face [`MhdMohammadi/SIDD-CC`](https://huggingface.co/datasets/MhdMohammadi/SIDD-CC)  
**License:** SIDD-CC card identifies `sidd-terms-of-use` and links the official SIDD page. That linked page currently states the dataset and associated code are under the MIT License. Retain SIDD-CC and original SIDD attribution; see the licensing correction in the P1 audit.
**Local Destination:** `data/external/sidd_cc/`

---

## 1. Executive Summary

A selective, representative subset of the SIDD-CC ("Color-Corrected") Bayer RAW dataset was downloaded directly into the PRISM project workspace using the Hugging Face Python API (`huggingface_hub.snapshot_download`). 

- **Full remote dataset size:** ~22.90 GB (Raw modality) / 8.1 GB (sRGB modality)
- **Target download budget:** 500 MB – 2.0 GB (hard gate: ≤ 3.0 GB)
- **Actual subset downloaded:** **1,060.01 MB (1.035 GB)** across 35 files (0 MB sRGB, 0 MB model weights)
- **Disk consumption:** 1,111,505,706 bytes
- **Scenes captured:** 5 complete scenes covering all 5 smartphone camera models in the SIDD distribution (Google Pixel, Samsung Galaxy S6, Apple iPhone 7, Motorola Nexus 6, LG G4) across diverse ISO levels (100 to 3200) and lighting conditions (3200K, 4400K, 5500K; Normal and Low illuminance).

---

## 2. Pre-Download Repository Inspection

Before issuing any download command, the repository tree and metadata were inspected via `huggingface_hub.HfApi`:
1. The remote repository holds 1,164 files totaling ~31 GB (Raw: 696 files / 22.90 GB; SRGB: 468 files / 8.1 GB).
2. The `Raw/train/Data/` directory contains 115 scene instances, each containing exactly 6 files:
   - 2 clean RAW images (`*_GT_RAW_*.MAT`)
   - 2 noisy RAW images (`*_NOISY_RAW_*.MAT`)
   - 2 metadata structures (`*_METADATA_RAW_*.MAT`)
3. The average size per complete scene is ~186 MB to ~244 MB.
4. The test split (`Raw/test/`) consists of pre-extracted block tensors (`BenchmarkGtBlocksRaw.mat`, `BenchmarkNoisyBlocksRaw.mat`) totaling 464 MB, but lacks per-instance capture metadata (black level, white level, AsShotNeutral, ColorMatrix2, CFA pattern).
5. Full train scenes were selected because they provide complete, full-resolution 2D Bayer mosaic arrays with self-contained sensor calibration metadata.

---

## 3. Subset Selection Rationale

To maximize diversity under the 2 GB guideline without duplicating redundant captures:

| Instance Identifier | Camera Model | Scene ID | ISO | Shutter | Illuminant Temp | Brightness | Rationale |
|---|---|---|---|---|---|---|---|
| `0017_001_GP_00100_00060_5500_N` | Google Pixel (GP) | 001 | 100 | 1/60s | 5500 K | Normal | Baseline daylight scene; benchmark instance in SIDD-CC docs; BGGR CFA. |
| `0005_001_S6_00100_00060_4400_N` | Samsung S6 (S6) | 001 | 100 | 1/60s | 4400 K | Normal | Fluorescent lighting; 16:9 16MP aspect; GRBG CFA. |
| `0032_001_IP_00800_01000_3200_N` | Apple iPhone 7 (IP) | 001 | 800 | 1/1000s | 3200 K | Normal | 12-bit sensor bit depth (WhiteLevel 4095); higher noise regime; RGGB CFA; warm tungsten. |
| `0076_004_N6_03200_00320_3200_L` | Motorola Nexus 6 (N6) | 004 | 3200 | 1/320s | 3200 K | Low | Extreme high ISO (3200); low-light shot-noise dominated; scene 004; BGGR CFA. |
| `0025_001_G4_00100_00060_5500_L` | LG G4 (G4) | 001 | 100 | 1/60s | 5500 K | Low | Low-light daylight capture; BGGR CFA; 16MP wide aspect. |

Documentation and split index assets included:
- `README.md`
- `Raw/README.md`
- `Raw/train/Scene_Instances.txt`
- `Raw/test/test_scenes_info.txt`
- `Raw/test/BenchmarkBlocks32.mat`

Total calculated download size prior to execution: **1,060.01 MB** (well below the 3.0 GB hard ceiling).

---

## 4. Download Execution

### Environment & Tooling
- **OS:** macOS 14.x (Apple Silicon arm64)
- **Python:** 3.9.6
- **Tooling:** Virtual environment `.venv` with `huggingface_hub` 1.8.0, `scipy` 1.13.1, `numpy` 2.0.2
- **Download mechanism:** Python script `download_sidd_subset.py` calling `huggingface_hub.snapshot_download` with explicit `allow_patterns`.

### Exact Download Execution Command
```bash
.venv/bin/python3 download_sidd_subset.py
```

### Script Implementation (`download_sidd_subset.py`)
```python
from pathlib import Path
from huggingface_hub import snapshot_download

TARGET_DIR = Path("data/external/sidd_cc").resolve()
SELECTED_SCENES = [
    "0017_001_GP_00100_00060_5500_N",
    "0005_001_S6_00100_00060_4400_N",
    "0032_001_IP_00800_01000_3200_N",
    "0076_004_N6_03200_00320_3200_L",
    "0025_001_G4_00100_00060_5500_L",
]
ALLOW_PATTERNS = [
    "README.md",
    "Raw/README.md",
    "Raw/train/Scene_Instances.txt",
    "Raw/test/test_scenes_info.txt",
    "Raw/test/BenchmarkBlocks32.mat",
]
for s in SELECTED_SCENES:
    ALLOW_PATTERNS.append(f"Raw/train/Data/{s}/*")

snapshot_download(
    repo_id="MhdMohammadi/SIDD-CC",
    repo_type="dataset",
    allow_patterns=ALLOW_PATTERNS,
    local_dir=str(TARGET_DIR),
    max_workers=4,
)
```

---

## 5. File Manifest & Cryptographic Hashes

| Relative File Path | Size (Bytes) | SHA-256 Digest |
|---|---|---|
| `README.md` | 4,084 | `7f7522ed2544ad5fc3671f760da6814fcba3c690f074a38753238ca7f4fc6659` |
| `Raw/README.md` | 4,111 | `f2dd417c63e146747201c107be998797b1f66e01a884fafeecae7dfbe5127ee8` |
| `Raw/test/BenchmarkBlocks32.mat` | 294 | `a18dcea9fd1448c7bb86ec11299f1eeef229ab38c5cdf4b0da91ca1c6bf6ccc4` |
| `Raw/test/test_scenes_info.txt` | 660 | `cccedc4d3fc39e083c267a8406541fce9a5fe9bc353724ff00dcf85b4122d1fe` |
| `Raw/train/Scene_Instances.txt` | 3,565 | `cebdbe58fd908b8b0a9bd66c77d48386a34ea4d7ffb9fc1eb3736dd65db8b438` |
| `Raw/train/Data/0005_001_S6_00100_00060_4400_N/0005_GT_RAW_065.MAT` | 63,936,184 | `c9dacd55be7540dccb43163a28746b3c238fcb0712c96c66d9099e2aab8c5b26` |
| `Raw/train/Data/0005_001_S6_00100_00060_4400_N/0005_GT_RAW_091.MAT` | 63,936,184 | `2c8a02db5a79f856fc1e9a6f1d0d31dc4705f3204374a50e08ea0186c9a23e08` |
| `Raw/train/Data/0005_001_S6_00100_00060_4400_N/0005_METADATA_RAW_065.MAT` | 592 | `08b263d2383c60fb6d78aa5140a0c6e35ee683166f8610ac860c381c22e1aef4` |
| `Raw/train/Data/0005_001_S6_00100_00060_4400_N/0005_METADATA_RAW_091.MAT` | 592 | `08b263d2383c60fb6d78aa5140a0c6e35ee683166f8610ac860c381c22e1aef4` |
| `Raw/train/Data/0005_001_S6_00100_00060_4400_N/0005_NOISY_RAW_065.MAT` | 63,936,184 | `2c4ee568c1c374f9b163cd174270ff30754d27fe779de3c29ffed488fe80da93` |
| `Raw/train/Data/0005_001_S6_00100_00060_4400_N/0005_NOISY_RAW_091.MAT` | 63,936,184 | `d5f092a86b6de49a8616938d97e389df8e4ee2d7cf20ad195d4a79f3a88b70de` |
| `Raw/train/Data/0017_001_GP_00100_00060_5500_N/0017_GT_RAW_106.MAT` | 49,159,096 | `65b11509fc080141ab97b69dedf4b3a64c05b3d1d4e0b13dd6e8beceee8dbe7d` |
| `Raw/train/Data/0017_001_GP_00100_00060_5500_N/0017_GT_RAW_133.MAT` | 49,159,096 | `c4cdf554d080cff09f474f517eb8c0cedeeac68cbcbb0b8c067931e9de8d40e6` |
| `Raw/train/Data/0017_001_GP_00100_00060_5500_N/0017_METADATA_RAW_106.MAT` | 592 | `2c92da213b34746b2e5700ef16077efc0393bf2148c7cefb117ef49ef48f5dec` |
| `Raw/train/Data/0017_001_GP_00100_00060_5500_N/0017_METADATA_RAW_133.MAT` | 592 | `2c92da213b34746b2e5700ef16077efc0393bf2148c7cefb117ef49ef48f5dec` |
| `Raw/train/Data/0017_001_GP_00100_00060_5500_N/0017_NOISY_RAW_106.MAT` | 49,159,096 | `4f5aeb0c3984ecd3815b1acd6325f78d27d92ff9de11a567c908860f5fee4d8e` |
| `Raw/train/Data/0017_001_GP_00100_00060_5500_N/0017_NOISY_RAW_133.MAT` | 49,159,096 | `3e75b25940e99259d3e1dc1ac8c4d9f4882c752c03006b283e0f0876eb772bff` |
| `Raw/train/Data/0025_001_G4_00100_00060_5500_L/0025_GT_RAW_003.MAT` | 63,489,208 | `6884d0c43cd9d5f165cad3a62e5fbc287958df398ab6d9925f845928a582afae` |
| `Raw/train/Data/0025_001_G4_00100_00060_5500_L/0025_GT_RAW_096.MAT` | 63,489,208 | `fc30d793987d4b518697cc9809162930051b98333c1a95402ce77a36df77221f` |
| `Raw/train/Data/0025_001_G4_00100_00060_5500_L/0025_METADATA_RAW_003.MAT` | 592 | `6c9bc3d49cc5071f4afe716650f25873cf3a1452efa467a0dd2e2e4db5443a90` |
| `Raw/train/Data/0025_001_G4_00100_00060_5500_L/0025_METADATA_RAW_096.MAT` | 592 | `6c9bc3d49cc5071f4afe716650f25873cf3a1452efa467a0dd2e2e4db5443a90` |
| `Raw/train/Data/0025_001_G4_00100_00060_5500_L/0025_NOISY_RAW_003.MAT` | 63,489,208 | `afc2c7c5e6e1db77f33f89b4b4fcc70f2a59bba67d67cd16edb6deea1e62114d` |
| `Raw/train/Data/0025_001_G4_00100_00060_5500_L/0025_NOISY_RAW_096.MAT` | 63,489,208 | `ed5a3601d72f6522901cb05d63965d0fd9cf227d4211559364c433adec073118` |
| `Raw/train/Data/0032_001_IP_00800_01000_3200_N/0032_GT_RAW_019.MAT` | 48,771,256 | `d4a8a42c0b3006ca34e4c4fd6ea55312833b47b292838c5ed6b3ca504d2f5e13` |
| `Raw/train/Data/0032_001_IP_00800_01000_3200_N/0032_GT_RAW_054.MAT` | 48,771,256 | `aec2a546ca26b6aa6a589e219a4d68542465cb987354cd369ab4caa3413551b5` |
| `Raw/train/Data/0032_001_IP_00800_01000_3200_N/0032_METADATA_RAW_019.MAT` | 592 | `6039b7cf4ad4c411b5f92e3a78145091aab56d7494dd64305598dffe4856abab` |
| `Raw/train/Data/0032_001_IP_00800_01000_3200_N/0032_METADATA_RAW_054.MAT` | 592 | `6039b7cf4ad4c411b5f92e3a78145091aab56d7494dd64305598dffe4856abab` |
| `Raw/train/Data/0032_001_IP_00800_01000_3200_N/0032_NOISY_RAW_019.MAT` | 48,771,256 | `53d110519df85e2777c3dc5cb3fe6c63dff307fc16138ea9d8f7098104a57387` |
| `Raw/train/Data/0032_001_IP_00800_01000_3200_N/0032_NOISY_RAW_054.MAT` | 48,771,256 | `0bb6361200fc1ef97a836fe611ce592f51bbe598d74a3aef849f7a2ff9c71a3d` |
| `Raw/train/Data/0076_004_N6_03200_00320_3200_L/0076_GT_RAW_094.MAT` | 52,516,024 | `7f04153ddf5626d17ce0a847c3bb7d120aad131900c3c9109bb0a51b442fee08` |
| `Raw/train/Data/0076_004_N6_03200_00320_3200_L/0076_GT_RAW_118.MAT` | 52,516,024 | `0d4936919e50716d75f3c27368d46edea1c60c04f1ae19b6f0ab17d345eec6b4` |
| `Raw/train/Data/0076_004_N6_03200_00320_3200_L/0076_METADATA_RAW_094.MAT` | 592 | `f95f412fb84abf42c23847c468cdb9c2e5d6b33df467d42b885dcc299afedd6f` |
| `Raw/train/Data/0076_004_N6_03200_00320_3200_L/0076_METADATA_RAW_118.MAT` | 592 | `f95f412fb84abf42c23847c468cdb9c2e5d6b33df467d42b885dcc299afedd6f` |
| `Raw/train/Data/0076_004_N6_03200_00320_3200_L/0076_NOISY_RAW_094.MAT` | 52,516,024 | `a878185d98a2b514505eef657ed638e0be908d2c6d3e7b02f57f4fba72396b8c` |
| `Raw/train/Data/0076_004_N6_03200_00320_3200_L/0076_NOISY_RAW_118.MAT` | 52,516,024 | `67c17f2850bfc36cb097b6101086bb64db369ca471bb9b220265e70059537a4e` |

---

## 6. Verification Summary

1. **Existence:** 35 files expected; 35 files verified on disk.
2. **Hierarchy:** Original directory structure (`Raw/train/Data/{scene}/*.MAT`, `Raw/test/*`, `README.md`) preserved.
3. **Resumability & Isolation:** Temporary hub metadata cleaned; no extraneous cache files retained in workspace.
4. **Integrity:** SHA-256 hashes generated and stored in manifest CSV.

## 7. Licensing source clarification

The original report classified the data as academic/non-commercial based on an earlier interpretation. The current SIDD-CC card labels the license `sidd-terms-of-use` and links the official SIDD page, which currently states the dataset and associated code are under the MIT License. The P1 audit has been updated to record this source correction. Preserve attribution and verify the applicable notice before redistribution.

## 8. Expanded SIDD-CC RAW/SRGB acquisition

The follow-on acquisition is recorded in `experiments/P3/SIDD_CC_EXPANDED_DATA_AUDIT.md`. It fetched the RAW and SRGB modalities at revision `5eb2ab6375d7f99f36b59bb5aacb46af18409412` into `data/external/sidd_cc_expanded/`; it did not overwrite this original selective P1 subset. The expanded copy includes 115 complete train instances, two indexed pairs each, and the official 29-instance test blocks in both modalities. The source files and pair mappings have a separate SHA-256 manifest and split manifest under `data/manifests/`.
