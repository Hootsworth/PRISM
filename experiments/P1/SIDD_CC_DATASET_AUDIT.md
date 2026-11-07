# SIDD-CC RAW Dataset Audit and Feasibility Assessment

**Project:** PRISM (Physics-guided Reconstruction Inspection and Sensor Monitoring)  
**Phase:** P1 (Dataset Verification, Audit, and PRISM Feasibility Review)  
**Date:** 2026-11-06  
**Auditor:** Codex / Antigravity Agent  
**Dataset Reference:** `data/external/sidd_cc/` (Derived from `MhdMohammadi/SIDD-CC`)

---

## 1. Technical Audit of Ingested RAW Arrays

All 30 downloaded image and metadata `.MAT` files were parsed using `scipy.io.loadmat` and inspected across shapes, memory representations, value distributions, and CFA mosaic structures.

### Summary Table of Audited Scenes

| Scene Identifier | Camera | Split | Shape $(H \times W)$ | Stored Dtype | Min Value | Max Value | Mean Value | Verified CFA | BlackLevel (DNG) | WhiteLevel (DNG) | Equivalent Bit Depth |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `0005_001_S6_00100_00060_4400_N` | Samsung S6 | Train | $3000 \times 5328$ | `float32` | $0.00000$ | $0.83187$ | $0.2241$ | `GRBG` | $0.0$ | $1023.0$ | 10-bit |
| `0017_001_GP_00100_00060_5500_N` | Google Pixel | Train | $3036 \times 4048$ | `float32` | $-0.06646$ | $0.85927$ | $0.1950$ | `BGGR` | $63.75$ | $1023.0$ | 10-bit |
| `0025_001_G4_00100_00060_5500_L` | LG G4 | Train | $2988 \times 5312$ | `float32` | $-0.06674$ | $0.20751$ | $0.0422$ | `BGGR` | $64.0$ | $1023.0$ | 10-bit |
| `0032_001_IP_00800_01000_3200_N` | Apple iPhone 7 | Train | $3024 \times 4032$ | `float32` | $-0.14802$ | $0.25175$ | $0.0475$ | `RGGB` | $528.0$ | $4095.0$ | 12-bit |
| `0076_004_N6_03200_00320_3200_L` | Nexus 6 | Train | $3120 \times 4208$ | `float32` | $-0.06674$ | $1.00000$ | $0.0285$ | `BGGR` | $64.0$ | $1023.0$ | 10-bit |

---

## 2. Mathematical Verification of Bayer Mosaic Structure

A critical prerequisite for PRISM's physics-guided pipeline is verifying that the arrays are genuine single-plane Bayer mosaics rather than pre-demosaiced or multi-channel tensor structures.

### Empirical Analysis
For each 2D array, pixels were decoupled into their four interleaved $2 \times 2$ sampling sub-grids:
- Top-Left: $A[0::2, 0::2]$
- Top-Right: $A[0::2, 1::2]$
- Bottom-Left: $A[1::2, 0::2]$
- Bottom-Right: $A[1::2, 1::2]$

Under standard physical Bayer CFA configurations:
1. Two sub-grids correspond to Green filtration ($G_1, G_2$), which sample identical physical spectra through identical optical filters. Consequently, their local means must match and their spatial correlation must approach $1.0$.
2. The remaining two sub-grids sample Red and Blue, which deviate significantly based on scene chromaticity and illumination color temperature.

### Empirical Audit Measurements
- **Google Pixel (`0017`, `bggr`):**
  - Top-Left ($B$) Mean: $0.15056$
  - Top-Right ($G_1$) Mean: $0.25995$
  - Bottom-Left ($G_2$) Mean: $0.26064$ ($\Delta < 0.00069$)
  - Bottom-Right ($R$) Mean: $0.10892$
  - Green Sub-channel Correlation ($500 \times 500$ patch): **$0.9922$**
- **Apple iPhone 7 (`0032`, `rggb`):**
  - Top-Left ($R$) Mean: $0.02567$
  - Top-Right ($G_1$) Mean: $0.06364$
  - Bottom-Left ($G_2$) Mean: $0.06355$ ($\Delta < 0.00009$)
  - Bottom-Right ($B$) Mean: $0.03553$
  - Green Sub-channel Correlation ($500 \times 500$ patch): **$0.9932$**
- **Samsung Galaxy S6 (`0005`, `grbg`):**
  - Top-Left ($G_1$) Mean: $0.28828$
  - Top-Right ($R$) Mean: $0.15781$
  - Bottom-Left ($B$) Mean: $0.16140$
  - Bottom-Right ($G_2$) Mean: $0.28910$ ($\Delta < 0.00082$)
  - Green Sub-channel Correlation ($500 \times 500$ patch): **$0.9953$**

**Conclusion:** The arrays strictly represent genuine, single-plane Bayer mosaics with documented, verified phase alignments (`BGGR`, `RGGB`, `GRBG`).

---

## 3. Data Representation and Normalization Semantics

### Critical Operational Finding: Pre-subtracted, Unclipped Normalization
As documented in SIDD-CC and confirmed by empirical array analysis:
$$\text{stored\_pixel} = \frac{\text{raw\_pixel} - \text{BlackLevel}}{\text{WhiteLevel} - \text{BlackLevel}}$$

1. **Do not subtract black level again:** The stored `.MAT` values are already normalized to nominal $[0, 1]$.
2. **Sub-zero values are physical noise, not corrupted pixels:** Values below zero (e.g. $-0.1480$ on iPhone 7) occur because the ground-truth and noisy values were black-level-subtracted without zero-clipping. In dark regions, read noise fluctuates below the black point. For learned baseline models that require strictly positive inputs, clipping or offset handling must be declared in preprocessing.

---

## 4. Metadata Inventory & Gap Analysis

### Verified Available Metadata
- **CFA Layout (`cfa_str`):** Explicitly present in every `METADATA_RAW_*.MAT` (`bggr`, `rggb`, `grbg`).
- **BlackLevel & WhiteLevel:** Original camera levels preserved (e.g. Samsung: 0 / 1023; Pixel: 63.75 / 1023; iPhone: 528 / 4095).
- **AsShotNeutral:** 3-element white balance scaling coefficients $[R_w, G_w, B_w]$ (e.g., iPhone: $[0.4063, 1.0, 0.6015]$).
- **ColorMatrix2:** $3 \times 3$ DNG matrix mapping XYZ coordinates to reference camera-native coordinates (row-scan coefficient order); a camera-to-XYZ transform requires inversion. The earlier “sensor RGB to XYZ/sRGB” description was incorrect; see the DNG convention and limitation notes in `experiments/P2/B0_IMPLEMENTATION_REPORT.md` and `B0_VERIFICATION_REPORT.md`.
- **Exposure / Sensor Parameters:** ISO rating, shutter speed, illuminant temperature, brightness code, camera ID encoded in instance strings and verified against test scene logs.
- **Alignment:** Clean ground truth is estimated from burst averages of ~150 aligned DNG frames. Alignment between NOISY and GT pairs is verified.

### Documented Missing / Unavailable Metadata
1. **Noise Level Function (NLF) Coefficients:** Parametric shot/read noise parameters ($\beta_1, \beta_2$) are **not** bundled in `METADATA_RAW_*.MAT`. They must be calibrated directly from the paired NOISY/GT raw arrays in PRISM.
2. **Lens Shading / Vignette Maps:** Optical falloff calibration maps are not included.
3. **Paired sRGB Ground Truth:** The downloaded subset contains Bayer RAW modality only; rendered camera sRGB images (`.PNG`) are **not** present in this subset.
4. **Raw Burst Frames:** The underlying 150 individual burst frames are held in a separate repository (`SIDD-CC-DNG`) and not included locally.

---

## 5. Licensing and Rights Audit

- **Dataset card:** SIDD-CC labels its license `sidd-terms-of-use` and links the official SIDD page. The linked official page currently states that the dataset and associated code are under the MIT License. Follow that linked terms/source and the SIDD-CC attribution requirements; preserve the copyright/license notice with any permitted use.
- **Correction to the earlier audit text:** the phrase “Academic / Non-commercial research” was not supported by the linked official license page and has been removed. Do not infer a non-commercial restriction from that earlier wording.
- **Origin:** Derived from the Smartphone Image Denoising Dataset (Abdelhamed et al., CVPR 2018) via SIDD-CC (Mohammadi et al., 2026).
- **Compliance Restrictions:**
  - Preserve the relevant MIT notice and attribution, and cite both SIDD-CC and the original SIDD as required by SIDD-CC.
  - Do not redistribute third-party image archives without checking applicable dataset terms and rights for the specific files.
  - Large binary data must never be committed to Git.
  - Citations to both Mohammadi et al. (2026) and Abdelhamed et al. (2018) are required by the SIDD-CC dataset card.

---

## 6. PRISM Experiment Feasibility Matrix

| PRISM Capability / Experiment | Feasibility Status | Scientific Rationale & Constraints |
|---|---|---|
| **Bayer RAW Ingestion** | **FEASIBLE (VERIFIED)** | Full-resolution Bayer mosaics with verified CFA phase patterns (`bggr`, `rggb`, `grbg`) can be ingested directly into NumPy/PyTorch raw loaders. |
| **Noise Modelling** | **FEASIBLE (VERIFIED)** | High/low ISO pairs across 5 cameras enable heteroscedastic Gaussian/Poisson-Gaussian noise parameter fitting ($\sigma^2(y) = \beta_1 y + \beta_2$) from residual difference arrays $(I_{\text{noisy}} - I_{\text{gt}})$. |
| **Sensor-Space Remosaicing** | **FEASIBLE (VERIFIED)** | Verified CFA patterns and metadata enable sensor-native mosaicing and remosaicing transformations. |
| **Residual Consistency Auditing** | **FEASIBLE (VERIFIED)** | Clean ground-truth RAW allows computing ground-truth residual statistics to evaluate audit scores. |
| **Tile-Level Anomaly Detection** | **FEASIBLE (VERIFIED)** | High-resolution arrays ($3000 \times 4000+$) support tile partitioning ($64 \times 64$, $128 \times 128$) for localized noise and anomaly inspection. |
| **RAW-Domain Denoising (Baseline B1)** | **FEASIBLE (VERIFIED)** | Exact paired noisy/clean RAW arrays exist with verified spatial alignment, enabling paired RAW denoising training and validation. |
| **Classical ISP (Baseline B0)** | **FEASIBLE (WITH ISP STAGE)** | Demosaicing, white-balancing (`AsShotNeutral`), and color conversion (`ColorMatrix2`) can be applied to produce uncalibrated or linear RGB outputs. |
| **End-to-End RAW-to-sRGB Reconstruction Evaluation** | **BLOCKED (INCOMPLETE GT)** | Paired sRGB ground truth is **not** present in this download subset. Full-pipeline PSNR/SSIM/$\Delta E_{00}$ against manufacturer ISP cannot be computed without sRGB targets or the Modular ISP data subset. |
| **Perceptual Metric Evaluation (LPIPS in display space)** | **BLOCKED (INCOMPLETE GT)** | LPIPS requires sRGB display-space target pairs. Must not be evaluated on raw Bayer mosaics. |

---

## 7. Initial subset gate decision

**Decision at the original five-scene ingest: PASS for the RAW-only scope.** The initial P1 copy satisfied Bayer RAW ingestion and sensor-audit needs, but did not contain sRGB targets. Section 9 records the follow-on expanded acquisition, which closes that initial target-data gap for supervised RAW-to-sRGB B1.

## 8. P3 follow-on: paired RAW feasibility

On 2026-11-06, all five local scene directories were inspected directly. Ten noisy/GT filename pairs (two per scene) were loaded from their MAT files. All 10 pairs had matching 2D float32 dimensions, finite values, and matching supported CFA metadata; shared AsShotNeutral, ColorMatrix2, and original level metadata were present. The source MAT arrays were read only. The full pair map and descriptive residual findings are in `experiments/P3/B1_FEASIBILITY_REPORT.md`.

The noisy-to-GT residual RMSE spans 0.00439–0.03508 in already-normalized sensor units. A binned linear fit of residual variance versus target intensity is positive in every pair, but is not a calibrated noise-level function: it includes scene texture, any registration error, and burst-target uncertainty. Do not use these fit coefficients as universal shot/read noise parameters.

These pairs support a small RAW-domain denoising pilot. They do not provide RGB targets. The upstream SIDD-CC card documents a separate 8-bit rendered SRGB modality, 115 train and 29 test instances, and SIDD terms of use. That modality has not been acquired here; exact cross-modality sample-index mapping, crop and pixel alignment must be verified on files before a Bayer-RAW-to-sRGB experiment. The current five scene groups are all train and each combines one camera with one scene/condition, so they do not support credible independent camera/scene generalization estimates.

## 9. Expanded RAW/SRGB acquisition and audit

The next milestone acquired the full SIDD-CC RAW and SRGB modalities into the separate `data/external/sidd_cc_expanded/` directory at pinned revision `5eb2ab6375d7f99f36b59bb5aacb46af18409412`. The original P1 subset remains unmodified; its inventory SHA-256 remains `21ec28480a6029c02b7679e4242a9a1eedd1ed206b5ea70b18399d5a149222df`.

The expanded audit verifies 115 identical train instance lists, 230 complete noisy-RAW to GT-sRGB sample-index mappings, full-resolution dimensions, Bayer float32 RAW and RGB PNG formats, and per-training-capture CFA/AsShotNeutral/ColorMatrix2/BlackLevel/WhiteLevel metadata. Frozen B0 edge-map registration on center crops found the best integer offset at `(0,0)` for all 230 pairs in a ±3-pixel search. The 29 test scene lists match across modalities and all 32 block coordinates match; all four test tensors have the documented dimensions/dtypes and the RAW block arrays are finite. Exact details, limitations, and manifests are in `experiments/P3/SIDD_CC_EXPANDED_DATA_AUDIT.md`.

The official test block files contain no per-capture CFA, camera, ISO, AsShotNeutral or ColorMatrix2 metadata. These fields remain unknown. The paired data now supports a full learned Bayer-RAW-to-camera-sRGB baseline, but the metadata-dependent frozen B0 cannot be evaluated on that official test split until an authorized, verified metadata mapping is found. A leak-safe instance split is locked at 98 train / 17 validation / 29 official test; no patches cross instance groups.


### B1 scene-group split correction (2026-11-06)

The 98/17 instance-random split previously recorded for B1 crosses the dataset's documented scene-number field. It is superseded for B1 by a grouped split: 96 train instances, 12 validation instances, 7 internal paired B0/B1 comparison instances, and the unchanged 29-scene official test. See `experiments/P3/B1_SPLIT_INTEGRITY_REPORT.md`. This changes no P1 source data.
