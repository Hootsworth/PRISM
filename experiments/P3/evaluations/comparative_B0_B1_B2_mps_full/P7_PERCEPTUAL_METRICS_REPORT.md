# PRISM P7 Perceptual Metrics Evaluation Report

**Evaluation Run:** `PRISM-P6-COMP-MPS-FULL-20261107-001`  
**Date:** `2026-11-07T12:00:42+0530`  
**Frameworks:** `pyiqa` v0.1.15.post2, `torch` v2.8.0, `torchvision` v0.23.0  
**Dataset:** Frozen SIDD-CC Internal Test Set (14 image pairs across 7 distinct scene instances)  

---

## Executive Summary

This evaluation completes the missing perceptual metrics (**LPIPS** and **NIQE**) for the PRISM P6 comparative evaluation across all 126 generated output images ($14 \text{ samples} \times 9 \text{ models/budgets}$) strictly utilizing the frozen P6 output artifacts without rerunning ISP pipelines or retraining any models.

### Key Findings:
1. **Full-Reference Perceptual Quality (LPIPS, AlexNet)**: Lower is better.
   - **B0 Classical**: Scene-macro LPIPS = **0.5747** (Sample mean: 0.5747 ± 0.1760)
   - **B1 Learned**: Scene-macro LPIPS = **0.3311** (Sample mean: 0.3311 ± 0.0814)
   - **B2 PRISM (5% Budget)**: Scene-macro LPIPS = **0.5727** (Sample mean: 0.5727 ± 0.1656)
   - **B2 PRISM (100% Budget)**: Scene-macro LPIPS = **0.5801** (Sample mean: 0.5801 ± 0.1085)

2. **No-Reference Naturalness (NIQE)**: Lower denotes statistical resemblance to natural image statistics.
   - **B0 Classical**: Scene-macro NIQE = **7.0642** (Sample mean: 7.0642 ± 1.5542)
   - **B1 Learned**: Scene-macro NIQE = **10.4047** (Sample mean: 10.4047 ± 1.6959)
   - **B2 PRISM (5% Budget)**: Scene-macro NIQE = **6.5855** (Sample mean: 6.5855 ± 1.7490)

> [!IMPORTANT]
> **Methodological Clarification regarding NIQE**:
> NIQE is a *no-reference* metric evaluating deviation from a corpus of natural scene statistics (NSS). Highly smoothed, over-regularized or oversaturated outputs often trigger lower (better) NIQE scores despite lacking fine ground-truth textural fidelity. Conversely, high-frequency granular sensor noise and aggressive sharpening can penalize NIQE. Full-reference LPIPS and PSNR/SSIM remain the definitive metrics for ground-truth restoration fidelity.

---

## Comparative System Summary (Paper-Ready Table)

Below is the complete comparison of B0 Classical, B1 Learned, and B2 PRISM configurations across fidelity, perceptual, and efficiency metrics on the frozen 14-image held-out test set:

| Architecture / Model | PSNR (dB) ↑ | SSIM ↑ | LPIPS (AlexNet) ↓ | NIQE ↓ | Median Full-Path Latency (s) | Learned Conv GMACs |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **B0 Classical Baseline** | 26.08 | 0.5721 | 0.5747 | 7.0642 | 1.379 | 0.00 |
| **B1 Learned UNet** | 29.35 | 0.8873 | 0.3311 | 10.4047 | 0.573 | 383.53 |
| **B2 PRISM (0% Budget)** | 26.08 | 0.5726 | 0.5747 | 7.0642 | 1.819 | 0.00 |
| **B2 PRISM (1% Budget)** | 26.11 | 0.5736 | 0.5749 | 6.8746 | 1.650 | 2.68 |
| **B2 PRISM (3% Budget)** | 26.21 | 0.5762 | 0.5738 | 6.6146 | 1.616 | 7.93 |
| **B2 PRISM (5% Budget)** | **26.30** | **0.5788** | **0.5727** | **6.5855** | **1.613** | **13.07** |
| **B2 PRISM (10% Budget)** | 26.43 | 0.5848 | 0.5700 | 6.5328 | 1.630 | 24.59 |
| **B2 PRISM (25% Budget)** | 26.43 | 0.6007 | 0.5697 | 6.4558 | 1.712 | 52.55 |
| **B2 PRISM (100% Budget)** | 26.24 | 0.6468 | 0.5801 | 7.2316 | 1.796 | 153.95 |

---

## Detailed Scene-by-Scene Breakdown

### 1. Scene-Macro Means vs Sample-Level Dispersion

| Architecture | Scene-Macro LPIPS | Sample Mean ± Std (LPIPS) | Scene-Macro NIQE | Sample Mean ± Std (NIQE) |
| :--- | :---: | :---: | :---: | :---: |
| `B0_Classical` | 0.5747 (±0.1831) | 0.5747 ± 0.1760 | 7.0642 (±1.6175) | 7.0642 ± 1.5542 |
| `B1_Learned` | 0.3311 (±0.0846) | 0.3311 ± 0.0814 | 10.4047 (±1.7649) | 10.4047 ± 1.6959 |
| `B2_budget_000` | 0.5747 (±0.1831) | 0.5747 ± 0.1760 | 7.0642 (±1.6175) | 7.0642 ± 1.5542 |
| `B2_budget_001` | 0.5749 (±0.1809) | 0.5749 ± 0.1738 | 6.8746 (±1.6257) | 6.8746 ± 1.5624 |
| `B2_budget_003` | 0.5738 (±0.1766) | 0.5738 ± 0.1697 | 6.6146 (±1.8038) | 6.6146 ± 1.7331 |
| `B2_budget_005` | 0.5727 (±0.1723) | 0.5727 ± 0.1656 | 6.5855 (±1.8203) | 6.5855 ± 1.7490 |
| `B2_budget_010` | 0.5700 (±0.1625) | 0.5700 ± 0.1561 | 6.5328 (±1.8604) | 6.5328 ± 1.7875 |
| `B2_budget_025` | 0.5697 (±0.1401) | 0.5697 ± 0.1346 | 6.4558 (±1.9180) | 6.4558 ± 1.8428 |
| `B2_budget_100` | 0.5801 (±0.1129) | 0.5801 ± 0.1085 | 7.2316 (±1.5265) | 7.2316 ± 1.4668 |

### 2. Per-Scene Perceptual Scores (7 Instances)

| Group ID | Camera | Model | PSNR (dB) | SSIM | LPIPS ↓ | NIQE ↓ |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| `0192_009_IP_00100_00200_3200_N` | IP | `B0_Classical` | 25.01 | 0.7213 | 0.4738 | 5.2505 |
| `0194_009_IP_01600_04000_3200_N` | IP | `B0_Classical` | 26.46 | 0.4079 | 0.7475 | 6.4549 |
| `0195_009_IP_01600_04000_5500_L` | IP | `B0_Classical` | 27.64 | 0.3811 | 0.6340 | 9.5743 |
| `0196_009_IP_00800_02000_5500_L` | IP | `B0_Classical` | 27.22 | 0.4010 | 0.6991 | 8.5732 |
| `0197_009_IP_00100_00200_5500_L` | IP | `B0_Classical` | 29.43 | 0.5776 | 0.7012 | 7.8025 |
| `0198_010_GP_00100_00200_5500_N` | GP | `B0_Classical` | 23.78 | 0.8916 | 0.2207 | 6.2046 |
| `0199_010_GP_00800_01600_5500_N` | GP | `B0_Classical` | 22.99 | 0.6243 | 0.5463 | 5.5891 |
| `0192_009_IP_00100_00200_3200_N` | IP | `B1_Learned` | 25.49 | 0.9318 | 0.3202 | 8.9150 |
| `0194_009_IP_01600_04000_3200_N` | IP | `B1_Learned` | 31.95 | 0.9169 | 0.2988 | 10.1279 |
| `0195_009_IP_01600_04000_5500_L` | IP | `B1_Learned` | 34.85 | 0.7746 | 0.2319 | 13.7638 |
| `0196_009_IP_00800_02000_5500_L` | IP | `B1_Learned` | 34.38 | 0.8515 | 0.2876 | 11.6251 |
| `0197_009_IP_00100_00200_5500_L` | IP | `B1_Learned` | 33.01 | 0.9119 | 0.2820 | 10.2471 |
| `0198_010_GP_00100_00200_5500_N` | GP | `B1_Learned` | 22.85 | 0.9141 | 0.4436 | 8.8879 |
| `0199_010_GP_00800_01600_5500_N` | GP | `B1_Learned` | 22.92 | 0.9101 | 0.4536 | 9.2664 |
| `0192_009_IP_00100_00200_3200_N` | IP | `B2_budget_005` | 25.27 | 0.7260 | 0.4801 | 5.2548 |
| `0194_009_IP_01600_04000_3200_N` | IP | `B2_budget_005` | 26.59 | 0.4295 | 0.7279 | 5.7825 |
| `0195_009_IP_01600_04000_5500_L` | IP | `B2_budget_005` | 27.64 | 0.3813 | 0.6340 | 9.5743 |
| `0196_009_IP_00800_02000_5500_L` | IP | `B2_budget_005` | 27.22 | 0.4012 | 0.6991 | 8.5732 |
| `0197_009_IP_00100_00200_5500_L` | IP | `B2_budget_005` | 29.58 | 0.5919 | 0.6825 | 6.6159 |
| `0198_010_GP_00100_00200_5500_N` | GP | `B2_budget_005` | 24.30 | 0.8907 | 0.2363 | 5.6252 |
| `0199_010_GP_00800_01600_5500_N` | GP | `B2_budget_005` | 23.52 | 0.6312 | 0.5489 | 4.6727 |

---

## Technical Validation & Reproducibility Audit

- **Reference Integrity**: All 14 ground truth pairs were verified against the audited SHA-256 manifests.
- **Image Integrity**: All 126 evaluated PNGs were verified to match the frozen P6 outputs identically.
- **Zero Alteration**: No image cropping, resizing, or lossy conversions were applied.
- **Numerical Stability**: 100% of the 126 LPIPS evaluations and 126 NIQE evaluations produced finite, non-NaN values.
- **Execution Hardware**: LPIPS evaluated on Apple MPS (Metal Performance Shaders); NIQE evaluated on CPU (IEEE 754 64-bit precision).

```
perceptual_metrics.csv SHA-256: 723c0c6345da326e7e534b8134d815e3a551f07b0e02e822c42b5e81731e46c0
perceptual_metrics_summary.json SHA-256: 9d311748ef55ff2c2bc1d14746311dac8e4bd2e99b6565e3e57d0920ad1a9ce1
```