1RUA24CSE0023, 1RUA24CSE0024, 1RUA24CSE0012, 1RUA24CSE0020, 1RUA24CSE0047
# Comparative Evaluation Report: Problem Statement 2 (P2)
## PRISM: Intelligent and Computationally Efficient Smartphone ISP

- **Problem ID**: P2 (Intelligent Smartphone ISP)
- **Team ID**: T37
- **Team Members & USNs**:
  - Aditya P Dixit (`1RUA24CSE0023`)
  - Aditya Pujari (`1RUA24CSE0024`)
  - Adarsh Suri (`1RUA24CSE0012`)
  - Aditya Anandh (`1RUA24CSE0020`)
  - Ananya Pramod Kulkarni (`1RUA24CSE0047`)
- **Institution / Hackathon**: Computer Vision Hackathon 2026
- **Evaluation Run ID**: `PRISM-P6-COMP-MPS-FULL-20261107-001`
- **Evaluation Protocol**: 14 held-out full-frame 12 MP images across 7 scenes (Scenes 009 & 010; Apple iPhone & Google Pixel), 10 interleaved round-robin repetitions per configuration on Apple Silicon GPU (Metal Performance Shaders).

---

## 1. Primary Metrics Comparison (Strict P2 Document Order)

| Metric | B0 Classical Baseline | B1 Learned Baseline (U-Net) | B2 PRISM Hybrid (5% Budget) | B2 PRISM Hybrid (10% Budget) |
|---|---|---|---|---|
| **PSNR (dB)** | 26.08 dB | 29.35 dB | 26.30 dB | 26.43 dB |
| **SSIM** | 0.5721 | 0.8873 | 0.5788 | 0.5848 |
| **$\Delta E_{00}$ (Color Error)** | 2.14 | 4.86 | 2.18 | 2.21 |
| **Perceptual Metric (LPIPS/MSE)** | 0.00277 | 0.00118 | 0.00261 | 0.00252 |
| **Model Parameters** | 0 | 357,155 | 357,155 (Shared) | 357,155 (Shared) |
| **Model Size (MB)** | 0.00 MB | 1.43 MB | 1.43 MB | 1.43 MB |
| **Inference Latency (s)** | 1.387 s | 0.578 s | 1.598 s | 1.640 s |
| **Memory Consumption (Peak RSS)**| 2.75 GB | 2.75 GB | 2.75 GB | 2.75 GB |
| **Computational Footprint (GMACs)**| 0.00 GMACs | 383.53 GMACs | **13.07 GMACs** | **24.59 GMACs** |
| **Compute Reduction vs B1 (%)** | 100.0% | 0.0% (Reference) | **96.6% Reduction** | **93.6% Reduction** |

---

## 2. Complete Multi-Budget Evaluation for B2 (PRISM Hybrid)

| Configuration | Scene PSNR (dB) | Scene SSIM | Mean Latency (s) | Conv2d GMACs | Arithmetic Savings vs Full Neural |
|---|---|---|---|---|---|
| **$B_0$ Classical** | 26.08 | 0.5721 | 1.387 s | 0.00 GMACs | 100.0% |
| **$B_1$ Learned** | 29.35 | 0.8873 | 0.578 s | 383.53 GMACs | 0.0% (Reference) |
| **$B_2$ ($\alpha = 0.00$)** | 26.08 | 0.5726 | 1.829 s | 0.00 GMACs | 100.0% |
| **$B_2$ ($\alpha = 0.01$)** | 26.11 | 0.5736 | 1.611 s | 2.68 GMACs | 99.3% |
| **$B_2$ ($\alpha = 0.03$)** | 26.21 | 0.5762 | 1.604 s | 7.93 GMACs | 97.9% |
| **$B_2$ ($\alpha = 0.05$)** | **26.30** | **0.5788** | **1.598 s** | **13.07 GMACs** | **96.6%** |
| **$B_2$ ($\alpha = 0.10$)** | **26.43** | **0.5848** | **1.640 s** | **24.59 GMACs** | **93.6%** |
| **$B_2$ ($\alpha = 0.25$)** | 26.43 | 0.6007 | 1.738 s | 52.55 GMACs | 86.3% |
| **$B_2$ ($\alpha = 1.00$)** | 26.24 | 0.6468 | 2.055 s | 153.95 GMACs | 59.9% |

---

## 3. Hardware & Experimental Metadata
- **Split**: Locked `internal_test` (14 paired full-frame test images across 7 scene groups).
- **Split Hash (SHA-256)**: `9e93f75071fb3fc4bf5e2c0e0ab127cc90104caafe231dda83f2a2d388ec441c`.
- **Model Checkpoint**: `experiments/P3/runs/B1-contract100-mps-20261106-002/best.pt` (Epoch 45).
- **Checkpoint SHA-256**: `6010a29a5efbb03761161a2e92ff82fac8b6df4f69a8d8d907860ada22f640cd`.
- **Device & Precision**: Apple Silicon MPS GPU, 32-bit floating point unclipped sensor processing.
