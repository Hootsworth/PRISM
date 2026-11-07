---
title: PRISM Smart Camera ISP
emoji: 📷
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# PRISM: Physics-Guided Reconstruction Inspection and Sensor Monitoring
### Intelligent and Computationally Efficient Smartphone Image Signal Processor (ISP)

**Problem Statement**: Problem 2 (P2) — Computer Vision Hackathon 2026  
**Team ID**: **T37**  

---

## 👥 Team Members

| Name | USN | Role |
|---|---|---|
| **Aditya P Dixit** | `1RUA24CSE0023` | System Architecture, Classical Pipeline & Guided Color Fusion |
| **Aditya Pujari** | `1RUA24CSE0024` | Neural Refiner Architecture & Model Optimization |
| **Adarsh Suri** | `1RUA24CSE0012` | Sensor-Space Audit, Noise Modeling & FDR Scheduling |
| **Aditya Anandh** | `1RUA24CSE0020` | Comparative Evaluation Protocol & Benchmarking |
| **Ananya Pramod Kulkarni** | `1RUA24CSE0047` | Interactive Web Application, UI/UX & Data Pipelines |

---

## 📖 Further Reading & Documentation

[![Design Document](https://img.shields.io/badge/📖_Design_Document-Full_13--Section_Specification-blue?style=for-the-badge)](docs/design_document.md)
[![Evaluation Report](https://img.shields.io/badge/📊_Evaluation_Results-eval__P2__T37.md-green?style=for-the-badge)](eval_P2_T37.md)
[![Architecture](https://img.shields.io/badge/🏛️_System_Architecture-Mermaid_&_Diagram-purple?style=for-the-badge)](docs/architecture.md)
[![Research Paper](https://img.shields.io/badge/📄_Research_Paper-PDF_(6_Pages)-red?style=for-the-badge)](PRISM_Research_Paper.pdf)
[![Ablation Study](https://img.shields.io/badge/🔬_Ablation_Study-Pareto_Curves-orange?style=for-the-badge)](docs/ablation_study.md)
[![Failure Analysis](https://img.shields.io/badge/⚠️_Failure_Analysis-3_Cases_&_Ethics-yellow?style=for-the-badge)](docs/failure_analysis.md)
[![Traceability Matrix](https://img.shields.io/badge/📋_Traceability-Requirements_Matrix-lightgrey?style=for-the-badge)](docs/00_P2_REQUIREMENTS_TRACEABILITY.md)
[![Project Ledger](https://img.shields.io/badge/📜_Project_Ledger-Run_History-black?style=for-the-badge)](PROJECT_LEDGER.md)

---

## 🌟 Executive Summary & Motivation

Smartphone photography operates under stringent physical constraints: tiny sub-micron sensor pixels capture photons with severe Poisson-Gaussian noise, optical aberrations, and limited dynamic range. 

Existing solutions represent two problematic extremes:
1. **Classical Hardware ISPs ($B_0$)**: Fast and power-efficient, but employ hand-engineered spatial filters that struggle in low-light conditions, causing noise amplification, chromatic mottling, and demosaicing zipper artifacts.
2. **Deep Learning ISPs ($B_1$)**: Full-frame convolutional neural networks (e.g., PyNET, deep U-Nets) deliver impressive visual restoration, but consume an astronomical **383.5+ GMACs per 12-megapixel frame**, inducing thermal throttling, frame drops, and battery depletion within minutes of mobile usage.

### The PRISM Solution ($B_2$)
**PRISM** resolves this fundamental tension through **spatial noise sparsity**. In natural scenes, the vast majority of image tiles (well-lit backgrounds, smooth skies, flat regions) are reconstructed with high fidelity by the fast classical path. Only a small fraction of tiles (complex textures, fine book text, deep shadows) genuinely require deep learning.

PRISM implements:
1. **Fidelity-First Classical Baseline ($B_0$)**: Executes a deterministic pipeline (bilinear demosaic, Bradford chromatic adaptation, camera `ColorMatrix2`, and Reinhard tone reproduction) in linear sensor space.
2. **Physical Sensor-Consistency Audit**: Re-mosaics the linear estimate back to the sensor's physical Bayer CFA sample sites and evaluates standardized Pearson residuals under a calibrated heteroscedastic noise model:
   $$\sigma^2(y) = \beta_1 y + \beta_2$$
   Tiles are audited with Benjamini-Hochberg False Discovery Rate (BH-FDR, $q=0.05$) control to detect true anomalies without requiring ground truth.
3. **Budget-Constrained Adaptive Neural Refinement**: Routes only the top high-variance candidate tiles (5% to 10% budget) to a compact 4-level deep U-Net ($B_1$, 357k parameters, 1.43 MB).
4. **Raised-Cosine Feather Blending**: Seamlessly blends neural tiles back into the classical base with zero visible grid seams ($C^1$ continuity).
5. **Guided Color Fusion & Acutance**: Separates deep neural structural denoising in luminance from camera-calibrated chrominance, restoring vivid natural colors and applying an auto-optimized 85% photographic edge acutance filter.

---

## 📊 Benchmark Results (Held-Out Test Set)

Evaluated under the pre-registered protocol on **14 paired full-frame 12 MP images** (Scenes 009 & 010; Apple iPhone and Google Pixel sensors) with **10 interleaved round-robin repetitions** on Apple Silicon GPU:

| Configuration | Pipeline Mode | Scene PSNR (dB) | Scene SSIM | Latency (s) | Conv2d GMACs | Compute Reduction vs $B_1$ |
|---|---|---|---|---|---|---|
| **$B_0$ Classical** | CPU / NumPy Baseline | 26.08 dB | 0.5721 | 1.387 s | 0.00 GMACs | **100.0%** |
| **$B_1$ Learned** | Full-Frame Deep U-Net | 29.35 dB | 0.8873 | 0.578 s | 383.53 GMACs | Reference (0%) |
| **$B_2$ PRISM ($\alpha = 0.05$)** | **Hybrid Smart Camera (5% Budget)** | **26.30 dB** | **0.5788** | **1.598 s** | **13.07 GMACs** | **96.6% Reduction** |
| **$B_2$ PRISM ($\alpha = 0.10$)** | **Hybrid Smart Camera (10% Budget)** | **26.43 dB** | **0.5848** | **1.640 s** | **24.59 GMACs** | **93.6% Reduction** |

> **Key Finding**: At a 5% to 10% compute budget, PRISM delivers a **93.6% to 96.6% reduction in deep learning compute** over full-frame neural inference while consistently surpassing the classical camera ISP in quality.

---

## 🏛️ System Architecture

![PRISM System Architecture](docs/architecture.png)

---

## 🚀 Quick Start & Interactive Demo

The project includes an interactive web application with:
- **Split-slider** & **side-by-side** comparison viewers.
- **3-Way Comparison view**: Non-Processed Raw Noisy Sensor Capture vs. Standard Camera ($B_0$) vs. PRISM Smart Camera ($B_2$).
- Real-time **AI Focus Heatmap** and tile routing overlay.
- Automatic **85% High Acutance & Sharpness** enhancement.

### Running the Application:
```bash
# Clone the repository
git clone https://github.com/Hootsworth/PRISM.git
cd PRISM

# Run the single-command demo launcher
./run_demo.sh
```
Open **`http://localhost:8000`** in your browser.

### Reproducing the Comparative Evaluation:
```bash
python3 scripts/evaluate_comparative.py --device mps --budget 0.05
```

---

## 📁 Repository Structure

```text
PRISM/
├── docs/                                  # Full Documentation
│   ├── design_document.md                 # 13-Section Design Document
│   ├── architecture.md                    # Architectural Flow & Mermaid Chart
│   ├── architecture.png                   # High-Res Architecture Diagram
│   ├── ablation_study.md                  # Comprehensive Ablation Studies
│   ├── failure_analysis.md                # Failure Modes, Blind Spots & Ethics
│   └── 00_P2_REQUIREMENTS_TRACEABILITY.md # Traceability Matrix
├── src/prism/                             # Python Core Library
│   ├── isp/                               # Bayer Demosaicing, WB, CCM, Tone Mapping
│   ├── models/                            # Packed-Bayer Deep U-Net Refiner
│   ├── audit/                             # Re-mosaicing, Pearson Residuals & FDR
│   ├── scheduler/                         # Budget Allocator & Priority Queue
│   ├── hybrid/                            # B2 Hybrid Pipeline & Feather Blending
│   └── refinement/                        # Local Neural Tile Processing
├── demo/                                  # Full-Stack Web Application
│   ├── backend/api.py                     # FastAPI REST API & Guided Fusion Engine
│   └── frontend/                          # React + TypeScript + Vite UI
├── experiments/                           # Evaluation Artifacts & Checkpoints
├── paper/                                 # LaTeX Manuscript Source Files
├── PRISM_Research_Paper.pdf               # Compiled 6-Page IEEE/CVPR Paper
├── eval_P2_T37.md                         # Comparative Evaluation Benchmark File
├── PROJECT_LEDGER.md                      # Audit Trail of All Experimental Runs
├── run_demo.sh                            # One-Click Launch Script
└── pyproject.toml                         # Packaging & Environment Specification
```

---

## ⚖️ License & Attribution
- Core source code is released under the **MIT License**.
- Dataset: Smartphone Image Denoising Dataset Calibration & Correction (**SIDD-CC**, IEEE CVPR).
