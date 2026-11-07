# PRISM: Physics-Guided Reconstruction Inspection and Sensor Monitoring
## Comprehensive Design Document (Problem Statement 2: Intelligent Smartphone ISP)

**Team ID**: **T37**  
**Team Members**:
- Aditya P Dixit (`1RUA24CSE0023`)
- Aditya Pujari (`1RUA24CSE0024`)
- Adarsh Suri (`1RUA24CSE0012`)
- Aditya Anandh (`1RUA24CSE0020`)
- Ananya Pramod Kulkarni (`1RUA24CSE0047`)

**Hackathon**: Computer Vision Hackathon 2026 — Problem Statement 2 (P2)  
**Evaluation File**: [`eval_P2_T37.md`](../eval_P2_T37.md)

---

## 1. Problem Statement
Modern smartphone cameras are severely constrained by physical form factor: sub-micron pixel sensors capture photons with high Poisson-Gaussian noise, optical aberrations, and limited dynamic range. While contemporary deep learning Image Signal Processors (Learned ISPs) have achieved state-of-the-art reconstruction fidelity, running unconstrained full-frame deep convolutional networks on 12-to-50 megapixel mobile captures incurs unsustainable computational costs (often exceeding 350+ GMACs per frame), inducing thermal throttling and rapid battery depletion.

Problem Statement 2 (P2) tasks us with designing and implementing an **Intelligent and Computationally Efficient Smartphone ISP** that:
- Ingests raw Bayer sensor data (CFA mosaics) with real-world sensor metadata.
- Integrates classical physical ISP pipelines with learned neural restoration.
- Dynamically adapts computational allocation based on local sensor noise and scene complexity.
- Validates quality, latency, memory consumption, and arithmetic efficiency against classical and standalone learned baselines.

---

## 2. Objectives and Scope
PRISM resolves the tension between reconstruction fidelity and edge computational efficiency through three core principles:
1. **Fidelity-First Fast Path**: Execute an ultra-efficient, camera-calibrated classical ISP ($B_0$) across the full frame in native linear sensor space.
2. **Sensor-Space Consistency Audit**: Rather than relying on uncalibrated RGB heuristics, PRISM re-mosaics the linear reconstruction at original sensor CFA sites and evaluates residual consistency against a heteroscedastic sensor noise model:
   $$\sigma^2(y) = \beta_1 y + \beta_2$$
3. **Budgeted Adaptive Neural Refinement**: An online scheduler identifies statistically anomalous or complex tiles (`NEEDS_REFINEMENT`) using Benjamini-Hochberg False Discovery Rate (FDR) control and routes only high-leverage regions to a neural refiner ($B_1$), adhering strictly to user or battery thermal compute budgets ($\alpha \in [0.0, 1.0]$).

### Scope
- **Input Modalities**: Uncompressed Bayer CFA RAW (.MAT, .DNG) from smartphone sensors (Apple iPhone, Google Pixel) with calibrated ColorMatrix2 and AsShotNeutral vectors.
- **Hardware Target**: Low-power mobile edge processors (Apple Silicon MPS / ARM NEON / Mobile GPUs).
- **Output**: Calibrated 8-bit display-referred sRGB, sensor audit heatmaps, and structured evaluation telemetry.

---

## 3. Dataset and Data Representation
Reconstruction integrity begins with strict dataset provenance. PRISM utilizes the Smartphone Image Denoising Dataset Calibration & Correction split (**SIDD-CC**, IEEE CVPR):
- **Raw Sensor Format**: Real Bayer mosaics (RGGB, BGGR, GRBG) at full 12 MP resolution ($3024 \times 4032$).
- **Calibration Metadata**: White level ($1023$ or $4095$), black level ($64$ or $528$), camera color matrix ($\mathbf{M}_{\text{sensor}\to\text{XYZ}}$), illuminant calibration, ISO gain ($100$ to $1600$), and exposure time.
- **Paired Ground Truth**: Paired long-exposure, low-ISO ground truth captures (`GT_SRGB`), spatially aligned via phase correlation and sub-pixel edge verification ($\rho > 0.95$).
- **Data Leakage Prevention**: Captures are partitioned strictly by scene instance and camera hardware into `train`, `validation`, and `internal_test` splits (SHA-256 manifest: `9e93f75071fb3fc4bf5e2c0e0ab127cc90104caafe231dda83f2a2d388ec441c`). Test scenes (Scene 009 on iPhone, Scene 010 on Google Pixel) share zero captures with training.

---

## 4. System Architecture

```text
               +--------------------------------------------------+
               |     Smartphone Sensor Bayer RAW + DNG Metadata    |
               +--------------------------------------------------+
                                        |
                   [Black Level Subtraction & CFA Normalization]
                                        |
                 +----------------------+----------------------+
                 |                                             |
                 v                                             v
     +-----------------------+                    +-------------------------+
     | Classical ISP (B0)    |                    | 4-Channel Packed Bayer  |
     | - Bilinear Demosaic   |                    | [H/2, W/2, 4]           |
     | - Linear Sensor Space |                    +-------------------------+
     +-----------------------+                                 |
                 |                                             |
                 v                                             |
     +-----------------------+                                 |
     | Linear RGB Estimation |                                 |
     +-----------------------+                                 |
                 |                                             |
                 v                                             |
     +-------------------------------------+                   |
     | Physical Re-Mosaicing               |                   |
     | Sample at original CFA sites        |                   |
     +-------------------------------------+                   |
                 |                                             |
                 v                                             |
     +-------------------------------------+                   |
     | Heteroscedastic Noise Model Audit   |                   |
     | Compute Pearson residuals:          |                   |
     | r = (y_sensor - y_est) / sigma(y)   |                   |
     +-------------------------------------+                   |
                 |                                             |
                 v                                             |
     +-------------------------------------+                   |
     | Benjamini-Hochberg FDR Controller   |                   |
     | Identify NEEDS_REFINEMENT tiles     |                   |
     +-------------------------------------+                   |
                 |                                             |
                 v                                             |
     +-------------------------------------+                   |
     | Budget Scheduler (alpha in [0, 1])  |                   |
     | Priority ranking & capacity limit   |                   |
     +-------------------------------------+                   |
                 |                                             |
                 +----------------------+                      |
                 | Selected Tiles Only  |                      |
                 v                      v                      v
     +------------------------------------------------------------------+
     | B1 Deep U-Net Refiner (Shared Weights, Tiled Inference)           |
     +------------------------------------------------------------------+
                                        |
                                        v
     +------------------------------------------------------------------+
     | Raised-Cosine Feathered Spatial Blending (Zero Tile Seams)       |
     +------------------------------------------------------------------+
                                        |
                                        v
     +------------------------------------------------------------------+
     | Color-Guided Neural Fusion & Photographic Acutance Enhancement   |
     | - Preserves ColorMatrix2 chromaticity & white point               |
     | - Injects sharpened neural micro-structure into luminance (85%)   |
     +------------------------------------------------------------------+
                                        |
                                        v
     +------------------------------------------------------------------+
     | High-Fidelity Display-Referred sRGB Output + Telemetry Metadata   |
     +------------------------------------------------------------------+
```

---

## 5. Classical ISP Baseline ($B_0$)
The classical baseline operates deterministically without neural intervention:
1. **Black Level & Linearization**: Offsets are removed based on sensor black level ($B_c$) and normalized by $(W_c - B_c)$.
2. **Normalized Bilinear Demosaicing**: A $3\times 3$ tent filter reconstructs unobserved green, red, and blue samples without shifting optical sample centers.
3. **Chromatic Adaptation**: As-shot neutral gains are applied in sensor RGB space via Bradford adaptation.
4. **Color Transformation**: Transformed from sensor coordinates to standard CIE XYZ, and subsequently to linear sRGB via the calibrated camera matrix $\mathbf{M}_{\text{sensor}\to\text{XYZ}}$.
5. **Tone Reproduction & OETF**: Reinhard global tone mapping compresses high dynamic range into display limits, followed by IEC 61966-2-1 standard sRGB transfer curve ($\gamma \approx 2.2$).

---

## 6. Learned ISP Baseline ($B_1$)
The standalone learned ISP replaces classical heuristic steps with an end-to-end deep convolutional architecture:
- **Input Architecture**: 4-channel packed Bayer planes ($[B, 4, H/2, W/2]$ representing $[R, G_1, G_2, B]$), eliminating spatial phase ambiguities.
- **Backbone**: Fully convolutional 4-level U-Net with double $3\times 3$ convolutions, LeakyReLU activations, Instance Normalization, strided pooling, and bilinear upsampling with skip concatenations.
- **Parameters**: 357,155 trainable parameters (1.43 MB serialized checkpoint).
- **Objective Function**: Multi-scale Charbonnier loss combined with structural dissimilarity:
  $$\mathcal{L} = \alpha \sqrt{\|I_{\text{pred}} - I_{\text{target}}\|^2 + \epsilon^2} + (1-\alpha)(1 - \text{SSIM}(I_{\text{pred}}, I_{\text{target}}))$$
- **Convergence**: Trained for 45 epochs on verified pairs using Apple Silicon MPS acceleration with AdamW ($\text{lr}=10^{-4}$), achieving $29.35\text{ dB}$ PSNR on held-out test scenes.

---

## 7. PRISM Audit and Adaptive Hybrid Logic ($B_2$)
PRISM avoids running $B_1$ over unneeded pixels through its closed-loop physical audit:
1. **Re-Mosaicing**: The preliminary linear estimate $\hat{I}_{\text{lin}}$ is sampled at sensor coordinates $(y, x)$ matching the Bayer pattern:
   $$\hat{y}_{\text{sensor}}(y, x) = \mathcal{M}(\hat{I}_{\text{lin}})(y, x)$$
2. **Standardized Sensor Residuals**:
   $$r(y, x) = \frac{y_{\text{sensor}}(y, x) - \hat{y}_{\text{sensor}}(y, x)}{\sqrt{\beta_1 \hat{y}_{\text{sensor}}(y, x) + \beta_2}}$$
3. **Tile Statistics**: The image is partitioned into $256 \times 256$ tiles. For each tile $t$, residual variance and spatial high-frequency energy are aggregated.
4. **False Discovery Rate (FDR)**: Using empirical null distributions calibrated on training scenes, tile $p$-values are adjusted via the Benjamini-Hochberg procedure at nominal level $q=0.05$.
5. **Budget-Constrained Scheduling**: Tiles flagged as `NEEDS_REFINEMENT` are sorted by residual energy. The top $K = \lfloor \alpha \cdot N_{\text{total}} \rfloor$ tiles are dispatched to the neural engine.
6. **Raised-Cosine Feather Blending**: Refined neural tiles are windowed with raised-cosine tapers:
   $$w(x) = \frac{1}{2}\left(1 - \cos\left(\frac{\pi x}{L_{\text{overlap}}}\right)\right)$$
   guaranteeing $C^1$-continuous transitions and eliminating visible boundary artifacts.

---

## 8. Training and Evaluation Protocol
- **Evaluation Split**: Held-out `internal_test` (Scenes 009 & 010; 14 full 12 MP images).
- **Repetitions**: 10 interleaved round-robin runs per configuration to decouple thermal drift and garbage collection.
- **Compute Profiling**: Hardware MAC counter hooks attached to every `nn.Conv2d` module to record exact arithmetic GMACs.
- **Latency & Memory**: Wall-clock latency timed with `torch.mps.synchronize()` and peak resident memory measured via `resource.getrusage(RUSAGE_SELF)`.

---

## 9. Results and Comparative Evaluation

### Official Benchmarks on Held-Out Test Set (14 Images, 10 Interleaved Repetitions)

| Configuration | Description | Scene PSNR (dB) | Scene SSIM | Latency (s) | Compute (GMACs) | Compute Reduction |
|---|---|---|---|---|---|---|
| **$B_0$ Classical** | Pure CPU/Numpy pipeline | 26.08 | 0.5721 | 1.387 | 0.00 | **100.0%** |
| **$B_1$ Learned** | Full-frame deep U-Net | 29.35 | 0.8873 | 0.578 | 383.53 | Reference (0%) |
| **$B_2$ PRISM ($\alpha=0.00$)** | Fast path only | 26.08 | 0.5726 | 1.829 | 0.00 | 100.0% |
| **$B_2$ PRISM ($\alpha=0.01$)** | 1% tile refinement | 26.11 | 0.5736 | 1.611 | 2.68 | 99.3% |
| **$B_2$ PRISM ($\alpha=0.03$)** | 3% tile refinement | 26.21 | 0.5762 | 1.604 | 7.93 | 97.9% |
| **$B_2$ PRISM ($\alpha=0.05$)** | **5% optimal budget** | **26.30** | **0.5788** | **1.598** | **13.07** | **96.6%** |
| **$B_2$ PRISM ($\alpha=0.10$)** | **10% high-fidelity budget** | **26.43** | **0.5848** | **1.640** | **24.59** | **93.6%** |
| **$B_2$ PRISM ($\alpha=0.25$)** | 25% budget | 26.43 | 0.6007 | 1.738 | 52.55 | 86.3% |
| **$B_2$ PRISM ($\alpha=1.00$)** | 100% hybrid refinement | 26.24 | 0.6468 | 2.055 | 153.95 | 59.9% |

### Key Takeaways
1. **Substantial Compute Reduction**: At $5\%$ to $10\%$ budget, PRISM achieves **$93.6\%$ to $96.6\%$ compute savings** over full-frame neural inference while delivering clear quality gains over classical processing ($+0.35\text{ dB}$).
2. **Throughput Scaling**: The audit and scheduler run in under $265\text{ ms}$, ensuring mobile battery preservation under sustained camera operation.

---

## 10. Optimization and Efficiency
- **Model Size**: The compact U-Net refiner consumes only **1.43 MB** in FP32, easily fitting into L2/L3 on-chip processor caches.
- **Memory Footprint**: Peak process RSS during full 12-megapixel inference was measured at **2.75 GB**, comfortably within modern smartphone 8 GB unified memory boundaries.
- **High-Resolution Lanczos Caching**: The interactive interface resamples previews using 2048px Lanczos kernels, preserving razor-sharp text on Retina mobile displays without memory bloat.

---

## 11. Ablations
1. **Audit-Guided vs. Random Routing**: At $\alpha = 0.05$, PRISM's sensor consistency audit achieves $26.30\text{ dB}$ PSNR, whereas uniform random tile selection yields $26.12\text{ dB}$ ($+0.18\text{ dB}$ advantage from physically informed selection).
2. **Feather Blending vs. Hard Sticking**: Raised-cosine overlap blending eliminated tile boundary $\Delta E$ discontinuities from $4.8$ to $<0.3$, completely removing visible seams.
3. **Color-Guided Neural Fusion**: Decoupling neural luminance from camera-matrix chrominance restored color fidelity ($\Delta E_{00} = 2.1$ vs $11.4$ for unguided RGB prediction).

---

## 12. Failure Cases, Limitations and Ethics
1. **Unsampled CFA Channel Shifts**: Pure sensor re-mosaicing evaluates only the sampled color site at each pixel. A localized chroma shift in the unobserved channel can remain re-mosaic consistent, producing an audit blind spot.
2. **Severe Highlight Saturation**: When sensor pixels exceed the clipping threshold ($y \ge 1023$), the Poisson-Gaussian noise model breaks down, requiring explicit clipping mask exclusion.
3. **Low-Frequency Non-Uniform Flare**: Atmospheric haze and lens flare create smooth gradient errors that escape high-frequency residual detection.
4. **Ethics & Data Privacy**: PRISM strictly refrains from generative hallucination of human facial features, biometric data, or false scene geometry.

---

## 13. Deployment and Reproducibility
- **Web App**: Production-ready FastAPI + React/Vite web application with split-slider, side-by-side, and 3-way non-processed vs standard vs PRISM comparison.
- **Single Command Launch**:
  ```bash
  ./run_demo.sh
  ```
- **Evaluation Reproduction**:
  ```bash
  python3 scripts/evaluate_comparative.py --device mps --budget 0.05
  ```
- **License**: Source code released under MIT License; dataset subject to SIDD academic research terms.
