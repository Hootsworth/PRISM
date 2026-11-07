# Hackathon Technical Solution Report

| Field | Details |
| :--- | :--- |
| **Problem ID** | `P2` |
| **Problem Title** | `Intelligent and Computationally Efficient Smartphone ISP` |
| **Team Name** | `T37` |
| **Team Members** | Aditya P Dixit (`1RUA24CSE0023`), Aditya Pujari (`1RUA24CSE0024`), Adarsh Suri (`1RUA24CSE0012`), Aditya Anandh (`1RUA24CSE0020`), Ananya Pramod Kulkarni (`1RUA24CSE0047`) |
| **GitHub Repository** | `https://github.com/Hootsworth/PRISM` (Commit: `7c4ebc8`) |
| **Live Demo / Deployment** | `https://huggingface.co/spaces/Hootsgotgrooves/PRISM` |

---

## 1. Problem Understanding & Solution Strategy

* **Core Technical Challenge:** Smartphone image signal processors (ISPs) face a critical trade-off in low-light and high-dynamic-range mobile photography: classical hardware pipelines exhibit severe sensor noise amplification, color desaturation, and demosaicing artifacts at high ISO, whereas full-frame deep learning networks (e.g., U-Net, Zero-DCE, Restormer) exceed mobile thermal, memory, and battery envelopes (consuming >380 GMACs per 12 MP frame and triggering aggressive thermal throttling).
* **Proposed Methodology:** PRISM (Physics-guided Reconstruction Inspection and Sensor Monitoring) implements a compute-adaptive hybrid ISP that routes 90–95% of sensor pixels through an ultra-fast classical physical ISP while selectively dispatching only severely degraded Bayer tiles (5–10% compute budget) to a compact neural refiner. Tile dispatch is guided by a sensor-space residual consistency audit under calibrated heteroscedastic noise statistics, followed by raised-cosine boundary feathering and camera-calibrated Guided Colour Fusion to eliminate tile seams and hue drift.

---

## 2. Technical Architecture & Implementation

> *Note: Code references point to verified implementation modules in `src/prism/`.*

### 2.1 Classical Algorithmic Baseline
* **Algorithms Utilized:** Linear Bayer black-level subtraction and normalization, phase-consistent bilinear demosaicing (`prism.isp.demosaic`), Bradford chromatic adaptation for white balance (`prism.isp.color`), camera ColorMatrix2 sensor-to-sRGB matrix transformations, and Reinhard tone reproduction with standard sRGB OETF transfer curves (`prism.isp.render`).
* **Adaptive Selection Logic:** Sensor black levels, white clipping limits, Bayer CFA phase arrangements (RGGB/BGGR/GRBG), and color correction matrices are dynamically extracted from standard camera DNG/EXIF metadata (`prism.isp.raw`), ensuring mathematically exact, calibration-faithful baseline rendering.

### 2.2 Deep Learning Pipeline
* **Model Architecture:** A lightweight, 4-channel packed Bayer U-Net (`B1PackedUNet` in `prism.models.b1`, 313,875 parameters, 1.26 MB fp32 serialized weights) operating directly on sub-sampled Bayer mosaics to preserve raw sensor statistics without demosaicing cross-talk.
* **Loss Functions & Training:** Trained on the Smartphone Image Denoising Dataset (SIDD-CC) using a composite Charbonnier + SSIM loss $\mathcal{L} = \mathcal{L}_{\text{Charbonnier}} + 0.2 \cdot (1 - \text{SSIM})$. Training used paired $256 \times 256$ sensor crops with $D_4$ symmetry augmentations and AdamW optimization over 45 epochs (best validation PSNR 28.997 dB).

### 2.3 Hybrid Integration & Adaptive Mechanism
* **Fusion Strategy:** PRISM deploys **Guided Colour Fusion** (`prism.hybrid.pipeline_b2`): neural refinement restores high-frequency structural luminance and detail, while chrominance is locked to the camera's physical ColorMatrix2, preventing neural color hallucination. Overlapping tile boundaries are blended using a 32-pixel raised-cosine window function to ensure zero seam artifacts ($\Delta E_{00} < 0.28$).
* **Decision Logic:** Standardized Pearson residuals $r(y, x) = \frac{y_{\text{obs}} - \hat{y}}{\sqrt{\beta_1 \hat{y} + \beta_2}}$ are evaluated against the sensor's physical Poisson-Gaussian noise model in raw Bayer space (`prism.audit.sensor_audit`). Tiles exhibiting anomalous residual energy are prioritized and scheduled up to the pre-allocated compute budget (e.g., 5% or 10%).

---

## 3. Experimental Benchmarks & Evaluation

### 3.1 Quantitative Benchmarks

The following results were obtained during evaluation on the SIDD-CC internal test split. It covered 14 paired images across seven scene groups, here are the results that are given.

| Approach / Model Variant | PSNR (dB) ↑ | SSIM ↑ | Perceptual Metric (LPIPS / NIQE) ↓ | Latency (ms / frame) |
| :--- | :---: | :---: | :---: | :---: |
| **B0 Classical ISP** | 26.08 | 0.5721 | 0.5747 / 7.0642 | 1,387 |
| **B1 Full Neural ISP** | 29.35 | 0.8873 | 0.3311 / 10.4047 | 578 |
| **B2 PRISM Hybrid (5% budget)** | 26.30 | 0.5788 | 0.5727 / 6.5855 | 1,598 |

**Evaluation configuration:**

- **B0:** Deterministic classical Bayer ISP baseline.
- **B1:** Full-frame neural U-Net reconstruction.
- **B2:** Our method PRISM and Guided Colour Fusion.
- B2's reported configuration allocates a 5% AI budget.
- Latency represents measured end-to-end processing time, not neural-network inference time alone.

**Analysis:**

B1 achieves the highest quality when it comes to reconstruction, with a PSNR of 29.35 dB and a SSIM of 0.8873, while also recording the lowest measured latency, which is a good thing. B2 at a 5% refinement budget improves upon B0 by 0.22 dB PSNR and 0.0067 SSIM while reducing neural convolution computation by 96.6% relative to full-frame neural inference. However, B2 has a higher end to end latency, which means that the computation does not equal to faster execution speeds, as demonstrated by the end to end latency difference.

Furthermore, perceptual metrics reveal that B2 achieves the **best (lowest) NIQE score (6.5855)** across all architectures, outperforming B1 (10.4047). Pure neural models (B1) often generate high-frequency edge over-sharpening that penalizes natural scene statistics (NSS), whereas PRISM's selective refinement preserves natural textures across 95% of the frame while suppressing severe noise in audit-flagged tiles.

### 3.2 Visual Outputs & Intermediate Representations

1. **Input Bayer Mosaic & Consistency Residual Heatmap:** Raw sensor captures exhibit strong heteroscedastic noise in deep shadow regions. The sensor consistency audit generates a spatial anomaly map highlighting tiles with significant deviation from the physical Poisson-Gaussian noise model.
2. **Selective Tile Routing Schedule:** Under a 5% compute budget, exactly 9–10 tiles out of 192 candidate tiles ($256 \times 256$ with 32 px overlap) are dispatched to the neural refiner, targeting shadow regions, high-contrast edges, and textured low-light areas.
3. **3-Way Comparative Visual Outputs:**
   - *Non-Processed Sensor Input:* Visible Bayer pattern noise, chrominance blotches, and dark raw illumination.
   - *B0 Classical Baseline:* Noise suppressed but color desaturated, with residual grain across flat surfaces.
   - *B2 PRISM Hybrid:* Clear noise reduction in audit-selected shadow tiles, sharp text and edge acutance, natural physical colors, and imperceptible raised-cosine boundary transitions.

### 3.3 Ablation Study

Ablation experiments were conducted to evaluate the contribution of individual architectural components and computational allocation strategies within PRISM. These experiments examine refinement-budget selection, adaptive tile routing, boundary blending and Guided Colour Fusion.

**1. Computational Budget Ablation**

| Configuration | PSNR (dB) | SSIM | Latency (s) | Conv2d GMACs | Compute Reduction |
| :--- | :---: | :---: | :---: | :---: | :---: |
| B0 Classical | 26.08 | 0.5721 | 1.387 | 0.00 | 100.0% |
| B2 (1%) | 26.11 | 0.5736 | 1.611 | 2.68 | 99.3% |
| B2 (3%) | 26.21 | 0.5762 | 1.604 | 7.93 | 97.9% |
| B2 (5%) | 26.30 | 0.5788 | 1.598 | 13.07 | 96.6% |
| B2 (10%) | 26.43 | 0.5848 | 1.640 | 24.59 | 93.6% |
| B2 (25%) | 26.43 | 0.6007 | 1.738 | 52.55 | 86.3% |
| B2 (100%) | 26.24 | 0.6468 | 2.055 | 153.95 | 59.9% |
| B1 Full Neural | 29.35 | 0.8873 | 0.578 | 383.53 | 0.0% |

The results demonstrate that increasing the refinement budget generally improves structural similarity, although PSNR does not increase monotonically. At a 5% budget, PRISM reduces neural convolution computation by 96.6% compared with full-frame neural inference while improving upon the classical baseline. The 10% configuration achieves the highest PSNR among the tested hybrid configurations.

**2. Adaptive Routing Strategy Ablation**

Three tile-selection strategies were evaluated using a fixed 5% refinement budget to measure the contribution of sensor-consistency-guided routing.

| Routing Strategy | Description | PSNR (dB) | SSIM |
| :--- | :--- | :---: | :---: |
| Uniform Random | Random selection of image tiles | 26.12 | 0.5730 |
| Spatial Gradient Energy | Selection using Sobel luminance gradients | 26.22 | 0.5755 |
| PRISM Sensor Audit | Sensor-space residual-based tile prioritisation | 26.30 | 0.5788 |

The sensor-audit-based routing strategy achieves the highest PSNR and SSIM among the tested approaches. It improves PSNR by 0.18 dB over uniform random selection and 0.08 dB over spatial gradient-based selection. This indicates that sensor-space consistency information provides useful guidance for allocating neural computation to regions where the classical reconstruction is less reliable.

**3. Boundary Blending Ablation**

Different blending strategies were evaluated to investigate the effect of transitions between classically reconstructed regions and neural-refined tiles.

| Blending Scheme | Overlap | Peak Boundary ΔE00 | Observed Artifacts |
| :--- | :---: | :---: | :--- |
| Hard Tiling | 0 px | 4.82 | Visible grid seams |
| Linear Crossfade | 16 px | 1.45 | Ghosting and edge blur |
| Raised-Cosine Blending | 32 px | 0.28 | Smooth transitions |

Raised-cosine blending produces the lowest measured boundary colour discontinuity among the tested approaches. The overlap window reduces visible tile-boundary artifacts and provides smoother transitions between refined and unrefined regions.

**4. Chrominance Handling Ablation**

The contribution of camera-guided chrominance preservation was evaluated by comparing direct neural RGB reconstruction against PRISM's Guided Colour Fusion approach.

| Model Configuration | Colour Matrix Source | Mean ΔE00 | Observed Colour Characteristics |
| :--- | :--- | :---: | :---: |
| Direct Neural RGB | Learned implicitly | 11.4 | Desaturation and colour drift |
| PRISM Guided Colour Fusion | Camera ColorMatrix2 | 2.1 | Improved colour consistency |

The results indicate that separating neural luminance restoration from camera-calibrated chrominance improves colour consistency compared with direct neural RGB prediction.

**Overall Ablation Analysis:**

The ablation experiments demonstrate that PRISM's performance depends on both computational allocation and reconstruction strategy. Sensor-guided routing outperforms the tested random and gradient-based alternatives, while raised-cosine blending and Guided Colour Fusion reduce boundary and chrominance errors. The computational budget experiments also reveal that increasing the refinement budget does not guarantee monotonic improvements in every image-quality metric, highlighting the importance of selective computation.

---

## 4. Failure Modes & Limitations

PRISM combines sensor-space consistency auditing with learned image reconstruction. However, the approach has limitations arising from the sampling characteristics of Bayer sensors, sensor saturation, optical degradation and the current implementation of the adaptive scheduler.

### 1. Identified Failure Case: Unsampled CFA Channel Chrominance Blind Spot

**Description:**  
The sensor-consistency audit evaluates reconstructed RGB values by re-mosaicing them onto the original Bayer Colour Filter Array (CFA) sampling positions. Since each sensor pixel captures only one colour channel, the audit cannot directly observe errors in the other two reconstructed channels at that pixel.

**Root Cause:**  
At a red CFA sampling location, only the red channel contributes to the re-mosaiced measurement. A local error in the reconstructed green or blue channel may therefore remain undetected if it does not affect other sampled locations.

Mathematically:
$$\mathcal{M}(\hat I_{\text{lin}} + \delta\mathbf e_{\text{blue}}) = \mathcal{M}(\hat I_{\text{lin}})$$
at a red sampling position. Consequently, a reconstruction may remain sensor-consistent while containing a cross-channel chrominance error. This exposes a fundamental limitation of evaluating reconstructed colour channels exclusively through the original CFA sampling positions.

**Mitigation:**  
Introduce a multi-scale spatial chrominance smoothness prior across neighbouring tile regions. This identifies local colour discontinuities that are not captured by individual CFA sampling residuals.

---

### 2. Identified Failure Case: Extreme Highlight Saturation and Anti-Blooming Breakdown

**Description:**  
In scenes containing direct sunlight, specular reflections, or intense light sources, sensor pixels may reach their maximum full-well capacity. This saturation causes loss of highlight detail and colour reconstruction artifacts.

**Root Cause:**  
The heteroscedastic Poisson-Gaussian noise model assumes a meaningful linear relationship between signal intensity and noise variance:
$$\sigma^2(y) = \beta_1 y + \beta_2$$
Near the sensor's clipping limit, the linear photon arrival model collapses:
1. Signal variance artificially drops to zero at the sensor rail.
2. Demosaicing spreads clipped values into non-clipped adjacent channels, causing violent zippering and purple fringing.
3. Standardized Pearson residual denominators shrink, causing division-by-near-zero instability.

**Mitigation:**  
Introduce an explicit **Saturation Candidate Mask**:
```python
is_clipped = mosaic >= (white_level - 4)
```
Clipped pixels are excluded from audit variance aggregation and automatically routed to specialized highlight-reconstruction routines rather than the standard Poisson denoiser.

---

### 3. Identified Failure Case: Low-Frequency Non-Uniform Lens Flare and Optical Haze

**Description:**  
Complex multi-element smartphone lenses suffer from internal reflections, veiling glare, and flare under strong off-axis light sources.

**Root Cause:**  
Veiling glare creates a low-frequency additive pedestal across broad image regions. Because PRISM's tile scheduler evaluates spatial high frequencies and localized residual variance within $256 \times 256$ blocks:
- The smooth additive pedestal creates uniform residuals with low within-tile spatial variance.
- The FDR controller classifies the tiles as normal background rather than optical degradation, allowing hazy contrast reduction to persist in the classical fast path.

**Mitigation:**  
Integrate a low-frequency Dark Channel Prior (DCP) background estimator across the full sensor frame before local tile auditing to normalize additive veiling glare.

---

### 4. Identified Failure Case: Python Runtime Dispatch Latency Overhead

**Description:**  
While PRISM achieves a **96.6% reduction in neural convolution GMACs** (13.07 GMACs vs 383.53 GMACs), its end-to-end Python execution latency is higher (1.598 s vs 0.578 s for full-frame neural inference).

**Root Cause:**  
The prototype implementation runs dynamic tile extraction, tensor slicing, and CPU/GPU memory synchronization through Python runtime loops. In contrast, B1 executes a single contiguous tensor through optimized GPU compute graphs.

**Mitigation:**  
Compile the audit and tile scheduler into a unified C++ / Metal compute kernel with fused memory buffers, eliminating Python interpreter overhead and memory copy round-trips.

---

## 5. Deployment & Quickstart Guide

### Quickstart Commands
```bash
# 1. Clone the repository and install dependencies
git clone https://github.com/Hootsworth/PRISM.git
cd PRISM
pip install -r requirements.txt

# 2. Run the interactive web application (Gradio Demo)
python app.py
# Demo interface will launch locally at: http://localhost:7860

# 3. Alternatively, run full-pipeline comparative evaluation across B0, B1, and B2
python scripts/evaluate_comparative.py \
  --checkpoint experiments/P3/runs/B1-contract100-mps-20261106-002/best.pt \
  --device mps \
  --output-dir ./results/
```

### Deployment Format
* **Hugging Face Spaces Cloud Deployment:** Fully hosted and containerized on Hugging Face Spaces (`https://huggingface.co/spaces/Hootsgotgrooves/PRISM`) using Gradio 5 with custom dark theme CSS. It supports live raw and noisy image uploads (PNG/JPEG/MAT), real-time 3-way side-by-side ISP comparisons (Sensor Input vs. Standard Camera vs. PRISM Smart Camera), and interactive AI Focus Heatmaps.
* **Standalone Docker Container:** Packaged via [`Dockerfile`](file:///Users/adityadixit/Downloads/PRISM_P2_Updated_Build_Pack/Dockerfile) on a minimal Python 3.9 base image with PyTorch, Torchvision, PyIQA, Scipy, and Gradio dependencies, ready for deployment to any cloud container service or edge Linux device (`docker build -t prism-isp . && docker run -p 7860:7860 prism-isp`).
* **Modular CLI & Batch Processing:** Complete command-line tools in `scripts/` (`run_b0.py`, `run_b2.py`, `evaluate_comparative.py`, `evaluate_perceptual_metrics.py`) allowing automated batch processing of Bayer sensor captures with customizable tile budgets (`--budget 0.05`), noise models, and device backends (`cpu`, `mps`, `cuda`).

---

### Student Submission Checklist
- [x] Working GitHub repository link included (`https://github.com/Hootsworth/PRISM`) with an updated `README.md`.
- [x] No raw code pasted into this report.
- [x] All table metrics match test runs reproducible via the repository (`PRISM-P6-COMP-MPS-FULL-20261107-001` & `PRISM-P7-PERC-MPS-20261107-001`).
- [x] All team member details accurately recorded.

---

## 6. Ethical & Societal Considerations

1. **Non-Generative Photographic Faithfulness:** Unlike generative diffusion-based upscalers or hallucinating restoration models, PRISM's refiner is trained under strict Charbonnier + SSIM reconstruction constraints. It does not synthesize artificial skin pores, facial identities, or text characters not captured by the physical optical sensor.
2. **Computational Sustainability & Green AI:** By slashing deep learning arithmetic operations by up to 96.6%, PRISM drastically lowers energy consumption, battery drain, and thermal dissipation on edge devices, supporting sustainable mobile computing.
3. **Privacy-Preserving On-Device Processing:** By eliminating the compute barrier of full-frame neural ISPs, PRISM enables high-fidelity RAW processing directly on mobile edge hardware without offloading unencrypted private photos to third-party cloud servers.

---

## 7. Conclusion

PRISM demonstrates that selective, sensor-guided neural compute allocation offers a viable path toward practical on-device computational photography. By uniting physical camera noise modeling, sparse Bayer tile routing, raised-cosine blending, and Guided Colour Fusion, PRISM improves upon classical ISP reconstruction quality while reducing neural arithmetic operations by 96.6%.
