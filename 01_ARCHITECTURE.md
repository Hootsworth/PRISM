# PRISM Architecture for P2

## 1. Project objective
Build a hybrid, computationally efficient smartphone ISP that accepts supported Bayer RAW/DNG (and, where explicitly supported, paired Raw-RGB data), produces high-quality RGB, adapts to noise and illumination, and compares classical, learned and hybrid approaches.

PRISM's research component is a calibrated sensor-consistency audit used as a runtime compute controller. It is not a replacement for a complete ISP and does not, by itself, prove RGB correctness.

## 2. Three required ISP approaches

### B0 — Classical ISP baseline
A deterministic pipeline:
1. Validate RAW metadata and unpack CFA.
2. Black-level subtraction and white-level normalization.
3. Classical denoising (document algorithm/parameters).
4. Demosaicing.
5. White balance.
6. Colour correction matrix.
7. Tone mapping, gamma/OETF, optional sharpening.
8. Export RGB and intermediate stages.

Use a stable library implementation where appropriate, but document every stage and keep parameters reproducible. Avoid comparing a poorly tuned classical baseline against a heavily trained neural model.

### B1 — Learned ISP baseline
A standalone full-frame neural RAW-to-RGB model. Initial practical design:
- Input: 4-channel packed Bayer planes plus selected metadata/noise/illumination conditioning.
- Output: 3-channel RGB (preferably display-referred sRGB with a clearly specified target transform; alternatively linear RGB plus an explicit fixed renderer).
- Architecture: compact U-Net / encoder-decoder.
- Loss: combination of L1/Charbonnier and optional SSIM/perceptual terms; justify weights and avoid optimizing on the test split.
- Training: only verified paired RAW-to-RGB examples. Store split manifest, preprocessing, target colour space, seed and checkpoint hash.

If the available dataset cannot provide verified Bayer-to-sRGB pairs, do not silently substitute Raw-RGB MAT arrays. Use the approved compatible dataset path documented in `02_DATASET_AND_DATA_CONTRACT.md`, or clearly mark the Bayer learned baseline blocked until data pairing is solved.

### B2 — PRISM hybrid
1. Run B0 fast classical fidelity path in camera-native linear space.
2. Re-mosaic the linear RGB estimate at original CFA sample positions.
3. Compute normalized residual using a calibrated noise model.
4. Produce tile diagnostics and reliability masks.
5. Calibrate empirical nulls on training/calibration captures; apply tile-level FDR on held-out captures.
6. Separate statistical significance from actionability.
7. Route only eligible `NEEDS_REFINEMENT` tiles to the same learned refiner used in the learned baseline where feasible.
8. Blend overlapping tile outputs, re-audit, stop at budget or no eligible tiles.
9. Apply final WB/CCM/tone mapping/gamma/sharpening and export.

For fair comparisons, B1 must also be evaluated as full-frame inference. The hybrid may use the same network weights on selected tiles. If different heads or input channels are required, document the difference and add a controlled equal-capacity comparison.

## 3. End-to-end data flow

```text
Supported RAW/DNG + metadata
       |
       v
Format validation + dataset manifest
       |
       v
Black-level / white-level / CFA preprocessing
       |
       +---------------------------+
       |                           |
       v                           v
Classical ISP B0              Packed Bayer + metadata
       |                           |
       |                           v
       |                      Learned ISP B1
       |                           |
       v                           |
Linear RGB estimate                |
       |                           |
       v                           |
Re-mosaic at sensor CFA sites      |
       |                           |
       v                           |
Calibrated residual audit          |
       |                           |
Empirical null + FDR               |
       |                           |
Tile states + actionability        |
       |                           |
Budget scheduler                  |
       |                           |
Local learned refinement           |
       |                           |
Re-audit + stopping                |
       |                           |
       +------------+--------------+
                    |
                    v
             PRISM hybrid B2
                    |
                    v
        WB / CCM / tone / gamma / sharpen
                    |
                    v
       RGB output + heatmap + run report
                    |
                    v
       Evaluation + public demo + download
```

B0, B1 and B2 are distinct experimental modes. Do not label B1 as a mere tile refiner. It must be a working standalone learned ISP comparison.

## 4. RAW and data contracts

### RawFrame
```python
RawFrame = {
  "mosaic": Tensor,             # H x W, single-plane Bayer where applicable
  "packed_bayer": Tensor,      # 4 x H/2 x W/2 after verified packing
  "cfa_pattern": str | None,   # RGGB/BGGR/GRBG/GBRG only when known
  "black_level": Tensor | float | None,
  "white_level": Tensor | float | None,
  "camera_id": str | None,
  "iso": int | None,
  "exposure": float | None,
  "white_balance": Tensor | None,
  "ccm": Tensor | None,
  "metadata": dict,
  "valid_mask": Tensor
}
```
Do not invent missing metadata. Unknown CFA or ambiguous packing must raise a clear unsupported-input result.

### DataSample
```python
DataSample = {
  "sample_id": str,
  "capture_id": str,
  "scene_id": str,
  "camera_id": str | None,
  "input_path": str,
  "input_representation": "bayer_dng|raw_rgb_mat|other",
  "target_path": str | None,
  "target_representation": "srgb|linear_rgb|raw_rgb|none",
  "alignment_verified": bool,
  "target_colour_space": str | None,
  "split": "train|validation|test|calibration",
  "metadata_path": str | None,
  "license_note": str
}
```

## 5. Noise and illumination conditioning
Initial noise model:
\[
\operatorname{Var}(n_i\mid x_i)=a x_i+b
\]
Treat `a,b` as calibrated estimates, not universal constants. Use camera/ISO strata when supported. Estimate or read illumination descriptors (e.g. exposure metadata, robust luminance statistics, local illumination map) and pass them to adaptive denoising/refinement. If metadata is absent, mark the source of the estimate and confidence.

## 6. Sensor-consistency audit
Re-mosaic:
\[
\hat y=S_{\mathrm{CFA}}(\hat x_{\mathrm{lin}})
\]
Normalized residual:
\[
z_i=\frac{y_i-\hat y_i}{\sqrt{a\hat y_i+b+\epsilon}}
\]
Diagnostics remain separate until evidence supports combination:
- tile residual bias;
- residual energy/variance;
- neighbour/spatial correlation;
- CFA-channel-specific diagnostics;
- reliability: clipping, near-black, valid count, calibration confidence.

Important statistical warning: the reconstructed RGB is derived from the same noisy RAW being tested. The residual is therefore dependent on the observation. Do not use textbook Gaussian thresholds as production thresholds. Use empirical null calibration and held-out validation. Test whether audit statistics correlate with actual tile error and whether they can identify actionable defects.

States:
- `VERIFIED`: no significant inconsistency detected under calibrated test; not proof of correctness.
- `NEEDS_REFINEMENT`: significant and actionable discrepancy.
- `UNCERTAIN`: clipped, near-black, insufficient samples, unsupported camera/ISO, or unreliable calibration.

Significance and actionability are separate. A statistically unusual tile is not automatically a tile that benefits from refinement.

## 7. Scheduler and refinement
- Default ranking: raw audit evidence, after calibration and actionability checks.
- Budgets: 0%, 10%, 25%, 50%, 75%, 100% of eligible tiles.
- Skip `VERIFIED` and `UNCERTAIN` by default.
- Use overlap and feather blending to reduce seams.
- Re-audit after refinement and record stop reason.
- Optional learned gain-per-compute scheduler is an ablation, not core.

## 8. Rendering
Perform sensor-space audit before white balance, CCM, tone mapping, gamma and sharpening. Keep linear and display-referred outputs distinct. Avoid applying gamma twice. Store colour transform provenance.

## 9. Efficiency
Benchmark complete B0/B1/B2 pipeline:
- per-stage and end-to-end latency;
- audit overhead;
- peak RAM and accelerator memory;
- model parameters and serialized size;
- tiles refined and budget;
- quality-versus-latency curve.
Report device, OS, runtime, precision, warm-up and repetitions. Do not claim smartphone real-time performance unless tested on a named phone.

## 10. Limitations
- Re-mosaicing cannot detect arbitrary errors in unsampled colour channels when sampled sensor values remain consistent.
- A reference-free audit cannot establish perceptual quality.
- Clipped samples do not provide reliable evidence.
- Dataset RAW formats may differ; never treat Raw-RGB arrays as Bayer mosaics.
- Dataset camera/scene diversity is finite; generalization claims must match tested cameras.
