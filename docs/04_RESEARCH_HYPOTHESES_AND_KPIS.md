# Research Question, Contribution and KPIs

## 1. P2-aligned research question
Can a hybrid smartphone ISP combine a conventional RAW processing path with a learned RAW-to-RGB/refinement model and a calibrated sensor-consistency controller to preserve image quality while reducing unnecessary learned computation across different noise and illumination conditions?

## 2. Candidate contribution
PRISM investigates whether calibrated, camera-aware CFA measurement residuals can act as a runtime compute controller for local learned refinement in a hybrid Bayer ISP.

The potentially distinctive part is the use of a calibrated sensor-consistency audit to decide where learned refinement should run. Residual statistics themselves are not claimed as new. The contribution is conditional on evidence that:
1. the audit predicts actual local reconstruction error;
2. it remains calibrated on held-out real captures;
3. it selects useful tiles better than random and simple gradient routing at equal compute;
4. the saved computation exceeds audit overhead;
5. final RGB quality remains competitive.

Do not claim "first", "novel", "state of the art" or guaranteed hallucination prevention without a proper literature review and evidence.

## 3. Hypotheses
- H1: The full-frame learned ISP (B1) improves at least one relevant quality dimension over the documented classical baseline (B0), without unacceptable colour/detail regressions.
- H2: PRISM hybrid (B2) can approach B1 quality with lower measured compute on some budget/condition regimes.
- H3: Calibrated audit ranking selects tiles with higher expected true error than random routing and a gradient-based heuristic at equal tile budgets.
- H4: Adaptive noise/illumination conditioning improves robustness over a fixed-parameter pipeline on held-out conditions.
- H5: At least one supported optimization (INT8 or FP16) improves measured efficiency with a quantified quality trade-off.

All hypotheses may fail. Negative results are valid findings.

## 4. Metrics and exact reporting rules
### Reconstruction quality
- PSNR (dB), with domain stated: linear RAW-derived RGB or display sRGB.
- SSIM, with data range and domain stated.
- LPIPS where available and appropriate; specify backbone/version and input normalization. If unavailable, choose and justify an alternative perceptual metric.
- ΔE00: state RGB-to-XYZ/Lab conversion, white point, reference illuminant and whether images are colour-managed.

Do not compare values computed in different domains or with different preprocessing.

### Efficiency
- End-to-end latency in ms/image, median and spread across repeated runs after warm-up.
- Per-stage latency and audit overhead.
- Peak memory and method of measurement.
- Parameter count and serialized model size.
- Tile count/fraction refined.
- Quality-versus-latency and quality-versus-budget curves.

### Audit quality
- Spearman correlation between audit score and true tile error where references exist.
- Per-defect detection rate, precision/recall and false positives/misses.
- Empirical null calibration plots.
- Nominal vs realized FDR, number of tests/discoveries and confidence/uncertainty.
- Spatial dependence and camera/ISO calibration drift.

### Robustness
Report metrics stratified by camera, ISO/noise level, illumination, clipping/near-black and input resolution where sample counts permit.

## 5. Required baselines
- B0 Classical ISP: documented conventional pipeline.
- B1 Learned ISP: standalone full-frame RAW-to-RGB model or verified adapted learned ISP.
- B2 PRISM hybrid: classical fidelity path + calibrated audit + tile-selected learned refinement.
- Routing controls: random tiles and highest-gradient tiles at equal budgets.
- B1 full refinement is the quality/compute reference for the same refiner if applicable.

Do not omit B1 full-frame learned ISP from the P2 comparison.

## 6. Required ablations
At minimum:
1. B2 without audit (random tile selection).
2. B2 with gradient-based tile selection.
3. B2 without re-audit / closed-loop iteration.
4. B2 without noise/illumination conditioning.
5. B2 without one audit statistic at a time (only retain statistics that add independent value).
6. B2 at 0/10/25/50/75/100% tile budgets.
7. B1/B2 before and after chosen quantization/reduced-precision optimization.

## 7. Interpretive limits
Sensor consistency is not RGB truth. Re-mosaicing observes only the CFA-sampled colour at each sensor site. An image can be sensor-consistent and still have errors in unsampled channels, colour rendering or perceptual detail. Show at least one such blind spot. Do not turn `VERIFIED` into a claim that the image is correct.
