# Testing and Evaluation Plan

## 1. Rules
Every run has an ID, manifest hash, config hash, seed, split and status: PASS, FAIL or INCONCLUSIVE. Synthetic tests are useful but do not replace real RAW validation. Never tune on final test data. Every bug fix adds a regression test.

## 2. Unit tests
### RAW and CFA
- DNG decode for supported files.
- CFA pack/unpack round-trip for RGGB, BGGR, GRBG and GBRG.
- Hand-built CFA indexing fixtures.
- Odd/even dimensions and borders.
- Black-level subtraction, white-level normalization and clipping masks.
- Dtype/bit-depth preservation.
- Unknown CFA/missing metadata fails clearly; no silent guessing.
- Raw-RGB MAT adapter tests its own channel layout and must reject Bayer-only operations.

### Classical ISP
- Demosaicing shape/range checks.
- WB gain application and metadata provenance.
- CCM shape, numerical stability and colour-space conversion.
- Denoiser preserves edges within defined fixture tolerances.
- Tone mapping and gamma are applied exactly once.
- Linear and display-referred outputs are labelled distinctly.

### Learned ISP
- Input/output shape and range.
- Dataset pair alignment and split leakage checks.
- Training checkpoint save/load/hash.
- Deterministic validation inference.
- Finite output and numerical stability.
- Parameter count and model size recorded.

### Audit/scheduler
- Re-mosaic samples correct CFA sites.
- Residual sign and normalization.
- Clipped/near-black/invalid samples excluded.
- Tile size/sample count and empty tile handling.
- Bias, residual energy, spatial correlation and per-CFA statistics independently tested.
- Missing calibration produces UNCERTAIN.
- FDR changes significance as expected.
- Only actionable NEEDS_REFINEMENT tiles enter queue.
- Budgets respected; deterministic tie-breaking.
- Feather blending seam tests and re-audit.

## 3. K0 synthetic audit test
Generate clean linear scenes, Bayer-sample them, inject controlled Poisson-Gaussian noise, run the fast classical path, re-mosaic and audit. Inject:
- over-smoothing / removed texture;
- under-denoising / leftover noise;
- demosaic zipper artefacts;
- sampled-channel colour shift;
- unsampled-channel colour error that remains re-mosaic-consistent (expected blind spot);
- noise-model mismatch and spatially correlated noise.

Measure audit-score vs true tile error, detection rates, false alarms and statistic redundancy. Theoretical Gaussian thresholds are not production thresholds.

## 4. K1/K2 real data calibration and validation
K1:
- Inventory DNGs and verify camera/ISO/CFA/bit depth/black-white levels.
- Select calibration captures only from calibration split.
- Estimate empirical null by supported strata.
- Measure residual spatial correlation and calibration mismatch.
- Version calibration artifacts and freeze thresholds.

K2:
- Apply tile-level multiple-testing correction (start with Benjamini-Hochberg).
- Evaluate spatial dependence and calibration drift.
- Report nominal FDR, realized FDR, number of tests, discoveries and false discoveries where reference labels permit.
- If sample size is insufficient, mark INCONCLUSIVE.

## 5. P2 image-quality evaluation
Run B0, B1 and B2 on the exact same held-out samples and target references. Report:
- PSNR and SSIM in the explicitly named image domain;
- ΔE00 with conversion and white-point details;
- LPIPS or justified perceptual alternative;
- edge/detail preservation metric or documented qualitative crop analysis;
- results stratified by camera, ISO/noise and illumination when possible.

Where no aligned reference exists, do not invent full-reference metrics. Report no-reference/qualitative results separately and clearly label them.

## 6. Fair comparison protocol
- Same test samples, target transform and crop/resize policy.
- No train/test capture overlap.
- Same hardware/runtime for latency comparisons.
- Warm up, repeat at least several times, report median and spread.
- Include preprocessing, audit and rendering overhead in end-to-end latency.
- Report both per-stage and end-to-end numbers.
- Compare equal tile budgets for routing controls.
- Record model parameter count and serialized size.
- Report where PRISM loses.

## 7. Optimization test
Implement at least one: INT8 quantization or FP16 reduced-precision inference. Prefer the option supported by the target runtime. Compare:
- output quality (PSNR/SSIM/ΔE00/LPIPS where available);
- end-to-end and model-only latency;
- peak memory;
- model size;
- device/runtime and precision.
An exported model file without measured comparison does not satisfy the optimization requirement.

## 8. Integration tests
1. Valid Bayer DNG → B0 → RGB and intermediate outputs.
2. Valid supported Bayer DNG → B1 → RGB.
3. Valid Bayer DNG → B2 → audit map → budgeted refinement → RGB.
4. Unknown CFA / corrupt file → readable error, no crash.
5. Missing calibration → UNCERTAIN state and safe fallback.
6. Clipped/near-black tiles → UNCERTAIN, not falsely certified.
7. Zero budget → fast-path output.
8. Full budget → all eligible tiles processed.
9. Re-audit stops correctly.
10. Same config/seed → equivalent output within tolerance.
11. Hosted app accepts a supported unseen input and provides download/report.

## 9. Required artifacts
1. Dataset inventory and pairing report.
2. Architecture diagram matching actual code.
3. Classical, learned and hybrid output gallery.
4. Audit score vs true tile error scatter plot.
5. Empirical null/calibration plots.
6. Nominal vs realized FDR plot.
7. Audit heatmaps and tile-state overlays.
8. Quality vs tile budget.
9. Quality vs measured latency including audit overhead.
10. Per-stage latency and peak-memory chart.
11. Baseline comparison table.
12. Ablation table.
13. At least three documented failure cases, including one audit blind spot.
14. Optimization before/after table.

## 10. Required evaluation file
Create `eval_P2_TeamID.md` using the exact team identifier assigned by organizers. The first line must contain every team USN separated by commas, with no heading before it. Follow with:
- problem ID and project name;
- dataset version/splits;
- baseline descriptions;
- metric definitions and results;
- model size/parameters;
- latency and memory/device;
- optimization comparison;
- ablation results;
- failure cases and limitations;
- links to live app and repository.
Do not fill unknowns with estimates. Use `NOT RUN` / `INCONCLUSIVE`.
