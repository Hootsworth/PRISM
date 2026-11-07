# B1 Evaluation Contract

**Status:** frozen before implementation. Quality metrics are not yet computed.

## Data and output rules

For RGB evaluation, convert prediction float32 and paired GT sRGB uint8 to RGB float32 `[0,1]`; require exact dimensions/channel order and the declared paired crop. No per-image rescaling, automatic registration, or metric-specific exposure adjustment. The network sigmoid already bounds output to `[0,1]`; no further clipping. The original RAW remains unclipped for any sensor diagnostics. Report alignment caveat: current registration evidence is a center-crop integer-offset diagnostic, not proof of full-frame subpixel/local registration.

Evaluation splits:

- Validation scene 003: selection only.
- Internal paired B0/B1 comparison: scene numbers 009 and 010, 7 instances, metadata-complete; no checkpoint selection or hyperparameter tuning on these groups.
- Official B1 test: all 29 official test scene groups and 32 corresponding blocks each. Test remains untouched until the model/checkpoint is frozen. Do not infer missing metadata.
- B0 official-test metrics: unavailable unless verified test metadata is recovered. B1-only test results are reported separately; they are not called B0/B1 comparisons.

## Metrics

| Metric | Representation, data range and clipping | Alignment / aggregation | Interpretation |
|---|---|---|---|
| PSNR (primary) | Display sRGB RGB float32 `[0,1]`, `data_range=1`. Prediction already sigmoid-bounded; no rescaling. | Exact corresponding pixels. Compute each block, average blocks within each official test scene, then macro-average 29 scenes. For full-resolution train-derived internal test, average two pair scores within instance, then macro-average 7 instances. Also publish per-scene values and dispersion. | Agreement with SIDD-CC rendered target, not universal color fidelity. |
| SSIM (secondary) | Same sRGB pair and `[0,1]`; fixed implementation, Gaussian/window settings and channel handling registered in code/config before run. No clipping/rescaling. | Same spatial correspondence; block scores → scene mean → macro mean. | Structural similarity to this rendered target. |
| Delta E 2000 (descriptive) | Convert standard sRGB using IEC transfer decoding, D65 sRGB→XYZ matrix, D65 white→Lab, then CIEDE2000; no image-specific adaptation. | Per-pixel corresponding values; report per image mean, median, 95th percentile; macro-average image-level summaries. | Target-relative appearance difference only. No chart/profile, so not a color-accuracy result. |
| LPIPS (optional) | Display sRGB only, named implementation/backbone/version, fixed RGB normalization to its expected input interval. | Same dimensions; image-level then scene-macro. | Perceptual similarity to SIDD-CC rendering, not physical truth. Do not make primary without installing/fixing dependency before evaluation. |
| RAW input baseline RMSE/PSNR | Native normalized float32 Bayer, fixed nominal `data_range=1.0`; retain negatives/overrange and report unclipped error. | Noisy RAW vs paired GT RAW, pair-level then instance-macro. | Sensor-domain denoising baseline only; not B1 output quality. |
| Runtime | Full path includes file load/validation, pack, network, tiling/stitch, output save separately; model-only uses preloaded tensor. | Same named device/software, one warm-up, ≥10 measured repeats, interleave methods, sync accelerator, median/mean/IQR by scene and overall. | No cross-device claim. |
| Memory, params, storage | Process high-water RSS and accelerator peak separately; parameter count from instantiated model; checkpoint file bytes. | Training and inference reported separately. | 357,276 is proposal count; report actual instantiated count and serialized bytes after implementation. |

No metric is pooled over pixels across all images as the sole result; scene groups are the primary independent units. Report all 29 official test scene scores and macro-average, plus interval/dispersion. With only 29 scenes and five camera families, do not claim generalization across cameras/lighting beyond this dataset. Do not use sensor consistency as a substitute for independent RGB evaluation.

## Reproducible interface

The evaluation implementation, when written, must accept explicit `--split-manifest`, `--split-column split_frozen_scene_grouped`, `--checkpoint`, `--data-root`, `--output-dir`, `--device`, and `--metrics-config`; emit per-image CSV, per-scene CSV, aggregate JSON, environment JSON, and output/checkpoint hashes. Validate manifest and dataset hashes before reading images. It must refuse validation use as official test and refuse B0 official-test evaluation when required metadata is unknown.

## B0 / B1 / B2 system comparison

- **B0:** frozen deterministic full-frame Bayer reconstruction and color pipeline; no retuning, no denoising. Exact frozen hash verifier must pass.
- **B1:** this standalone full-image learned Bayer-to-sRGB model.
- **B2:** future B0 output plus sensor-consistency audit, then selective neural refinement only on selected regions; this contract does not authorize implementing B2 yet.

For every direct quality comparison, hold constant the RAW source frames, paired target, crop/block coordinates, alignment, metric code/settings, device, software, precision, and output RGB convention. The primary matched B0/B1 comparison uses exactly the seven internal test instances (same RAW, GT_SRGB, crop) and reports only descriptive results because it covers two scenes/two cameras. Official B1-only test is a separate table. B0 may not run against official test by substituting guessed metadata.

For B2 report (1) RAW load + B0 reconstruction, (2) audit-only overhead, (3) selection/scheduling overhead, (4) neural model time, (5) stitching/output time, and (6) total end-to-end latency. Also report neural-only latency, percentage and count of tiles refined, tile size/overlap, total and peak RSS/accelerator memory, and PSNR/SSIM under the same target. Compare B2 to both B0 and full-frame B1 on the same internal seven groups. Fewer tiles is not evidence of lower latency unless measured end-to-end. Any threshold selection is validation-only; the internal comparison and official test cannot tune it.

## Testable research hypotheses

| Hypothesis | Independent variable | Dependent variables | Controls and evidence | Failure condition |
|---|---|---|---|---|
| H1: standalone B1 improves reconstruction agreement over B0 under the frozen SIDD-CC conditions. | System (B0 vs B1) on identical internal held-out inputs. | Scene-macro sRGB PSNR primary; SSIM secondary; runtime/memory separate. | Seven internal test captures, frozen checkpoint selected on scene 003, same target and metric implementation. | B1 does not improve scene-macro PSNR over B0, or gains are within uncertainty / tradeoffs are unacceptable. Do not rescue with official test tuning. |
| H2: the sensor-consistency audit identifies regions where refinement yields useful quality gains. | Audit score / selection decision vs no audit, evaluated at fixed tile and threshold settings. | Association between audit score and independent tile error; precision/recall or AUROC for predeclared “refinement helpful” label; gain by selected tiles. | Label helpfulness by held-out reference error after a predeclared refiner; no audit-score fitting on internal test; compare against random and uniform tile controls. | No ranking above chance, or audit-selected tiles fail to improve quality relative to same-compute random/uniform controls. |
| H3: selective refinement lowers compute versus full-frame learned inference while retaining competitive quality. | Full-frame B1 vs B2 selection/refinement fraction and scheduling policy. | End-to-end latency, neural latency, refined-tile fraction, PSNR/SSIM and memory. | Same seven internal test groups, device/software, quality metrics; separately report audit/scheduling cost. | End-to-end latency is not lower at matched quality, or quality degradation exceeds preregistered tolerance. |

These are unproven hypotheses. H2/H3 require a future B2 contract, refinement labels and a separate scene-held-out tuning protocol before execution. No numeric acceptance margin is asserted here; set it from project requirements before B2 experiments, never after observing test results.
