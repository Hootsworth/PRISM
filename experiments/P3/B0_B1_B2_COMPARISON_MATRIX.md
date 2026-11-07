# B0 / B1 / B2 Comparison Matrix

| Dimension | B0: frozen classical | B1: standalone learned ISP | B2: hybrid prototype |
|---|---|---|---|
| Status | Implemented, tested, hash-frozen | Implemented; available MPS run has 2 epochs and is exploratory, not the frozen 100-epoch run | Pipeline implemented; noise model is uncalibrated, so default audit marks tiles uncertain and schedules no refinement |
| Input | Normalized Bayer RAW plus CFA and color metadata | Bayer RAW packed by fixed cell positions; no metadata input for official-test compatibility | Bayer input, B0 estimate, audit statistics and selected tiles |
| Output | Linear sRGB plus optional tone-mapped/encoded sRGB | Full-resolution camera-pipeline sRGB | RGB reconstruction plus audit maps/report and selective refinement |
| Targets | SIDD-CC GT_SRGB available for verified train/validation pairs; no test metadata for B0 transform | SIDD-CC GT_SRGB aligned to 230 train indices and 928 official test blocks | Same RGB target contract as B1, plus sensor consistency evidence |
| Quality metrics | sRGB PSNR/SSIM possible where B0 metadata exists; target-relative only | sRGB PSNR/SSIM and LPIPS possible; ΔE as image difference only, not absolute color accuracy | Same, plus audit and quality-vs-compute measures |
| Sensor consistency | No learned sensor audit | Not the main objective; can be audited separately where RAW prediction exists | Exploratory relative-residual diagnostic; calibration and error-prediction validity remain unestablished |
| Efficiency evidence | CPU median 1986.2 ms/image optimized, 2.28 GiB process high-water RSS on five scenes; environment discrepancy documented | Prior MPS timing is one pass and not the registered repeated protocol | Prior MPS comparison is one pass; a 10-repeat MPS rerun remains pending |
| Main risk | Approximate ColorMatrix2-only path; no denoising/sharpening; requires metadata absent from official test | Must infer CFA phase/camera response without test metadata; only 5 camera families; renders SIDD-CC target | No calibrated noise law or independently validated error labels; no post-refinement sensor audit exists for display-sRGB output |

## Fair comparison rules

1. Run the B0 freeze verifier before and after development; preserve reference outputs.
2. Use the same RAW input, target, crop and output domain. B0/B1/B2 full-ISP comparison uses the same GT_SRGB and fixed `[0,1]` display-sRGB range. RAW metrics belong in a separate table for RAW inputs or a RAW-denoiser variant.
3. B0 is not a denoiser; B1 target `GT_SRGB` includes denoising and rendering. Report this scope difference and include the noisy-sRGB input baseline.
4. The official test has RGB references but no test capture metadata required by B0. Do not run B0 with inferred train metadata. Until metadata is authorized and verified, compare B0/B1 on the metadata-complete validation split and report B1's official-test result separately.
5. Report reconstruction similarity, sensor consistency, target-relative color difference, perceptual similarity and computation separately. No target-relative RGB score proves absolute color accuracy.
6. Use the same named hardware, warm-ups, repetitions, backend/precision, peak memory method, model-only and end-to-end protocols from [B1_EXPERIMENTAL_PROTOCOL.md](B1_EXPERIMENTAL_PROTOCOL.md).
7. B1 must reconstruct a full image independently. A tile refiner alone belongs to future B2 and does not replace B1.

## Research questions

After B1 is implemented: (H1) Does learned Bayer-to-sRGB improve similarity to SIDD-CC GT_SRGB over frozen B0 on the same validation captures? (H2) Does B1's no-metadata inference remain stable across CFA patterns and cameras represented in the dataset? (H3) Does B2 improve quality per compute by selectively refining audited regions? A held-out-camera conclusion is premature because all five camera models occur in training. H3 requires a calibrated sensor audit, useful refinement selection and a fixed compute budget.

## Frozen split update (PRISM-P3-20261106-003)

The fair comparison rule above is superseded for execution. Do not use the prior random-instance 98/17 split. The dataset README documents a scene-number field, and the old split placed the same scene numbers on both sides. Use the scene-grouped manifest: train 96 instances, validation 12 (scene 003), internal matched B0/B1 comparison 7 (scenes 009/010), official test 29 unchanged. The internal comparison is limited to IP and GP cameras and is descriptive. B1 may separately report official-test results; frozen B0 cannot run there until authorized test metadata is verified. Full frozen comparison rules are in `B1_EVALUATION_CONTRACT.md`.
