# B1/B2 Review Remediation

**Status: code/documentation corrections applied; full registered B1 training and the corrected MPS benchmark remain pending.**

## B1 run evidence

The archived MPS run at `runs/B1-test-gpu/` is a real Apple MPS run. Its summary identifies checkpoint SHA-256 `6940e82b0aa313858a3e995bb5284f037f913378c0008562e4b30a5e951e6f9d`, epoch 2, and validation scene-macro MSE 0.0472397420. Its run configuration explicitly caps training at two epochs. The frozen training specification allows up to 100 epochs and requires 15 stale validation epochs for early stopping. Accordingly, the MPS artifact is retained and labeled a two-epoch exploratory checkpoint; it is not evidence that the frozen training protocol completed. The earlier interrupted CPU run is retained as a separate historical run.

The full-training script supports `--epochs 100` and an explicit run ID. A fresh registered run could not be started from this execution runtime: Python 3.9.6 / PyTorch 2.8.0 reports `torch.backends.mps.is_available() == False`, CUDA is unavailable, and Homebrew Python 3.11 has no PyTorch installation. The previous CPU run measured about 157–162 seconds per epoch, making a full run several hours. No CPU substitute was run after the user requested use of the GPU. The next B1 run must execute in the MPS-enabled environment and preserve the epoch-2 files.

On the MPS-enabled runtime, the full run command is:

```sh
PYTHONPATH=src python3 scripts/train_b1.py --run-dir experiments/P3/runs/B1-contract100-mps-20261106-002 --run-id PRISM-P3-B1-20261106-002 --device mps --threads 4 --epochs 100
```

## B2 corrections

- `run_b2` no longer labels a second audit of the unchanged B0 camera-linear estimate as a post-refinement audit. A post-audit now requires a separately supplied camera-linear estimate; the display-sRGB B2 result cannot be remosaiced directly. By default, post-audit is reported as not computed.
- The hard-coded camera/ISO coefficients and image-derived variance estimates have no independent calibration provenance in this workspace. They are now explicitly provisional and `calibrated=False`. The audit therefore marks all tiles `UNCERTAIN`; the default scheduler selects none. No sensor verification, false discovery rate, or refinement-benefit claim is authorized by these coefficients.
- An explicit `allow_uncalibrated_selection` option supports exploratory budget ablations: it ranks non-extreme `UNCERTAIN` tiles by within-frame residual score while preserving their uncertain label. This opt-in is not calibration and makes no sensor-verification claim.
- Constructing any future `calibrated=True` noise model now requires a nonempty calibration reference, and coefficients must be finite/nonnegative.
- Tile labels now state `LOW_RELATIVE_RESIDUAL`, `HIGH_RELATIVE_RESIDUAL_CANDIDATE`, or `UNCERTAIN`. The within-frame percentile is documented as a relative heuristic, not a calibrated error threshold. Deprecated enum aliases remain for caller compatibility.
- Sensor consensus is now computed independently within each of the four Bayer phase planes; the previous full-resolution 3×3 filter mixed adjacent CFA colors. This fixes the indexing convention but does not remove scene-texture confounding or validate the sensor-error detector.
- `budget_fraction` is now the maximum fraction of all sensor tiles available for refinement; selection is capped by eligible candidates. Reports include actual selected count, fraction of eligible candidates selected, and fraction of total tiles refined. B2 raises if actionable tiles are selected without a model.
- The B2 CLI requires an explicit checkpoint. Its JSON summary records checkpoint path/hash/epoch, calibration state, and post-audit status; it no longer silently uses random weights.

## Comparative evaluation corrections

The old comparison remains at `evaluations/comparative_B0_B1_B2/` as historical evidence. It used one sequential timing pass and a two-epoch checkpoint; its runtime numbers do not satisfy the frozen timing protocol. The evaluator has been replaced with an explicit-checkpoint interface, frozen manifest/split validation, output and checkpoint hashes, per-image and per-scene files, one or more warm-ups, at least ten interleaved repetitions, accelerator synchronization, full-path load/decode/pack/inference timing, separate PNG save timing, median/mean/IQR/p95 reporting, the requested seven budgets, and a Conv2d MAC/FLOP proxy. The fixed metric parameters are in `B1_METRICS_CONFIG.json`.

The corrected benchmark was not completed. In the current shell, MPS is unavailable. A CPU attempt was interrupted after its first image at the user's direction; it wrote no metric artifacts and its partial timings must not be used. The saved MPS summary remains the only prior comparison and is explicitly one-pass evidence. The seven-budget ablation uses explicit exploratory relative-residual selection for non-extreme uncertain tiles; results must be labeled uncalibrated and cannot validate the audit.

After the full checkpoint is frozen, run the corrected comparison on MPS with:

```sh
PYTHONPATH=src python3 scripts/evaluate_comparative.py --checkpoint experiments/P3/runs/B1-contract100-mps-20261106-002/best.pt --device mps --warmup 1 --repetitions 10 --save-images --output-dir experiments/P3/evaluations/comparative_B0_B1_B2_mps_full
```

## Current interpretation

The prior MPS comparison reported internal-test scene-macro PSNR 26.08 dB for B0, 13.46 dB for B1, and 24.71 dB for B2; SSIM 0.572, 0.423, and 0.566 respectively. These are descriptive sRGB target-similarity scores on 14 internal images using the epoch-2 checkpoint. They show no B1 or B2 improvement over B0 for that checkpoint. The corrected B2 default does not refine uncalibrated tiles; only the explicit ablation mode will run residual-ranked uncertain tiles. These data do not establish B2 efficacy, colorimetric accuracy, or generalization.

## Verification performed

- Python syntax compilation and import/default/provenance smoke checks succeeded for the revised evaluator, B2 runner/pipeline, noise model, audit, and scheduler using `PYTHONPYCACHEPREFIX=/private/tmp/prism-pyc`.
- No unit or integration tests were run during this remediation. The prior ledger's 41/41 claim is historical and has not been independently reproduced in this pass.
- No dataset or frozen B0 artifacts were modified.

