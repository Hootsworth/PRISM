# B1 Implementation and Partial Training Run

**Status: implementation present; preregistered training incomplete and stopped at the user's request. No B1 evaluation was run.**

## Implementation

The frozen compact packed Bayer U-Net is implemented in `src/prism/models/b1.py`. It accepts float32 `[N,4,H/2,W/2]` input in `[TL,TR,BL,BR]` order and returns full-resolution three-channel sigmoid sRGB. Its parameter count is 357,276 as specified. `src/prism/models/b1_inference.py` provides overlap-weighted tiled full-frame inference. `scripts/train_b1.py` implements the registered FP32 AdamW/MSE training, deterministic seed, train-only crop/augmentation pipeline, hash preflight, full validation-scene selection and 15-epoch early stopping. `scripts/evaluate_b1.py` was written for post-freeze evaluation, but deliberately was not run.

An initial runtime check used a random 32×40 packed tensor: output shape was 64×80 RGB, values were finite, and parameter count matched. This checks basic tensor shape and execution only; it is not an ISP quality test.

## Environment and GPU availability

Run environment: macOS 27.0.1 arm64, Python 3.9.6, PyTorch 2.8.0, CPU with 4 PyTorch threads. The Apple M5 GPU is present and the installed PyTorch reports MPS built, but `torch.backends.mps.is_available()` is false and a direct MPS tensor operation fails to initialize. CUDA is unavailable, and MLX is not installed. The registered FP32 protocol was therefore run on CPU; no dependencies or environment were changed to attempt GPU setup.

## Partial run

- Run ID: `PRISM-P3-B1-20261106-001`.
- Exact command: `PYTHONPATH=src python3 scripts/train_b1.py --device cpu --threads 4`.
- Before training, all 1,163 source file hashes passed (33,262,165,122 bytes); frozen split, pair inventory and source manifest hashes are recorded in `runs/B1-20261106-001/config.json`.
- Two full epochs completed. Training MSE: 0.09086615, then 0.05182800. Validation scene-macro MSE: 0.08931139, then 0.03031943; both completed epochs improved the validation checkpoint. These are interim optimization values, not final model or test metrics.
- Epoch 1: train 34.22 s, validation 122.77 s. Epoch 2: train 35.11 s, validation 126.84 s. Process peak RSS: 2,418,167,312 bytes.
- Training was interrupted while loading a training sample during epoch 3, following the user's request to stop due to runtime. The interrupt did not complete epoch 3. `best.pt` and `last.pt` contain the epoch 2 state and are retained as artifacts from this partial run; neither is a frozen final checkpoint.
- A full epoch took about 2.7 minutes in the first two observations. A 100-epoch run was projected at about 4.5 hours, excluding later evaluation. Actual early-stopping duration was unknown.

## Evaluation and limitations

Internal B0/B1 and official-test evaluation were not run. No test data was used for optimization, and no PSNR/SSIM model comparison, colorimetric claim, or generalization claim is made. The two validation epochs are not enough to establish convergence or select a final checkpoint. The implementation still needs formal unit/regression tests, a completed registered training run, frozen checkpoint hash, and then the contract-ordered internal and official evaluations before B1 can be reported as complete.

