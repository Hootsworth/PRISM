# B1 Contract Freeze Report

**Run:** `PRISM-P3-20261106-003`  
**Date:** 2026-11-06  
**Decision:** **APPROVED WITH DOCUMENTED LIMITATIONS** for implementation of B1 under the contracts linked below. This approval does not mean implementation or training has occurred.

## What was verified

- Read and reconciled project ledger, B0 freeze/implementation/verification/performance records, expanded SIDD-CC audit, feasibility, architecture, protocol and dataset expansion documents.
- Re-ran `python3 scripts/audit_siddcc_expanded.py`: 115 train instances, 230 mapped RAW/sRGB pairs, 29 official test scenes and 32 matched blocks per scene passed format, metadata, dimensions and finite-value checks.
- Re-ran `python3 scripts/verify_b0_freeze.py`: PASS for all 45 frozen files.
- Verified old split counts and discovered the old instance-random split divides all ten documented scene numbers across train/validation. Exact-byte checks found no train/validation repeated files by matching RAW noisy, RAW GT and sRGB GT roles. Coarse pHash also surfaced related images across the old split, consistent with scene leakage.
- Created and verified a checksum-linked 144-group split manifest preserving old assignments alongside a corrected scene-group assignment. Scenes 009 and 010 are `internal_test`, scene 003 is `validation`, and all other official train scenes are `train`; no scene number spans those labels. Current split CSV SHA-256: `9e93f75071fb3fc4bf5e2c0e0ab127cc90104caafe231dda83f2a2d388ec441c`.

## Frozen scientific task

B1 learns a full Bayer mosaic to SIDD-CC camera-pipeline sRGB reconstruction. Input is native normalized, black-subtracted but unclipped RAW packed spatially into four fixed-position planes; output is full-resolution display-encoded RGB sRGB `[0,1]`. No CFA/camera/color metadata is supplied. Paired `GT_SRGB` is 8-bit and is the target; its rendering pipeline is only described at dataset level, not fully specified/calibrated. This is supervised reconstruction to a dataset target, not absolute colorimetric rendering.

## Data and model

The canonical scene-group split is train 96 instances, validation 12, internal matched B0/B1 comparison 7, plus untouched official test 29 scene groups ×32 blocks. Source pairing and instance checksums are in `B1_SPLIT_MANIFEST.csv`. The contract freezes compact two-level packed U-Net/PixelShuffle architecture at formula-estimated 357,276 parameters. The model must be runnable without test metadata. It cannot claim held-out-camera generalization.

## Objective and judgments

Training specification freezes float32 AdamW/MSE and scene-group validation selection. RGB sRGB PSNR is primary, SSIM secondary, Delta E 2000 descriptive and target-relative, LPIPS optional with fixed implementation; runtime/memory/parameters/storage are reported separately. Metrics are per block then macro-aggregated by scene, with all per-scene values. No quality metrics have been computed.

## Official metadata gap and comparison

Frozen B0 requires test CFA, AsShotNeutral and ColorMatrix2 absent from official test files. No inference is allowed. Primary fair B0/B1 matched comparison is on internal scenes 009/010, seven metadata-complete instances; validation is separate. This is small and only GP/IP, so descriptive. B1 additionally receives an independent official-test evaluation; it is reported B1-only. B0 official-test comparison remains blocked unless an authorized and verified metadata mapping becomes available.

## Remaining limitations and gate

- Only five camera families and ten documented train scene numbers; validation one scene; internal comparison two scenes/seven instances. No broad generalization claim.
- Target generation is camera-pipeline sRGB, not chart/profile ground truth. Colorimetric accuracy is unresolved and unclaimed.
- Registration verifies only zero integer offset on center crops; local/subpixel alignment remains uncertain.
- Near-duplicate pHash findings are heuristic. Scene grouping addresses documented scene identity, but exact cross-scene content equivalence is not proven.
- The prior 98/17 random-instance split was not used for model selection; the corrected scene-grouped split supersedes it for B1. The data/scripts/reports are frozen for implementation. No B1 code or training was performed during this contract task.
