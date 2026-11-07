# B0/B1 Test Compatibility and Metadata Gap

## Verified state

B0 is frozen under `experiments/P2/B0_FREEZE_RECORD.md` and its 45-file manifest. It requires capture CFA, AsShotNeutral and ColorMatrix2 for demosaicing/color conversion. The official SIDD-CC test package provides paired RAW and sRGB blocks, common block coordinates/order, and scene information, but not those per-capture color/CFA fields. They remain unknown. The metadata-free B1 contract needs only the RAW array and so is executable on official test data.

## Options

| Option | Feasibility / reproducibility | Leakage and fairness | B0 compatibility / official protocol |
|---|---|---|---|
| A. B1 official test; B0 on metadata-complete data separately | Immediately feasible for B1; reproducible. | No test leakage if checkpoint frozen. Cross-split B0 vs B1 scores cannot be compared numerically as a head-to-head result. | B0 can run on internal test/validation captures. Preserves official B1 test but does not solve same-split comparison. |
| B. Recover official test metadata from an authorized documented source | Not currently evidenced. Full-frame test images are unpublished; current downloaded docs/tensors do not map CFA/AsShotNeutral/ColorMatrix2. | Defensible only after independently verified mapping, source/license and checksum. | Would allow frozen B0 on official test and preserve official protocol; not available now. Do not infer. |
| C. Reserve metadata-complete capture groups from official train for common B0/B1 evaluation | Feasible and reproducible. Frozen manifest reserves scenes 009+010 (7 instances) as internal paired comparison; validation scene003 is separate. | No train/validation/internal-test scene overlap. Internal test is small and camera-limited (IP=5, GP=2); descriptive, not a replacement for official test. | Runs frozen B0 with actual capture metadata and B1 on identical RAW/GT pairs. Official test remains official; B1 may additionally report there. |
| D. Infer metadata from scene IDs, neighboring captures or train averages | Rejected. | Fabricates unavailable inputs and risks camera/scene leakage. | Violates frozen B0/data contract. |

## Recommendation

Use **C as the primary matched B0/B1 comparison** on the seven internal held-out instances (five IP, two GP), with B1 selected only on validation scene 003. Separately evaluate frozen B1 on all official test blocks and report it as B1-only. This supports an honest common-input comparison while preserving the official test. It is not a full official-test comparison and only represents two cameras in its internal matched subset.

Option A is the secondary presentation path if a future experiment elects not to use the internal comparison: report official B1 test and metadata-complete B0 validation/internal scores in separate tables, never as paired head-to-head. Option B can supersede this limitation only after the metadata source is found, authorized, mapped, independently audited and checksum-recorded. Do not change B0 or use validation as official test.
