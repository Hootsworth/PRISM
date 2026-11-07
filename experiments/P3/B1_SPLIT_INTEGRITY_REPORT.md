# B1 Split Integrity Audit

## Dataset-level facts rechecked

Pinned revision `5eb2ab6375d7f99f36b59bb5aacb46af18409412`; source-file manifest lists 1,163 files. Audit command `python3 scripts/audit_siddcc_expanded.py` passed: 115 official train instances, 230 exact-index RAW/SRGB pairs, 29 official test scenes ×32 blocks, documented MAT/PNG types/shapes, finite RAW, and common modality block coordinates. The official train/test instance ID lists have no overlap. Source data was not modified.

The dataset README parses instance names as `instance-id / scene-number / camera / ISO / shutter / illuminant / brightness`. The old manifest assigned 98/17 by randomized instances stratified on camera. Audit of that actual CSV found **all ten scene numbers occur in both train and validation**: scene 001 has 11/7; 002 10/1; 003 10/2; 004 12/1; 005 12/2; 006 13/0; 007 14/0; 008 9/4; 009 5/0; 010 2/0. This is capture/scene leakage for scene-generalization evaluation. Therefore the older claim that the 98/17 split has no scene leakage is withdrawn. Its assignments remain preserved in `split_existing_instance_random` for auditability.

## Frozen corrective assignment

`B1_SPLIT_MANIFEST.csv` records both original and corrected assignments. Canonical column is `split_frozen_scene_grouped`:

- Train: scenes 001, 002, 004–008 = 96 instances / 192 pairs.
- Validation: scene 003 = 12 instances / 24 pairs.
- Internal paired B0/B1 test: scenes 009 and 010 = 7 instances / 14 pairs.
- Official test: 29 scenes / 928 blocks, unchanged.

The split seed is n.a.; fixed scene IDs are explicit. Reproduce the checks with `python3 scripts/audit_b1_split_integrity.py`. Group checksums derive from the audited per-file SHA-256 manifest; source manifest hash is recorded in every row. SHA-256 of the current 144-row CSV: `9e93f75071fb3fc4bf5e2c0e0ab127cc90104caafe231dda83f2a2d388ec441c`.

Train distributions: five cameras (S6=13, GP=20, N6=15, G4=9, IP=39), ISO 50–6,400, brightness L=41/N=53/H=2. Validation scene 003 contains GP=4, IP=5, G4=1, N6=2 (no S6), ISO 100–10,000, brightness L=4/N=8. Internal test scenes 009/010 contain IP=5 and GP=2, ISO 100/800/1600, brightness L=3/N=4. Official test metadata remains unknown. There are only ten underlying train scene groups and five camera families; external generalization claims are out of scope.

## Duplicate/near-duplicate checks

- Exact SHA-256 duplicate comparison using the audited source file manifest found zero repeated noisy RAW, clean RAW, or clean sRGB files across the corrected train/validation/internal-test assignments. 
- A coarse 64-bit DCT perceptual hash (32×32 grayscale thumbnail, 8×8 low-frequency DCT) was computed for each of the 230 clean sRGB targets. Under the old split, cross train/validation comparisons included 6,664 pairs with minimum distance 0 and many ≤4-bit candidates, often sharing the documented scene number. Under the corrected assignment, 4,608 train/validation, 2,688 train/internal-test, and 336 validation/internal-test comparisons had minimum distances 20, 20, and 27 respectively; none were ≤8 bits. This is a useful duplicate candidate screen, not proof of semantic uniqueness; near-duplicate similarity is content- and hash-dependent.
- No image patches were assigned independently. Original pair and all derived crops remain in their instance/scene group.

No change is silent: the pre-existing split remains in its original CSV and in the side-by-side frozen manifest; the corrected scene-grouped split is the canonical assignment for B1 training/evaluation.
