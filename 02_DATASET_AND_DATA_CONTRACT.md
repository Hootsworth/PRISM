# Dataset and Data Contract

## 1. What the supplied P2 brief names
The P2 datasheet names:
1. SIDD dataset.
2. Modular Neural ISP datasets and code.

The assignment does not specify exact versions, splits, file subsets, or a guaranteed DNG-to-ground-truth pairing protocol. The team must document those choices and verify them against the downloaded files.

## 2. SIDD: use the representations correctly
The official SIDD dataset page describes:
- noisy Raw-RGB MAT arrays;
- ground-truth Raw-RGB MAT arrays;
- noisy and ground-truth sRGB PNGs;
- metadata extracted from DNG files, including black/saturation levels, as-shot neutral and noise-level-function information;
- DNG images corresponding to noisy Raw-RGB samples;
- Bayer-pattern and noise-level-function metadata tables.

**Critical distinction:** Raw-RGB MAT arrays are not automatically a single-plane Bayer mosaic. Do not feed them into a Bayer CFA packer or claim they are DNG Bayer samples. Inspect shapes, channel order, scaling and metadata. Use DNGs for Bayer-native input only when CFA and metadata are successfully parsed.

### SIDD roles
| SIDD asset | Intended role | Restrictions |
|---|---|---|
| DNG archives | Bayer RAW ingestion, CFA tests, sensor-native audit and RAW-to-RGB input experiments | Verify file-to-reference mapping, CFA, camera metadata, bit depth and decoding |
| Noisy Raw-RGB MAT | Paired denoising / raw-domain experiments if channel semantics are verified | Not a Bayer mosaic; do not use for CFA-specific audit as if single-plane |
| Ground-truth Raw-RGB MAT | Raw-RGB target for matching noisy Raw-RGB input | Preserve same representation and alignment |
| sRGB noisy/GT PNG | Display-space denoising or image-quality checks | Not proof of DNG-to-sRGB pairing until mapping is verified |
| Metadata MAT / CSV | Noise and camera metadata, WB/CCM clues | Missing fields must remain unknown, not fabricated |

### Dataset feasibility gate (must run before training)
Create `data/manifests/sidd_inventory.csv` with:
- path, sample ID, capture/scene ID, camera, ISO, CFA, dimensions, bit depth, representation, metadata presence, target path, alignment status, split and license note.
Run a small inventory on every downloaded subset. Verify:
1. DNG decoding works.
2. CFA pattern and black/white levels are available.
3. DNG sample can be mapped to the intended reference target.
4. Raw-RGB MAT tensor shape/channel meaning is understood.
5. noisy/GT pairs are aligned.
6. there are enough distinct captures/scenes for capture-level splits.

If DNG-to-sRGB target pairing cannot be established, do not pretend it exists. Use SIDD Raw-RGB pairs for the learned denoising component and find a verified RAW-to-sRGB paired source from the supplied Modular Neural ISP data path for the full learned ISP. Clearly report the split between those two uses.

## 3. Modular Neural ISP repository
Official code: https://github.com/SamsungLabs/modular_neural_isp

Treat it as:
- a reference for modular ISP stages and a potential learned ISP baseline;
- a source of preprocessing/evaluation ideas;
- a candidate compatible data pipeline only after its actual dataset files are available and licensed.

The repository's dataset documentation describes S24, MIT-Adobe FiveK and Zurich raw-to-sRGB data layouts. The documented layout includes raw images, denoised raw images, metadata JSON and sRGB targets. It notes that only S24 supplies denoised RAW images, while Zurich and Adobe FiveK use pseudo-denoised RAW targets generated through their documented preprocessing. Do not assume those datasets are bundled with the code or freely downloadable merely because the repository is public.

### Modular ISP data feasibility gate
Before adopting any subset:
- record exact dataset name/version/source and license;
- confirm the team can access the image files;
- inspect RAW representation, dimensions, bit depth and metadata;
- verify input-target alignment and target colour space;
- check camera overlap with SIDD;
- build scene/capture-level split manifests;
- run one sample through the repository's documented preprocessing/inference path;
- compare its outputs and dependencies with the project's environment.

Do not vendor or redistribute third-party datasets or checkpoints unless their license permits it. Cite the repository/paper in the final report.

## 4. Recommended dataset plan
| Role | Preferred data | Why | Gate |
|---|---|---|---|
| CFA/RAW loader and sensor audit | SIDD DNG archives | Bayer-native samples and metadata are available | DNG decode + CFA/metadata audit |
| Noise calibration and paired denoising | SIDD noisy/GT Raw-RGB pairs | Real smartphone noisy/GT pairs | Verify MAT semantics and pair alignment |
| Full learned RAW-to-RGB ISP training | Verified DNG-to-sRGB pairs from SIDD if mapping is proven; otherwise compatible Modular ISP raw-to-sRGB subset | Needed for true learned ISP baseline | Must prove pairing and colour-space contract |
| Cross-camera / generalization | Held-out camera/capture groups from available data | Tests robustness | Capture- and camera-level separation |
| Public demo examples | A small permitted subset plus user-uploaded supported DNG | Stable demo and unseen input | License and privacy checks |

A newer reprocessed SIDD-CC resource may be considered only as an optional supplementary dataset after reviewing its own license, split protocol and data format. It is not silently substituted for the SIDD version named in the assignment.

## 5. Split protocol
- Split by scene/capture before extracting patches.
- Keep all patches from a capture in one split.
- For the strongest cross-camera claim, hold out at least one camera entirely if sample counts allow.
- Calibration/null estimation data must be separate from the final held-out test data.
- Never tune thresholds, model selection or hyperparameters on the final test set.
- Save manifest hashes and split-generation seed.

## 6. Data rights and reproducibility
Record source URL, download date, version, license, permitted use, preprocessing scripts and hashes. Keep data outside Git unless license and repository policy explicitly permit it. Include instructions for obtaining data, not the dataset itself, in the repository.
