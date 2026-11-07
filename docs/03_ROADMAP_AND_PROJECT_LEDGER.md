# Roadmap and Project Ledger

## 1. Gated roadmap
Do not skip gates. A failed or inconclusive gate must be reported honestly.

| Phase | Build | Exit evidence | Blocks |
|---|---|---|---|
| P0 Foundation | Repository audit, environment, configs, manifest schema, run IDs, ledger, smoke test | Clean install + smoke test + repository map | All |
| P1 Data/RAW | SIDD inventory, Modular ISP data audit, DNG/Raw-RGB adapters, CFA tests | Inventory manifest; format and pairing report; RAW unit tests | P2–P5 |
| P2 Classical ISP | B0 full classical ISP and intermediate exports | B0 processes supported test input; PSNR/SSIM/ΔE where paired; timing | P6 |
| P3 Learned ISP | B1 full-frame packed-Bayer-to-RGB U-Net or verified learned ISP adaptation | Training split, checkpoint, held-out inference, metrics | P5/P6 |
| P4 Audit validation | K0 synthetic + K1 real calibration + K2 held-out FDR + K3 controlled defects | Audit/error relationship, empirical null, FDR report, known blind spot | P5 |
| P5 Hybrid | B2 scheduler, tile refiner, blending, re-audit, stopping | End-to-end hybrid runs and budget controls | P6 |
| P6 Comparative evaluation | B0/B1/B2, ablations, robustness, failure cases | Same held-out images, metric tables, plots | P7/P8 |
| P7 Optimization | INT8 or FP16 (at least one), optional ONNX | Before/after quality, latency, memory, model size on named device | P8 |
| P8 App/submission | Hosted web app, repo docs, evaluation file, diagram, demo | All P2 deliverables pass checklist | Final |

Optional stretch only after P0–P8: learned gain-per-compute scheduler, audit-gated sharpening, anytime preview, label-free sensor adaptation.

## 2. Experiment registry
| ID | Purpose | Dataset | Main outputs | Initial status |
|---|---|---|---|---|
| D0 | Dataset inventory and format/pairing audit | SIDD + Modular ISP candidate data | inventory CSV, pairing report | NOT RUN |
| C0 | Classical baseline | verified Bayer inputs / compatible pairs | B0 images, metrics, timing | NOT RUN |
| L0 | Learned full-frame ISP | verified paired RAW-to-RGB source | checkpoint, training curves, metrics | NOT RUN |
| K0 | Synthetic audit behaviour | synthetic Bayer + known noise/defects | audit-vs-true-error plots | NOT RUN |
| K1 | Empirical null calibration | SIDD real RAW/DNG where supported | calibration profiles and plots | NOT RUN |
| K2 | Held-out FDR validation | capture-held-out real data | nominal vs realized FDR | NOT RUN |
| K3 | Controlled defect tests | synthetic + real examples | detection/miss matrix and failure gallery | NOT RUN |
| H0 | Hybrid PRISM | same held-out evaluation set | B2 outputs, budget/latency curves | NOT RUN |
| A0 | Required ablations | same held-out set | ablation table | NOT RUN |
| O0 | Optimization | named target device/runtime | INT8/FP16 benchmark | NOT RUN |
| U0 | Deployment validation | hosted app | URL test, unseen upload, logs | NOT RUN |

## 3. Suggested research gates
These are provisional engineering targets, not assignment-mandated thresholds and not achieved results. Freeze them before held-out evaluation; revise only with a documented reason.
- K0 audit ranking: Spearman rho target ≥0.6; below 0.3 is a serious warning that the proposed audit may not route useful refinement.
- K2 FDR: realized FDR within 2× nominal target where sample size supports estimation; otherwise INCONCLUSIVE.
- H0 efficiency: investigate whether quality is within 0.1 dB PSNR of full refinement while refining fewer than half the tiles. Report if not achieved.
- Baseline fairness: same held-out images, target transform and metric implementation.
- Optimization: report actual before/after quality and efficiency; no minimum speedup is assumed.

## 4. Single source of truth: `docs/PROJECT_LEDGER.md`
Every experiment run must record:
- run ID, date/time, operator, Git commit and clean/dirty status;
- OS, Python, framework/library versions, device/runtime;
- dataset version/source/license and manifest hash;
- capture/scene/camera split, ISO, CFA and bit depth;
- config path/hash, seed and deterministic settings;
- model/checkpoint hash and parameter count where relevant;
- exact command, logs and artifact paths;
- metric value, units, aggregation method and sample count;
- pre-registered gate, PASS/FAIL/INCONCLUSIVE decision;
- deviations, interpretation, limitations and next action.

Run ID format: `PRISM-{EXPERIMENT}-{YYYYMMDD}-{NNN}`.

## 5. KPI registry template
| KPI | Metric | Domain / unit | Target | Latest | Run ID | Status |
|---|---|---|---|---|---|---|
| KPI-RAW-01 | Supported Bayer decode rate | % of inventory | Report | TBD | TBD | NOT RUN |
| KPI-QUAL-01 | PSNR | dB, specify domain | Compare B0/B1/B2 | TBD | TBD | NOT RUN |
| KPI-QUAL-02 | SSIM | unitless, specify domain | Compare B0/B1/B2 | TBD | TBD | NOT RUN |
| KPI-COLOR-01 | ΔE00 | ΔE00 + colour space/white point | Report | TBD | TBD | NOT RUN |
| KPI-PERC-01 | LPIPS / justified alternative | unitless | Report | TBD | TBD | NOT RUN |
| KPI-AUD-01 | Audit rank correlation | Spearman rho | K0 target ≥0.6 provisional | TBD | TBD | NOT RUN |
| KPI-FDR-01 | Realized FDR | ratio/proportion | K2 within 2× nominal provisional | TBD | TBD | NOT RUN |
| KPI-EFF-01 | End-to-end latency | ms/image + device | Compare all baselines | TBD | TBD | NOT RUN |
| KPI-EFF-02 | Peak memory | MB/MiB + method | Report | TBD | TBD | NOT RUN |
| KPI-EFF-03 | Model size | MB + parameters | Report | TBD | TBD | NOT RUN |
| KPI-EFF-04 | Refined tile fraction | count and % | Budget sweep | TBD | TBD | NOT RUN |
| KPI-OPT-01 | Optimization trade-off | quality/latency/memory/size | Before/after | TBD | TBD | NOT RUN |
| KPI-DEP-01 | Hosted app reliability | pass/fail checks | All deployment checks pass | TBD | TBD | NOT RUN |

## 6. Copyable run record
```markdown
## Run: PRISM-{EXPERIMENT}-{YYYYMMDD}-{NNN}
- Hypothesis:
- Date/time / operator:
- Git commit + clean/dirty:
- OS / Python / framework / libraries:
- Device / runtime / precision:
- Dataset/version/license:
- Manifest hash:
- Capture/scene/camera split:
- ISO / CFA / bit depth:
- Config path + hash:
- Seed / deterministic settings:
- Model/checkpoint hash / parameter count:
- Exact command:
- Artifacts/logs:
- Metrics (value, unit, aggregation, n):
- Pre-registered gate:
- Decision: PASS / FAIL / INCONCLUSIVE
- Deviations:
- Interpretation / limitations:
- Next action:
```

Never overwrite prior runs. Append new results. Unknown values remain TBD. Preserve failed runs.
