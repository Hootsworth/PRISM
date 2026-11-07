# Application, Deployment and Submission Contract

## 1. Required public application
The app must allow a judge to:
1. Open a public URL from phone or laptop.
2. Upload a supported RAW/DNG file (and optionally supported image formats with clear mode labels).
3. See validation results, including detected CFA/camera/ISO and unsupported metadata warnings.
4. Choose classical B0, learned B1 or hybrid B2.
5. Set hybrid compute/tile budget.
6. View input, output, before/after and zoomed crops.
7. View intermediate outputs (demosaic, denoised, linear RGB, rendered RGB).
8. View PRISM audit heatmap and tile states for B2.
9. See measured processing latency, refined tile count, model size and relevant metrics when references exist.
10. Download output image and a machine-readable / human-readable run report.
11. Receive clear loading, validation, error and unsupported-file messages.

Never display fabricated metrics. If the uploaded image has no reference target, show only metrics that can genuinely be computed and label them correctly.

## 2. Deployment acceptance
- Public URL loads in a clean browser session.
- Test on both phone and laptop.
- Upload, processing, display and download work.
- Long-running processing exposes progress.
- Error states are readable.
- No secrets or dataset credentials in Git.
- Repository can be run with one command or Dockerfile.
- Provide a small permitted demo sample or clear upload instructions.
- Do not claim phone-grade real-time performance unless measured on a phone.

## 3. GitHub repository must include
- Complete source code and tests.
- `README.md` with setup, data access instructions, run commands and result reproduction.
- `docs/00_P2_REQUIREMENTS_TRACEABILITY.md`.
- Architecture diagram matching code.
- Design document.
- Experiment configs and dataset manifests (not copyrighted image data).
- `docs/PROJECT_LEDGER.md`.
- Evaluation scripts and outputs.
- Ablation study.
- Failure-case analysis.
- `eval_P2_TeamID.md`.
- License and third-party attribution notices.

## 4. Design document sections
1. Problem statement (quote/paraphrase P2 faithfully).
2. Objectives and scope.
3. Dataset and data representation.
4. System architecture.
5. Classical ISP.
6. Learned ISP.
7. PRISM audit and adaptive hybrid logic.
8. Training and evaluation protocol.
9. Results and baseline comparison.
10. Optimization and efficiency.
11. Ablations.
12. Failure cases, limitations and ethics.
13. Deployment and reproducibility.

## 5. Three-minute demo plan
- 0:00–0:25 — State P2 problem and show upload.
- 0:25–0:55 — Run classical B0 and show intermediate stages.
- 0:55–1:25 — Run learned B1 and compare.
- 1:25–2:05 — Run PRISM B2; show audit heatmap, tile selection and budget.
- 2:05–2:35 — Show measured metrics/latency/optimization result.
- 2:35–3:00 — Show one failure case and explain the limitation honestly.

Prepare a backup local recording and one known-good sample in case the network or live host fails. Still demonstrate the live URL if possible.

## 6. Viva readiness
Every team member should be able to explain:
- Bayer CFA and demosaicing.
- Black/white levels, noise model and ISO dependence.
- WB, CCM, tone mapping and gamma.
- Classical vs learned vs hybrid pipeline.
- Dataset representation and split leakage prevention.
- Why sensor consistency is not RGB correctness.
- How audit thresholds and FDR were calibrated.
- What optimization was applied and its measured trade-off.
- One successful case and one failure case.
- Their own code contribution.

## 7. Final release checklist
- [ ] Public URL tested from phone and laptop.
- [ ] Supported unseen input processed successfully.
- [ ] GitHub clean clone works.
- [ ] One-command setup or Dockerfile works.
- [ ] Architecture diagram matches current implementation.
- [ ] Design document complete.
- [ ] `eval_P2_TeamID.md` exists and first line contains comma-separated USNs.
- [ ] B0/B1/B2 results on same held-out test samples.
- [ ] All P2 metrics reported or explicitly unavailable with reason.
- [ ] At least one optimization benchmarked.
- [ ] Ablation study complete.
- [ ] At least three failure cases documented.
- [ ] Three-minute demo rehearsed.
- [ ] Every member prepared for viva.
