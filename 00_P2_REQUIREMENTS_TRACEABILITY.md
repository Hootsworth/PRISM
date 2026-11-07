# P2 Requirements Traceability Matrix

This is the central compliance document. Every row must point to a code module, a test, and an evidence artifact. A feature is not complete merely because a prompt or design document mentions it.

**Source:** Computer Vision Hackathon 2026 PDF supplied for this project, Problem Statement 2 (P2), pages 3–4 and datasheet page 14; general submission instructions and rubric, pages 15–16.

## A. Mandatory technical requirements

| Assignment requirement | PRISM implementation | Code location (planned) | Acceptance evidence |
|---|---|---|---|
| Bayer RAW preprocessing and demosaicing, when Bayer RAW supplied | Validated DNG/Bayer loader, black/white level correction, CFA packing, demosaicing | `src/prism/raw/`, `src/prism/isp/demosaic.py` | CFA unit tests; sample DNG run; intermediate Bayer/demosaic exports |
| White balance correction | Metadata-based as-shot WB plus documented fallback estimator | `src/prism/isp/white_balance.py` | WB gains in run report; colour chart / image examples |
| Colour correction matrix | Camera-specific CCM where metadata supports it; calibrated fallback otherwise | `src/prism/isp/color.py` | Matrix provenance; Delta E report |
| Denoising and edge-preserving filtering | Classical denoising baseline plus edge-preserving filter; learned denoiser/ISP path | `src/prism/isp/denoise.py`, `src/prism/learned/` | Noise reduction and edge preservation metrics |
| Gamma correction and tone mapping | Explicit linear-to-display rendering stages | `src/prism/isp/render.py` | Linear and sRGB outputs; no double gamma test |
| Image sharpening and colour-space transformations | Controlled sharpening and RGB/XYZ/Lab transforms | `src/prism/isp/render.py`, `evaluation/color.py` | Sharpness/edge metrics; colour transform tests |
| Implement/adapt learned ISP (e.g. PyNET/U-Net) | Full-frame packed-Bayer-to-RGB U-Net baseline, or verified compatible learned ISP adaptation | `src/prism/learned/isp_unet.py` | Trained checkpoint; training manifest; inference on held-out RAW |
| Model adaptation/fine-tuning using suitable dataset | Train/fine-tune only on verified aligned input-target pairs | `src/prism/learned/train.py` | Split manifest, config, checkpoint hash, training curves |
| Learn image restoration and colour transformation | Learned model outputs display RGB or linear RGB with explicit rendering contract | `src/prism/learned/` | PSNR/SSIM/Delta E and visual examples |
| Robustness to noise and illumination variation | Stratified tests by ISO/noise level and illumination metadata; adaptive conditioning | `src/prism/noise/`, `src/prism/learned/` | Per-condition metrics and failure gallery |
| Integrate traditional stages with learned components | PRISM hybrid: classical fast path + learned tile refiner selected by calibrated audit | `src/prism/isp/`, `audit/`, `scheduler/`, `refinement/` | Hybrid pipeline run and ablation |
| Optimized processing sequence | Audit in linear sensor space; final WB/CCM/tone/sharpen after audit | `src/prism/pipeline.py` | Pipeline trace and stage timings |
| Avoid colour distortion and fine-detail loss | Colour and edge constraints, paired-reference evaluation | `evaluation/` | Delta E, edge metric, crops, failure cases |
| Compare traditional, learned and hybrid ISP | B0 classical, B1 full-frame learned, B2 PRISM hybrid | `src/prism/evaluation/baselines.py` | One comparison table on same held-out images |

## B. Advanced challenge requirements

| Advanced requirement | Implementation / experiment | Required proof |
|---|---|---|
| Adaptive pipeline for sensor noise and illumination | Noise-profile conditioning, illumination descriptors, adaptive denoising/refinement budget | Stratified results across ISO/brightness; fixed-vs-adaptive ablation |
| RAW-to-RGB reconstruction wherever Bayer RAW available | DNG/Bayer input path; preserve CFA and metadata | Input metadata audit and RAW-to-RGB output |
| At least one optimization: quantization, pruning, distillation or reduced precision | INT8 quantization or FP16 reduced precision; choose based on supported hardware | Before/after quality, latency, memory, model size |
| Measure model size, memory and inference latency | Benchmark script with warm-up and repeated runs | Named device/runtime and median latency, peak memory, serialized size |
| Show degradation/recovery under challenging conditions | Low-light/high-ISO, uneven illumination, clipped/near-black, noise mismatch | Condition-wise tables and failure gallery |
| Compare with conventional ISP baseline | Classical B0 plus learned B1 and hybrid B2 | Same test images and evaluation protocol |

## C. Evaluation metrics required by P2

| Metric in assignment | PRISM reporting rule | Artifact |
|---|---|---|
| PSNR | Report in dB, specify linear RAW or display RGB domain | `artifacts/tables/quality_metrics.csv` |
| SSIM | Report with data range and image domain | same |
| Delta E | Use ΔE00; state colour space, white point and conversion pipeline | `artifacts/tables/color_metrics.csv` |
| LPIPS or appropriate perceptual metric | LPIPS only if dependency/model is available; otherwise justify a suitable alternative | `artifacts/tables/perceptual_metrics.csv` |
| Model parameters and model size | Parameter count and serialized checkpoint MB | `artifacts/tables/efficiency_metrics.csv` |
| Inference latency | Median and spread, warm-up, batch size, device, runtime; include full pipeline and audit overhead | same |
| Memory consumption | Peak CPU RAM and/or accelerator memory; describe measurement method | same |

## D. Mandatory application output and submission

| Required by supplied instruction sheet | PRISM deliverable | Completion check |
|---|---|---|
| Live hosted application URL | Public web app, tested from phone and laptop | URL opens without login/permission issue; unseen image upload works |
| GitHub repository with complete source code | Public or judge-accessible repository | Clean clone and one-command run / Dockerfile |
| System architecture diagram | Diagram matching implemented code | `docs/architecture.png` or Mermaid source |
| Design document | Problem, objectives, method, architecture, data, experiments, results, limitations, ethics | `docs/design_document.md` |
| Comparative evaluation file | Exact filename `eval_P2_TeamID.md` (replace TeamID with assigned team identifier); first line is all team USNs comma-separated | File exists and format checked |
| Ablation study | Remove/disable one component at a time | Ablation table and interpretation |
| Failure-case analysis | At least three failure cases, including PRISM audit blind spot | `docs/failure_analysis.md` and visual gallery |
| Working demo | Three-minute demo on live URL, including judge unseen input | Rehearsed demo script and backup local run |
| Every member explains their part | Individual role / explanation sheet | Viva rehearsal checklist complete |

**Filename caution:** The instruction sheet's example says `eval_P1_T01` while the project is P2. Follow the general naming pattern but use `eval_P2_TeamID.md`, substituting the actual assigned team ID. Confirm the exact team-ID convention with the organizers if it is ambiguous.

## E. Rubric mapping

| Rubric category | How PRISM earns evidence |
|---|---|
| Evaluation Metrics (25) | Correct metrics, fair split, baseline table, measured efficiency, condition-wise performance |
| Implementation (20) | End-to-end unseen RAW/image inputs, modular code, train/validation split, robust invalid-input handling |
| UI/UX and deployment (15) | Public URL works on first click from phone/laptop, clear loading/error states, responsive UI, one-command repo/Docker |
| Project documentation (15) | Architecture diagram, complete design doc, README reproduces results, limitations and ethics |
| Demo and viva (25) | Reliable three-minute demo, unseen input, clear explanation by all members, metrics and failure case |

## F. Final sign-off
- [ ] Every row in A–D has a code path and an evidence artifact.
- [ ] No metric is reported without a run ID, split and sample count.
- [ ] Three ISP approaches are evaluated on the same held-out set.
- [ ] At least one optimization is measured, not merely implemented.
- [ ] Public URL and GitHub repository are tested from a clean session.
- [ ] Three failure cases are documented.
- [ ] `eval_P2_TeamID.md` first line is verified.
- [ ] Demo and viva rehearsal completed.
