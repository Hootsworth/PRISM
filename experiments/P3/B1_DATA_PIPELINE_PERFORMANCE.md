# B1.1 Data Pipeline Performance and Memory

**Host:** macOS 27.0.1 arm64, Python 3.9.6, PyTorch 2.8.0. CPU data path; no model/GPU kernel was run.

## End-to-end verification pass

Command:

```bash
PYTHONPATH=src python3 scripts/verify_b1_data_pipeline.py --all-samples
```

The command loaded and checked 230 full-resolution train/validation/internal pairs and all 928 official blocks, including first-use SHA-256 verification of referenced sample files, packing, unpacking, tensor conversion and four representative visualizations.

| Measure | Observed |
|---|---:|
| Total elapsed | 74.243 s |
| Process peak RSS | 1,902,313,472 bytes (about 1.77 GiB) |
| Verification rate | 1,158 sample/block tensors / 74.243 s (not model throughput) |
| Train view | 256×256 crop, batch size 8, 0 workers by default |
| Evaluation | full-resolution images or 256×256 official blocks, batch size 1 by default |

Peak RSS includes the Python/PyTorch process, full-frame conversion buffers and lazily cached official RAW/RGB test tensors. A separate full source-manifest checksum pass took 13.121 s for 1,163 files / 33,262,165,122 bytes (`PYTHONPATH=src python3 scripts/verify_b1_source_hashes.py`), using streaming reads. Combined with the 74.243 s pair/block tensor pass, these two audits took 87.364 s (separate commands). The profile is one machine/run and is not a model-training benchmark.

## Stage profile

`PYTHONPATH=src python3 scripts/profile_b1_data_pipeline.py` measured three repeats for the selected 3120×4208 validation sample after filesystem/library warm-up. These are cache-hot decode timings; complete checksum verification is separately included in the 74.243 s audit above.

| Stage | Median | Mean | Repeats |
|---|---:|---:|---:|
| MAT RAW decode | 4.709 ms | 5.629 ms | 3 |
| PNG RGB decode | 113.077 ms | 113.214 ms | 3 |
| Bayer packing | 8.973 ms | 9.488 ms | 3 |
| Tensor conversion | 45.613 ms | 48.493 ms | 3 |
| Full-resolution dataset item | 189.965 ms | 189.680 ms | 3 |
| One train batch, 8 paired 256 crops | 1,319.778 ms | n=1 | 0 workers |

Stage profile peak process RSS was 1,015,136,256 bytes. The PNG decode and target tensor conversion dominate this sample's warm data path; the one-batch measurement includes full-frame file reads and crop/augmentation work, not neural-network time. The all-sample audit's first-use SHA checks make it unsuitable as a steady-state epoch throughput estimate.

## Defaults and limits

Default train loading uses 256×256 sensor-space crops, batch size 8, 0 workers and no pinned memory unless CUDA is available. A target batch occupies about 6 MiB and packed RAW about 2 MiB, excluding framework overhead; the loader releases each source full frame after crop/tensor construction. Full-resolution evaluation defaults to batch 1. Worker count, pinning and persistent workers are configurable, but increasing workers can duplicate full-frame decode buffers and test tensor caches; measure RSS before raising worker count. These are CPU-side observed values and estimates, not a training-memory result.

`experiments/P3/B1_DATA_PIPELINE_PROFILE.json` contains raw measurements and environment details. The source-hash verifier is `scripts/verify_b1_source_hashes.py`. No neural model, optimizer, checkpoint or training run was created.
