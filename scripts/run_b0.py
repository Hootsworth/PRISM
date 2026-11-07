#!/usr/bin/env python3
"""Run B0 on the first verified noisy Bayer sample from every local scene."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from prism.isp import load_sidd_cc_mat, process_bayer  # noqa: E402
from prism.isp.render import to_uint8_srgb  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def max_rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes; Linux/BSD generally report KiB.
    return int(value if sys.platform == "darwin" else value * 1024)


def scene_samples(data_root: Path) -> list[Path]:
    folders = sorted(p for p in data_root.iterdir() if p.is_dir())
    if not folders:
        raise SystemExit("no scene directories found under {}".format(data_root))
    selected = []
    for folder in folders:
        candidates = sorted(folder.glob("*_NOISY_RAW_*.MAT"))
        if not candidates:
            raise SystemExit("scene has no noisy RAW samples: {}".format(folder))
        selected.append(candidates[0])
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/external/sidd_cc/Raw/train/Data")
    parser.add_argument("--output-root", type=Path, default=ROOT / "experiments/P2/outputs")
    parser.add_argument("--tone-map", choices=["reinhard", "none"], default="reinhard")
    args = parser.parse_args()
    samples = scene_samples(args.data_root)
    if len(samples) != 5:
        raise SystemExit("expected the verified five-scene P1 subset, found {} scenes".format(len(samples)))

    args.output_root.mkdir(parents=True, exist_ok=True)
    records = []
    for input_path in samples:
        scene_id = input_path.parent.name
        out_dir = args.output_root / scene_id
        out_dir.mkdir(parents=True, exist_ok=True)
        meta_path = input_path.with_name(input_path.name.replace("_NOISY_RAW_", "_METADATA_RAW_"))
        t0 = time.perf_counter()
        load_t0 = time.perf_counter()
        frame = load_sidd_cc_mat(input_path)
        load_ms = (time.perf_counter() - load_t0) * 1000
        processing_t0 = time.perf_counter()
        result = process_bayer(frame, None if args.tone_map == "none" else args.tone_map)
        process_ms = (time.perf_counter() - processing_t0) * 1000

        # Persist full-precision float32 stage arrays without display clipping.
        np.save(out_dir / "demosaiced_camera_rgb_linear.npy", result.camera_rgb_linear, allow_pickle=False)
        np.save(out_dir / "xyz_d50_linear.npy", result.xyz_d50, allow_pickle=False)
        np.save(out_dir / "linear_srgb.npy", result.linear_srgb, allow_pickle=False)
        image = Image.fromarray(to_uint8_srgb(result.srgb_encoded))
        image.save(out_dir / "rendered_srgb.png", optimize=False)
        metadata_record = {
            "sample_id": input_path.stem,
            "scene_id": scene_id,
            "input_path": str(input_path.relative_to(ROOT)),
            "input_sha256": sha256(input_path),
            "metadata_path": str(meta_path.relative_to(ROOT)),
            "metadata_sha256": sha256(meta_path),
            "input_dtype": str(frame.mosaic.dtype),
            "input_shape_hw": list(frame.mosaic.shape),
            "input_min": float(np.min(frame.mosaic)),
            "input_max": float(np.max(frame.mosaic)),
            "input_negative_sample_count": int(np.count_nonzero(frame.mosaic < 0)),
            "normalized_white_threshold": 1.0,
            "saturation_candidate_sample_count_ge_1": frame.saturation_candidate_count,
            "saturation_candidate_note": "candidate only; per-pixel sensor clipping flags are unavailable; samples remain unchanged",
            "cfa_pattern": frame.cfa_pattern,
            "original_black_level_provenance_only": frame.black_level_original,
            "original_white_level_provenance_only": frame.white_level_original,
            "normalization_applied_by_b0": False,
            "tone_map": args.tone_map,
            "stages": result.transform.metadata(),
            "outputs": {
                "camera_rgb_linear": "demosaiced_camera_rgb_linear.npy",
                "xyz_d50_linear": "xyz_d50_linear.npy",
                "linear_srgb": "linear_srgb.npy",
                "display_file": "rendered_srgb.png",
                "display_file_clipping": "only at uint8 PNG encoding to [0,1]",
            },
        }
        (out_dir / "run_metadata.json").write_text(json.dumps(metadata_record, indent=2), encoding="utf-8")
        total_ms = (time.perf_counter() - t0) * 1000
        records.append({
            "scene_id": scene_id,
            "sample_id": input_path.stem,
            "height": frame.mosaic.shape[0],
            "width": frame.mosaic.shape[1],
            "cfa": frame.cfa_pattern,
            "load_ms": load_ms,
            "isp_ms": process_ms,
            "end_to_end_ms": total_ms,
            "process_peak_rss_bytes_high_water": max_rss_bytes(),
            "outputs_dir": str(out_dir.relative_to(ROOT)),
            "negative_linear_rgb_count": int(np.count_nonzero(result.linear_srgb < 0)),
            "linear_rgb_min": float(np.min(result.linear_srgb)),
            "linear_rgb_max": float(np.max(result.linear_srgb)),
        })
        print("{} {}x{} {} {:.1f} ms".format(scene_id, frame.mosaic.shape[1], frame.mosaic.shape[0], frame.cfa_pattern, total_ms))
        del result, frame, image
        gc.collect()

    fieldnames = list(records[0].keys())
    with (args.output_root / "b0_performance.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    report = {
        "phase": "P2 B0",
        "scene_count": len(records),
        "processed_images": [r["sample_id"] for r in records],
        "tone_map": args.tone_map,
        "performance_csv": "b0_performance.csv",
        "process_peak_rss_bytes_high_water": max_rss_bytes(),
        "quality_metrics": "NOT COMPUTED: paired sRGB targets are absent from the ingested subset",
        "split_caution": "All five scenes are P1 train captures; this is an implementation smoke/evidence run, not held-out evaluation.",
    }
    (args.output_root / "b0_run_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
