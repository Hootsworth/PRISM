#!/usr/bin/env python3
"""Warm-stage B0 profiler for all five local SIDD-CC scenes.

The `reference` implementation reproduces the original four full-frame matrix
passes. The `optimized` implementation will exercise the fused matrix path.
PNG encoding is included, but large intermediate NPY disk writes are excluded
from this compute-oriented profile and identified as such in the report.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import resource
import statistics
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import psutil
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from prism.isp.color import (  # noqa: E402
    apply_matrix_rgb, apply_matrix_rgb_reference, build_color_transform,
)
from prism.isp.demosaic import bilinear_demosaic  # noqa: E402
from prism.isp.raw import load_sidd_cc_mat  # noqa: E402
from prism.isp.render import encode_srgb, tone_map_reinhard, to_uint8_srgb  # noqa: E402


class RSSSampler:
    """Sample current process RSS and attribute observed peaks to stage labels."""

    def __init__(self, period=0.005):
        self.process = psutil.Process()
        self.period = period
        self.label = "idle"
        self.base = 0
        self.peaks = {}
        self.stop = threading.Event()
        self.thread = None

    def _run(self):
        while not self.stop.is_set():
            rss = self.process.memory_info().rss
            self.peaks[self.label] = max(self.peaks.get(self.label, 0), rss)
            time.sleep(self.period)

    def start(self):
        self.base = self.process.memory_info().rss
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def mark(self, label):
        self.label = label

    def close(self):
        self.stop.set()
        self.thread.join()
        final = self.process.memory_info().rss
        self.peaks[self.label] = max(self.peaks.get(self.label, 0), final)
        return {key: max(0, val - self.base) for key, val in self.peaks.items()}


def stage(sampler, times, name):
    @contextmanager
    def measured():
        sampler.mark(name)
        start = time.perf_counter()
        try:
            yield
        finally:
            times[name] = (time.perf_counter() - start) * 1000
    return measured()


def run_one(path, implementation, tone_map="reinhard"):
    times = {}
    sampler = RSSSampler()
    sampler.start()
    total_start = time.perf_counter()
    retained = {}
    try:
        with stage(sampler, times, "raw_loading"):
            frame = load_sidd_cc_mat(path)
            retained["raw_mosaic"] = int(frame.mosaic.nbytes)
        with stage(sampler, times, "demosaicing"):
            camera_rgb = bilinear_demosaic(frame.mosaic, frame.cfa_pattern)
            retained["camera_rgb_linear"] = int(camera_rgb.nbytes)
        with stage(sampler, times, "wb_transform_setup"):
            transform = build_color_transform(frame.as_shot_neutral, frame.color_matrix2)
        matmul = apply_matrix_rgb_reference if implementation == "reference" else apply_matrix_rgb
        if implementation == "reference":
            with stage(sampler, times, "color_matrix_inverse"):
                xyz = matmul(camera_rgb, transform.camera_to_xyz)
            with stage(sampler, times, "white_balance_bradford"):
                xyz_d50 = matmul(xyz, transform.source_to_d50)
            del xyz
            with stage(sampler, times, "xyz_d50_to_linear_srgb"):
                xyz_d65 = matmul(xyz_d50, transform.d50_to_d65)
                linear_srgb = matmul(xyz_d65, transform.xyz_d65_to_linear_srgb)
            del xyz_d65
        else:
            # The optimized B0 pipeline composes adjacent linear transforms.
            with stage(sampler, times, "camera_to_xyz_d50_fused"):
                camera_to_d50 = transform.source_to_d50 @ transform.camera_to_xyz
                xyz_d50 = matmul(camera_rgb, camera_to_d50)
            with stage(sampler, times, "xyz_d50_to_linear_srgb_fused"):
                d50_to_srgb = transform.xyz_d65_to_linear_srgb @ transform.d50_to_d65
                linear_srgb = matmul(xyz_d50, d50_to_srgb)
        retained["xyz_d50_linear"] = int(xyz_d50.nbytes)
        retained["linear_srgb"] = int(linear_srgb.nbytes)
        with stage(sampler, times, "tone_mapping"):
            display_linear = tone_map_reinhard(linear_srgb) if tone_map == "reinhard" else linear_srgb.copy()
            retained["display_linear_rgb"] = int(display_linear.nbytes)
        with stage(sampler, times, "srgb_oetf"):
            encoded = encode_srgb(display_linear)
            retained["srgb_encoded"] = int(encoded.nbytes)
        with stage(sampler, times, "finite_validation"):
            valid = all(np.isfinite(a).all() for a in (xyz_d50, linear_srgb, encoded))
            if not valid:
                raise ValueError("non-finite output")
        with stage(sampler, times, "png_encoding"):
            buffer = io.BytesIO()
            Image.fromarray(to_uint8_srgb(encoded)).save(buffer, format="PNG", optimize=False)
            png_bytes = buffer.tell()
        times["end_to_end"] = (time.perf_counter() - total_start) * 1000
        # Return camera_rgb as B0Result does; keeping it live through rendering
        # makes the process memory profile match the actual API contract.
        return times, sampler.close(), retained, png_bytes, camera_rgb, xyz_d50, linear_srgb
    finally:
        if sampler.thread and sampler.thread.is_alive():
            sampler.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--implementation", choices=["reference", "optimized"], default="reference")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/external/sidd_cc/Raw/train/Data")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repetitions < 3 or args.warmup < 1:
        parser.error("at least 3 repetitions and 1 warm-up are required")
    scenes = sorted(p for p in args.data_root.iterdir() if p.is_dir())
    if len(scenes) != 5:
        parser.error("expected exactly five audited scenes, found {}".format(len(scenes)))
    records = []
    for folder in scenes:
        sample = sorted(folder.glob("*_NOISY_RAW_*.MAT"))[0]
        print("warm-up {}".format(sample.stem), flush=True)
        for _ in range(args.warmup):
            result = run_one(sample, args.implementation)
            del result
        for rep in range(1, args.repetitions + 1):
            times, memory, retained, png_bytes, camera_rgb, xyz_d50, linear_srgb = run_one(sample, args.implementation)
            records.append({
                "implementation": args.implementation,
                "scene_id": folder.name,
                "sample_id": sample.stem,
                "repetition": rep,
                "height": int(linear_srgb.shape[0]),
                "width": int(linear_srgb.shape[1]),
                "png_bytes": png_bytes,
                "rss_peak_delta_by_stage_bytes": json.dumps(memory, sort_keys=True),
                "retained_array_bytes_by_stage": json.dumps(retained, sort_keys=True),
                **times,
            })
            del camera_rgb, xyz_d50, linear_srgb
            print("{} rep {} end-to-end {:.1f} ms".format(folder.name, rep, times["end_to_end"]), flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    stage_names = [k for k in records[0] if k in {
        "raw_loading", "demosaicing", "wb_transform_setup", "color_matrix_inverse",
        "white_balance_bradford", "xyz_d50_to_linear_srgb", "camera_to_xyz_d50_fused",
        "xyz_d50_to_linear_srgb_fused", "tone_mapping", "srgb_oetf", "finite_validation",
        "png_encoding", "end_to_end",
    }]
    summary = {"implementation": args.implementation, "repetitions_per_scene": args.repetitions,
               "warmups_per_scene": args.warmup, "scenes": {}}
    for folder in scenes:
        group = [r for r in records if r["scene_id"] == folder.name]
        summary["scenes"][folder.name] = {
            name: {"mean_ms": statistics.mean(r[name] for r in group),
                   "median_ms": statistics.median(r[name] for r in group)}
            for name in stage_names
        }
    summary["overall_median_stage_ms"] = {
        name: statistics.median(r[name] for r in records) for name in stage_names
    }
    summary["max_sampled_rss_delta_by_stage_bytes"] = {
        name: max(json.loads(r["rss_peak_delta_by_stage_bytes"]).get(name, 0) for r in records)
        for name in set(k for r in records for k in json.loads(r["rss_peak_delta_by_stage_bytes"]))
    }
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    summary["process_peak_rss_bytes"] = int(peak_rss if sys.platform == "darwin" else peak_rss * 1024)
    summary["overall_mean_stage_ms"] = {
        name: statistics.mean(r[name] for r in records) for name in stage_names
    }
    summary_path = args.output.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"csv": str(args.output), "summary": str(summary_path),
                      "overall_median_stage_ms": summary["overall_median_stage_ms"],
                      "overall_mean_stage_ms": summary["overall_mean_stage_ms"],
                      "process_peak_rss_bytes": summary["process_peak_rss_bytes"],
                      "max_sampled_rss_delta_by_stage_bytes": summary["max_sampled_rss_delta_by_stage_bytes"]}, indent=2))


if __name__ == "__main__":
    main()
