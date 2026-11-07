#!/usr/bin/env python3
"""Compare optimized B0 against retained original arithmetic on five scenes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from prism.isp import load_sidd_cc_mat, process_bayer, process_bayer_reference  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/external/sidd_cc/Raw/train/Data")
    parser.add_argument("--reference-output-root", type=Path, default=ROOT / "experiments/P2/outputs")
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/P2/b0_regression_comparison.json")
    args = parser.parse_args()
    scenes = sorted(p for p in args.data_root.iterdir() if p.is_dir())
    if len(scenes) != 5:
        parser.error("expected five audited scenes, found {}".format(len(scenes)))
    rows = []
    for folder in scenes:
        sample = sorted(folder.glob("*_NOISY_RAW_*.MAT"))[0]
        frame = load_sidd_cc_mat(sample)
        reference = process_bayer_reference(frame)
        artifact_path = args.reference_output_root / folder.name / "linear_srgb.npy"
        artifact = np.load(artifact_path, mmap_mode="r", allow_pickle=False)
        if artifact.shape != reference.linear_srgb.shape:
            raise RuntimeError("reference artifact dimensions differ for {}".format(folder.name))
        artifact_diff = np.abs(artifact - reference.linear_srgb)
        optimized = process_bayer(frame)
        if optimized.linear_srgb.shape != reference.linear_srgb.shape:
            raise RuntimeError("optimized output dimensions differ for {}".format(folder.name))
        if not np.isfinite(optimized.linear_srgb).all():
            raise RuntimeError("optimized output contains non-finite values for {}".format(folder.name))
        diff = np.abs(optimized.linear_srgb - reference.linear_srgb)
        # This gate detects material changes while allowing documented float32
        # reassociation caused by the fused matrix evaluation.
        np.testing.assert_allclose(optimized.linear_srgb, reference.linear_srgb,
                                   rtol=2e-6, atol=2e-6)
        row = {
            "scene": folder.name,
            "sample": sample.stem,
            "shape": list(optimized.linear_srgb.shape),
            "stored_reference_artifact_max_abs": float(np.max(artifact_diff)),
            "optimized_vs_reference_max_abs": float(np.max(diff)),
            "optimized_vs_reference_rmse": float(np.sqrt(np.mean(diff.astype(np.float64) ** 2))),
            "camera_rgb_max_abs": float(np.max(np.abs(optimized.camera_rgb_linear - reference.camera_rgb_linear))),
            "xyz_d50_max_abs": float(np.max(np.abs(optimized.xyz_d50 - reference.xyz_d50))),
            "negative_linear_count_reference": int(np.count_nonzero(reference.linear_srgb < 0)),
            "negative_linear_count_optimized": int(np.count_nonzero(optimized.linear_srgb < 0)),
            "finite": True,
            "pass": True,
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
        del artifact, frame, reference, optimized
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print("PASS: {} scenes; max absolute RGB difference {:.9g}".format(
        len(rows), max(row["optimized_vs_reference_max_abs"] for row in rows)))


if __name__ == "__main__":
    main()
