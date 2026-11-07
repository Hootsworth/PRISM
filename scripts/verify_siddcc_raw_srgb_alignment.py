#!/usr/bin/env python3
"""Empirically check RAW-to-rendered-SRGB registration on every train pair.

Uses a center RAW crop with even CFA phase, runs the frozen B0 transform on
that crop, and compares edge maps against the matching noisy sRGB crop over
integer offsets. This is a registration diagnostic, not a quality score.
"""
from __future__ import annotations

import csv
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import sobel
from scipy.stats import pearsonr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from prism.isp import load_sidd_cc_mat, process_bayer

MANIFEST = ROOT / "data/manifests/sidd_cc_expanded_pair_inventory.csv"
OUT = ROOT / "data/manifests/sidd_cc_expanded_alignment_results.csv"
RADIUS = 3
CROP = 512
INTERIOR = 384


def edge(gray: np.ndarray) -> np.ndarray:
    return np.hypot(sobel(gray, axis=0, mode="reflect"), sobel(gray, axis=1, mode="reflect"))


def correlation_at(a: np.ndarray, b: np.ndarray, dy: int, dx: int) -> float:
    h, w = a.shape
    ay = slice(max(0, -dy), h if dy <= 0 else h - dy)
    by = slice(max(0, dy), h if dy >= 0 else h + dy)
    ax = slice(max(0, -dx), w if dx <= 0 else w - dx)
    bx = slice(max(0, dx), w if dx >= 0 else w + dx)
    return float(pearsonr(a[ay, ax].ravel(), b[by, bx].ravel()).statistic)


def main() -> int:
    rows = list(csv.DictReader(MANIFEST.open(newline="", encoding="utf-8")))
    results = []
    for i, row in enumerate(rows, 1):
        raw_path = ROOT / row["raw_noisy"]
        target_path = ROOT / row["srgb_noisy"]
        frame = load_sidd_cc_mat(raw_path)
        h, w = frame.mosaic.shape
        y0 = ((h - CROP) // 2) // 2 * 2
        x0 = ((w - CROP) // 2) // 2 * 2
        crop = frame.mosaic[y0:y0 + CROP, x0:x0 + CROP]
        crop_frame = replace(frame, mosaic=crop)
        predicted = process_bayer(crop_frame, tone_map="reinhard")
        pred = np.asarray(predicted.srgb_encoded, dtype=np.float32).mean(axis=2)
        with Image.open(target_path) as image:
            target = np.asarray(image.crop((x0, y0, x0 + CROP, y0 + CROP)), dtype=np.float32).mean(axis=2)
        # Compare the central area so CFA interpolation and crop boundaries do not dominate.
        a = edge(pred)[(CROP - INTERIOR)//2:(CROP + INTERIOR)//2,
                       (CROP - INTERIOR)//2:(CROP + INTERIOR)//2]
        b = edge(target)[(CROP - INTERIOR)//2:(CROP + INTERIOR)//2,
                         (CROP - INTERIOR)//2:(CROP + INTERIOR)//2]
        candidates = [(correlation_at(a, b, dy, dx), dy, dx)
                      for dy in range(-RADIUS, RADIUS + 1)
                      for dx in range(-RADIUS, RADIUS + 1)]
        best = max(candidates)
        zero = next(score for score, dy, dx in candidates if dy == 0 and dx == 0)
        results.append({
            "instance_id": row["instance_id"], "pair_index": row["pair_index"],
            "offset_y_pixels": best[1], "offset_x_pixels": best[2],
            "best_edge_correlation": best[0], "zero_offset_edge_correlation": zero,
            "margin_over_zero": best[0] - zero,
            "search_radius_pixels": RADIUS,
            "method": "B0 rendered noisy RAW vs camera-rendered noisy sRGB; center edge-map cross-correlation",
        })
        if i % 10 == 0 or i == len(rows):
            print("checked {}/{} pairs".format(i, len(rows)), flush=True)
        del frame, crop_frame, predicted, pred, target, a, b
    with OUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0]))
        writer.writeheader(); writer.writerows(results)
    offsets = [(r["instance_id"], r["pair_index"], r["offset_y_pixels"], r["offset_x_pixels"])
               for r in results if r["offset_y_pixels"] != 0 or r["offset_x_pixels"] != 0]
    print("wrote {}".format(OUT.relative_to(ROOT)))
    print("zero-offset best: {}/{}; nonzero offsets: {}".format(len(results)-len(offsets), len(results), len(offsets)))
    if offsets:
        print("nonzero examples:", offsets[:20])
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
