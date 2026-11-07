#!/usr/bin/env python3
"""Audit paired SIDD-CC RAW/SRGB files and freeze instance-level splits.

Run after scripts/acquire_siddcc_expanded.py has completed. This reads the new
expanded copy only; it never writes into data/external/sidd_cc (the P1 subset).
"""
from __future__ import annotations

import csv
import hashlib
import math
import random
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.io import loadmat, whosmat

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/external/sidd_cc_expanded"
OUT = ROOT / "data/manifests"
REVISION = "5eb2ab6375d7f99f36b59bb5aacb46af18409412"
SEED = 20261106
VAL_FRACTION = 0.15


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def pair_stems(directory: Path, modality: str, suffix: str) -> list[str]:
    return sorted(p.stem for p in directory.glob("*_{}_{}.{}".format("NOISY", modality, suffix)))


def assert_close_meta(a: dict, b: dict, name: str) -> None:
    if name == "cfa_str":
        va = str(np.asarray(a[name]).reshape(-1)[0]).strip().upper()
        vb = str(np.asarray(b[name]).reshape(-1)[0]).strip().upper()
        if va != vb:
            raise ValueError("pair metadata CFA mismatch: {} != {}".format(va, vb))
    else:
        if not np.array_equal(np.asarray(a[name]), np.asarray(b[name])):
            raise ValueError("pair metadata {} mismatch".format(name))


def discover_test_rows(modality: Path) -> list[dict[str, str]]:
    path = modality / "test/test_scenes_info.txt"
    with path.open(encoding="utf-8") as f:
        header = f.readline().strip().split("\t")
        rows = []
        for line in f:
            values = line.strip().split("\t")
            if not line.strip():
                continue
            rows.append(dict(zip(header, values)))
        return rows


def main() -> int:
    raw = DATA / "Raw"
    srgb = DATA / "SRGB"
    required = [raw / "README.md", srgb / "README.md", raw / "train/Scene_Instances.txt",
                srgb / "train/Scene_Instances.txt", raw / "test/test_scenes_info.txt",
                srgb / "test/test_scenes_info.txt"]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        print("missing required files:\n" + "\n".join(missing), file=sys.stderr)
        return 2

    raw_instances = lines(raw / "train/Scene_Instances.txt")
    srgb_instances = lines(srgb / "train/Scene_Instances.txt")
    if raw_instances != srgb_instances:
        raise ValueError("RAW and SRGB instance lists differ or are not in the same order")
    if len(raw_instances) != 115:
        raise ValueError("expected 115 train instances; found {}".format(len(raw_instances)))
    alignment_path = OUT / "sidd_cc_expanded_alignment_results.csv"
    alignment_records = {}
    if alignment_path.is_file():
        with alignment_path.open(newline="", encoding="utf-8") as f:
            alignment_records = {(r["instance_id"], r["pair_index"]): r for r in csv.DictReader(f)}

    # 15% validation, stratified by camera using largest-remainder allocation.
    by_camera: dict[str, list[str]] = {}
    for instance in raw_instances:
        fields = instance.split("_")
        if len(fields) != 7:
            raise ValueError("unexpected instance identifier: {}".format(instance))
        by_camera.setdefault(fields[2], []).append(instance)
    val_total = round(len(raw_instances) * VAL_FRACTION)
    quotas = {cam: len(ids) * val_total / len(raw_instances) for cam, ids in by_camera.items()}
    val_counts = {cam: math.floor(q) for cam, q in quotas.items()}
    remaining = val_total - sum(val_counts.values())
    for cam in sorted(by_camera, key=lambda c: (-(quotas[c] - val_counts[c]), c))[:remaining]:
        val_counts[cam] += 1
    rng = random.Random(SEED)
    validation = set()
    for camera, instances in sorted(by_camera.items()):
        validation.update(rng.sample(sorted(instances), val_counts[camera]))
    train = set(raw_instances) - validation

    inventory_rows = []
    split_rows = []
    for instance in raw_instances:
        fields = instance.split("_")
        inst_id, scene_no, camera, iso, shutter, kelvin, brightness = fields
        split = "validation" if instance in validation else "train"
        rdir = raw / "train/Data" / instance
        sdir = srgb / "train/Data" / instance
        if not rdir.is_dir() or not sdir.is_dir():
            raise FileNotFoundError("missing RAW/SRGB directory for {}".format(instance))
        noisy_raw = sorted(rdir.glob("*_NOISY_RAW_*.MAT"))
        if len(noisy_raw) != 2:
            raise ValueError("expected 2 noisy RAW files for {}; found {}".format(instance, len(noisy_raw)))
        suffixes = [p.stem.rsplit("_", 1)[-1] for p in noisy_raw]
        if len(set(suffixes)) != 2:
            raise ValueError("duplicate RAW sample suffix for {}".format(instance))
        noisy_srgb_names = {p.name for p in sdir.glob("*_NOISY_SRGB_*.PNG")}
        gt_srgb_names = {p.name for p in sdir.glob("*_GT_SRGB_*.PNG")}
        if len(noisy_srgb_names) != 2 or len(gt_srgb_names) != 2:
            raise ValueError("expected exactly 2 noisy and GT sRGB files for {}".format(instance))
        for raw_path, suffix in zip(noisy_raw, suffixes):
            stem_prefix = instance.split("_")[0]
            raw_noisy = raw_path
            raw_gt = raw_path.with_name(raw_path.name.replace("_NOISY_RAW_", "_GT_RAW_"))
            raw_meta = raw_path.with_name(raw_path.name.replace("_NOISY_RAW_", "_METADATA_RAW_"))
            srgb_noisy = sdir / ("{}_NOISY_SRGB_{}.PNG".format(stem_prefix, suffix))
            srgb_gt = sdir / ("{}_GT_SRGB_{}.PNG".format(stem_prefix, suffix))
            # The upstream naming convention shares the first-four-digit instance id.
            # Require exactly one matching modality file for each pair index.
            if (raw_gt.name not in {p.name for p in rdir.glob("*_GT_RAW_*.MAT")}
                    or not raw_meta.is_file() or srgb_noisy.name not in noisy_srgb_names
                    or srgb_gt.name not in gt_srgb_names):
                raise FileNotFoundError("incomplete RAW/SRGB pair {} suffix {}".format(instance, suffix))
            noisy_mat = loadmat(str(raw_noisy), variable_names=["data"])
            gt_mat = loadmat(str(raw_gt), variable_names=["data"])
            if "data" not in noisy_mat or "data" not in gt_mat:
                raise ValueError("MAT variable 'data' missing for {} {}".format(instance, suffix))
            noisy = noisy_mat["data"]
            clean = gt_mat["data"]
            if noisy.dtype != np.float32 or clean.dtype != np.float32 or noisy.ndim != 2 or noisy.shape != clean.shape:
                raise ValueError("RAW shape/dtype mismatch for {} {}".format(instance, suffix))
            if not np.isfinite(noisy).all() or not np.isfinite(clean).all():
                raise ValueError("non-finite RAW values for {} {}".format(instance, suffix))
            metadata = loadmat(str(raw_meta))
            required_meta = ("cfa_str", "AsShotNeutral", "ColorMatrix2", "BlackLevel", "WhiteLevel")
            if any(k not in metadata for k in required_meta):
                raise ValueError("required RAW metadata missing for {} {}".format(instance, suffix))
            cfa_value = np.asarray(metadata["cfa_str"]).reshape(-1)[0]
            if isinstance(cfa_value, bytes):
                cfa_value = cfa_value.decode("utf-8", errors="strict")
            cfa = str(cfa_value).strip().upper()
            if cfa not in {"BGGR", "GRBG", "RGGB"}:
                raise ValueError("unsupported or unknown CFA {} for {}".format(cfa, instance))
            neutral = np.asarray(metadata["AsShotNeutral"], dtype=np.float64).reshape(-1)
            matrix = np.asarray(metadata["ColorMatrix2"], dtype=np.float64).reshape(-1)
            if neutral.size != 3 or not np.isfinite(neutral).all() or (neutral <= 0).any():
                raise ValueError("invalid AsShotNeutral for {}".format(instance))
            if matrix.size != 9 or not np.isfinite(matrix).all():
                raise ValueError("invalid ColorMatrix2 for {}".format(instance))
            if np.linalg.cond(matrix.reshape(3, 3)) > 1e8:
                raise ValueError("unstable ColorMatrix2 for {}".format(instance))
            black = np.asarray(metadata["BlackLevel"], dtype=np.float64).reshape(-1)
            white = np.asarray(metadata["WhiteLevel"], dtype=np.float64).reshape(-1)
            if black.size != 1 or white.size != 1 or not np.isfinite(black).all() or not np.isfinite(white).all() or white[0] <= black[0]:
                raise ValueError("invalid BlackLevel/WhiteLevel for {}".format(instance))
            # Do not black-subtract or clip; only validate source values.
            raw_h, raw_w = noisy.shape
            del noisy_mat, gt_mat, noisy, clean
            with Image.open(srgb_noisy) as im_noisy:
                im_noisy.verify()
            with Image.open(srgb_noisy) as im_noisy:
                if im_noisy.mode != "RGB":
                    raise ValueError("NOISY_SRGB must be RGB; got {}".format(im_noisy.mode))
                srgb_w, srgb_h = im_noisy.size
            with Image.open(srgb_gt) as im_gt:
                im_gt.verify()
            with Image.open(srgb_gt) as im_gt:
                if im_gt.mode != "RGB" or im_gt.size != (srgb_w, srgb_h):
                    raise ValueError("SRGB pair mode/dimensions mismatch for {} {}".format(instance, suffix))
            if (srgb_h, srgb_w) != (raw_h, raw_w):
                raise ValueError("RAW/SRGB dimensions mismatch for {} {}: {} vs {}".format(instance, suffix, (raw_h, raw_w), (srgb_h, srgb_w)))
            registration = alignment_records.get((instance, suffix))
            if registration:
                alignment_status = ("empirical integer edge-offset check: best offset ({},{}) pixels within ±{}; verified"
                                    .format(registration["offset_y_pixels"], registration["offset_x_pixels"], registration["search_radius_pixels"]))
            else:
                alignment_status = "same source/index documented; full-resolution dimensions match; empirical edge-offset check pending"
            inventory_rows.append({
                "instance_id": instance, "pair_index": suffix, "split": split,
                "camera": camera, "iso": iso, "shutter": shutter, "illuminant_k": kelvin,
                "brightness": brightness, "cfa": cfa, "height": raw_h, "width": raw_w,
                "raw_noisy": raw_noisy.relative_to(ROOT).as_posix(),
                "raw_gt": raw_gt.relative_to(ROOT).as_posix(),
                "raw_metadata": raw_meta.relative_to(ROOT).as_posix(),
                "srgb_noisy": srgb_noisy.relative_to(ROOT).as_posix(),
                "srgb_gt": srgb_gt.relative_to(ROOT).as_posix(),
                "mapping": "same instance prefix + image index; upstream README states corresponding RAW/SRGB",
                "alignment": alignment_status,
                "alignment_edge_offset_y": registration["offset_y_pixels"] if registration else "",
                "alignment_edge_offset_x": registration["offset_x_pixels"] if registration else "",
                "alignment_edge_corr": registration["best_edge_correlation"] if registration else "",
                "target": "8-bit RGB PNG camera-pipeline rendering",
            })
        split_rows.append({
            "group_id": instance, "split": split, "source_split": "train",
            "camera": camera, "iso": iso, "scene_number": scene_no,
            "illuminant_k": kelvin, "brightness": brightness,
            "pair_count": len(suffixes), "leakage_group": instance,
        })

    raw_test = discover_test_rows(raw)
    srgb_test = discover_test_rows(srgb)
    if raw_test != srgb_test or len(raw_test) != 29:
        raise ValueError("RAW/SRGB test scene records differ or are not 29 instances")
    train_ids = {name.split("_")[0] for name in raw_instances}
    test_ids = {row.get("scene_id", "") for row in raw_test}
    if train_ids & test_ids:
        raise ValueError("official train/test scene ID overlap: {}".format(sorted(train_ids & test_ids)))
    raw_coords = loadmat(str(raw / "test/BenchmarkBlocks32.mat"))["BenchmarkBlocks32"]
    srgb_coords = loadmat(str(srgb / "test/BenchmarkBlocks32.mat"))["BenchmarkBlocks32"]
    if not np.array_equal(raw_coords, srgb_coords):
        raise ValueError("RAW/SRGB test block coordinates differ")
    if len(raw_coords) != 32:
        raise ValueError("expected 32 common test blocks")
    test_rows = []
    test_array_specs = {}
    for modality, names in ((raw, ("BenchmarkNoisyBlocksRaw.mat", "BenchmarkGtBlocksRaw.mat")),
                            (srgb, ("BenchmarkNoisyBlocksSrgb.mat", "BenchmarkGtBlocksSrgb.mat"))):
        for filename in names:
            mat_path = modality / "test" / filename
            variables = {name: (shape, dtype) for name, shape, dtype in whosmat(str(mat_path))}
            if "data" not in variables:
                raise ValueError("test MAT missing data variable: {}".format(mat_path))
            test_array_specs[(modality.name, filename)] = variables["data"]
    expected_raw = (29, 32, 256, 256)
    expected_srgb = (29, 32, 256, 256, 3)
    for filename in ("BenchmarkNoisyBlocksRaw.mat", "BenchmarkGtBlocksRaw.mat"):
        if test_array_specs[("Raw", filename)][0] != expected_raw or test_array_specs[("Raw", filename)][1] != "single":
            raise ValueError("unexpected test RAW block tensor in {}".format(filename))
    for filename in ("BenchmarkNoisyBlocksSrgb.mat", "BenchmarkGtBlocksSrgb.mat"):
        if test_array_specs[("SRGB", filename)][0] != expected_srgb or test_array_specs[("SRGB", filename)][1] != "uint8":
            raise ValueError("unexpected test SRGB block tensor in {}".format(filename))
    for modality, names, expected_shape, expected_dtype in (
            (raw, ("BenchmarkNoisyBlocksRaw.mat", "BenchmarkGtBlocksRaw.mat"), expected_raw, np.float32),
            (srgb, ("BenchmarkNoisyBlocksSrgb.mat", "BenchmarkGtBlocksSrgb.mat"), expected_srgb, np.uint8)):
        for filename in names:
            path = modality / "test" / filename
            array = loadmat(str(path), variable_names=["data"])["data"]
            if array.shape != expected_shape or array.dtype != expected_dtype:
                raise ValueError("test block value array shape/dtype mismatch: {}".format(path))
            if array.dtype.kind == "f" and not np.isfinite(array).all():
                raise ValueError("non-finite test RAW values: {}".format(path))
            del array
    for i, record in enumerate(raw_test):
        scene_id = record.get("scene_id", "")
        test_rows.append({
            "group_id": scene_id, "split": "test", "source_split": "official_test",
            "camera": "unknown in local test metadata", "iso": "unknown in local test metadata",
            "scene_number": scene_id, "illuminant_k": "unknown", "brightness": "unknown",
            "pair_count": 32, "leakage_group": scene_id,
            "source_image": record.get("img_dir", ""), "source_shape": record.get("shape", ""),
        })
    OUT.mkdir(parents=True, exist_ok=True)
    inv_path = OUT / "sidd_cc_expanded_pair_inventory.csv"
    split_path = OUT / "sidd_cc_expanded_split_manifest.csv"
    with inv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(inventory_rows[0]))
        writer.writeheader(); writer.writerows(inventory_rows)
    split_fields = list(split_rows[0]) + [k for k in test_rows[0] if k not in split_rows[0]]
    with split_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=split_fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(split_rows + test_rows)

    file_records = []
    paths = [DATA / "README.md"] + [p for d in (raw, srgb) for p in d.rglob("*") if p.is_file()]
    for path in sorted(paths):
        file_records.append({"path": path.relative_to(ROOT).as_posix(), "size_bytes": path.stat().st_size, "sha256": sha256(path)})
    file_path = OUT / "sidd_cc_expanded_file_manifest.csv"
    with file_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "size_bytes", "sha256"])
        writer.writeheader(); writer.writerows(file_records)

    print("PASS: train instances {} ({} train / {} validation), RAW/SRGB pairs {}, official test scenes {}".format(
        len(raw_instances), len(train), len(validation), len(inventory_rows), len(test_rows)))
    print("RAW/SRGB common test blocks: {} coordinates; split seed {}".format(len(raw_coords), SEED))
    print("pair inventory: {}".format(inv_path.relative_to(ROOT)))
    print("split manifest: {}".format(split_path.relative_to(ROOT)))
    print("file SHA-256 manifest: {} ({} files)".format(file_path.relative_to(ROOT), len(file_records)))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
