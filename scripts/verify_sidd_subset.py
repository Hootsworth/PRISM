#!/usr/bin/env python3
"""Comprehensive verification and audit of the downloaded SIDD-CC subset."""

import hashlib
import json
import os
from pathlib import Path
import scipy.io
import numpy as np

DATA_ROOT = Path("data/external/sidd_cc")

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def inspect_all():
    print("=== Scanning all files in data/external/sidd_cc ===")
    all_files = sorted([p for p in DATA_ROOT.rglob("*") if p.is_file()])
    print(f"Total files found on disk: {len(all_files)}")

    file_records = []
    total_disk_bytes = 0

    for p in all_files:
        rel = p.relative_to(DATA_ROOT).as_posix()
        sz = p.stat().st_size
        total_disk_bytes += sz
        h = sha256_file(p)
        file_records.append({
            "path": rel,
            "size": sz,
            "sha256": h,
        })
        print(f"{rel}: {sz} bytes ({sz/(1024*1024):.2f} MB), sha256: {h[:12]}...")

    print(f"\nTotal disk bytes: {total_disk_bytes} ({total_disk_bytes / (1024*1024):.2f} MB / {total_disk_bytes / (1024**3):.3f} GB)")

    # Group into scenes
    scenes_dir = DATA_ROOT / "Raw" / "train" / "Data"
    scene_folders = sorted([d for d in scenes_dir.iterdir() if d.is_dir()])
    print(f"\nDiscovered {len(scene_folders)} scenes in {scenes_dir}:")

    scene_audits = []

    for sdir in scene_folders:
        sname = sdir.name
        print(f"\n--- Auditing Scene: {sname} ---")
        parts = sname.split("_")
        inst_id, scene_idx, cam, iso, shutter, temp, bright = parts
        
        # Files in this scene
        mat_files = sorted(list(sdir.glob("*.MAT")))
        print(f"MAT files in scene ({len(mat_files)}): {[f.name for f in mat_files]}")

        # Find GT, NOISY, METADATA
        gt_files = sorted([f for f in mat_files if "_GT_RAW_" in f.name])
        noisy_files = sorted([f for f in mat_files if "_NOISY_RAW_" in f.name])
        meta_files = sorted([f for f in mat_files if "_METADATA_RAW_" in f.name])

        audit = {
            "scene": sname,
            "camera": cam,
            "iso": int(iso),
            "shutter": shutter,
            "color_temp_k": int(temp),
            "brightness_code": bright,
            "gt_files": [f.name for f in gt_files],
            "noisy_files": [f.name for f in noisy_files],
            "meta_files": [f.name for f in meta_files],
            "pairs": [],
        }

        # Inspect each pair
        for gt_f in gt_files:
            idx = gt_f.name.split("_")[-1].replace(".MAT", "")
            noisy_f = sdir / gt_f.name.replace("_GT_RAW_", "_NOISY_RAW_")
            meta_f = sdir / gt_f.name.replace("_GT_RAW_", "_METADATA_RAW_")

            pair_info = {
                "idx": idx,
                "gt_name": gt_f.name,
                "noisy_name": noisy_f.name if noisy_f.exists() else None,
                "meta_name": meta_f.name if meta_f.exists() else None,
            }

            # Inspect MAT contents
            gt_mat = scipy.io.loadmat(gt_f)
            noisy_mat = scipy.io.loadmat(noisy_f)
            meta_mat = scipy.io.loadmat(meta_f)

            gt_keys = [k for k in gt_mat.keys() if not k.startswith("__")]
            noisy_keys = [k for k in noisy_mat.keys() if not k.startswith("__")]
            meta_keys = [k for k in meta_mat.keys() if not k.startswith("__")]

            pair_info["gt_keys"] = gt_keys
            pair_info["noisy_keys"] = noisy_keys
            pair_info["meta_keys"] = meta_keys

            gt_data = gt_mat["data"]
            noisy_data = noisy_mat["data"]

            pair_info["gt_shape"] = list(gt_data.shape)
            pair_info["gt_dtype"] = str(gt_data.dtype)
            pair_info["gt_min"] = float(gt_data.min())
            pair_info["gt_max"] = float(gt_data.max())
            pair_info["gt_mean"] = float(gt_data.mean())
            pair_info["gt_has_negative"] = bool(np.any(gt_data < 0))

            pair_info["noisy_shape"] = list(noisy_data.shape)
            pair_info["noisy_dtype"] = str(noisy_data.dtype)
            pair_info["noisy_min"] = float(noisy_data.min())
            pair_info["noisy_max"] = float(noisy_data.max())
            pair_info["noisy_mean"] = float(noisy_data.mean())
            pair_info["noisy_has_negative"] = bool(np.any(noisy_data < 0))

            # Channel / mosaic dimension check
            # Bayer mosaic in SIDD is 2D: (Height, Width) single plane
            pair_info["is_2d_single_plane"] = (len(gt_data.shape) == 2)
            
            # Inspect metadata fields
            meta_parsed = {}
            for k in meta_keys:
                v = meta_mat[k]
                if isinstance(v, np.ndarray):
                    if v.size == 1:
                        val = v.item()
                        if isinstance(val, bytes):
                            val = val.decode("utf-8", errors="replace")
                        meta_parsed[k] = val
                    elif v.dtype.kind in ('U', 'S'):
                        meta_parsed[k] = str(v.tolist())
                    else:
                        meta_parsed[k] = v.tolist()
                else:
                    meta_parsed[k] = str(v)

            pair_info["metadata"] = meta_parsed
            print(f"  Pair idx={idx}:")
            print(f"    GT shape={gt_data.shape}, dtype={gt_data.dtype}, min={gt_data.min():.5f}, max={gt_data.max():.5f}, mean={gt_data.mean():.5f}, negative={pair_info['gt_has_negative']}")
            print(f"    NOISY shape={noisy_data.shape}, dtype={noisy_data.dtype}, min={noisy_data.min():.5f}, max={noisy_data.max():.5f}, mean={noisy_data.mean():.5f}, negative={pair_info['noisy_has_negative']}")
            cfa = meta_parsed.get("cfa_str", "UNKNOWN")
            bl = meta_parsed.get("BlackLevel", "UNKNOWN")
            wl = meta_parsed.get("WhiteLevel", "UNKNOWN")
            print(f"    Metadata: CFA={cfa}, BlackLevel={bl}, WhiteLevel={wl}")
            print(f"    AsShotNeutral={meta_parsed.get('AsShotNeutral')}")

            audit["pairs"].append(pair_info)

        scene_audits.append(audit)

    with open("scratch_verification_report.json", "w") as f:
        json.dump({"files": file_records, "scenes": scene_audits, "total_bytes": total_disk_bytes}, f, indent=2)
    print("\nVerification report saved to scratch_verification_report.json")

if __name__ == "__main__":
    inspect_all()
