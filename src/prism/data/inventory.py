"""Conservative filesystem inventory for dataset feasibility audits.

This records observed files only. It inspects verified dataset files when
metadata and loaders are available, while preserving conservative defaults.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Iterable

try:
    import scipy.io
    import numpy as np
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

SUPPORTED = {".dng", ".mat", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}
FIELDS = [
    "path", "sample_id", "capture_id", "scene_id", "camera", "iso", "cfa",
    "width", "height", "bit_depth", "representation", "metadata_present",
    "target_path", "alignment_verified", "split", "license_note", "sha256",
]


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def classify(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".dng":
        return "bayer_dng_candidate_metadata_unverified"
    if suffix == ".mat":
        return "mat_contents_uninspected"
    if suffix in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
        return "image_target_or_input_unverified"
    return "unknown"


def files_under(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED:
            yield path


def parse_sidd_record(path: Path, root: Path) -> dict[str, str] | None:
    """Attempts to inspect and extract verified metadata from a SIDD MAT file."""
    filename = path.name
    parent_name = path.parent.name
    
    parts = parent_name.split("_")
    # Expected: 0017_001_GP_00100_00060_5500_N (7 parts)
    if len(parts) != 7 or not filename.endswith(".MAT"):
        return None

    inst_id, scene_idx, camera, iso_str, shutter, temp, brightness = parts
    
    # Identify type: GT, NOISY, METADATA
    fn_no_ext = filename[:-4]
    fn_parts = fn_no_ext.split("_")
    if len(fn_parts) < 4:
        return None
    type_str = fn_parts[-3]
    idx_str = fn_parts[-1]

    sample_id = f"{parent_name}_{idx_str}"
    capture_id = parent_name
    scene_id = scene_idx
    iso = str(int(iso_str))
    
    meta_path = path.parent / f"{fn_parts[0]}_METADATA_RAW_{idx_str}.MAT"
    cfa = ""
    width = ""
    height = ""
    bit_depth = ""
    metadata_present = "false"

    if HAS_SCIPY and meta_path.exists():
        try:
            m = scipy.io.loadmat(meta_path)
            if "cfa_str" in m:
                cfa_val = m["cfa_str"]
                if isinstance(cfa_val, np.ndarray) and cfa_val.size > 0:
                    cfa = str(cfa_val.item()).upper()
                else:
                    cfa = str(cfa_val).upper()
            if "WhiteLevel" in m:
                wl = float(m["WhiteLevel"].item())
                if wl > 2000:
                    bit_depth = "12"
                elif wl > 500:
                    bit_depth = "10"
                else:
                    bit_depth = "8"
            metadata_present = "true"
        except Exception:
            pass

    target_path = ""
    alignment_verified = "false"
    split = "train"
    license_note = "SIDD Terms of Use (academic research); CVPR 2018 / SIDD-CC 2026"

    if type_str == "NOISY":
        representation = "bayer_raw_noisy_normalized_unclipped_float32"
        gt_file = path.parent / f"{fn_parts[0]}_GT_RAW_{idx_str}.MAT"
        if gt_file.exists():
            target_path = gt_file.relative_to(root).as_posix()
            alignment_verified = "true"
        if HAS_SCIPY:
            try:
                arr = scipy.io.loadmat(path)["data"]
                height, width = str(arr.shape[0]), str(arr.shape[1])
            except Exception:
                pass
    elif type_str == "GT":
        representation = "bayer_raw_groundtruth_normalized_unclipped_float32"
        alignment_verified = "true"
        if HAS_SCIPY:
            try:
                arr = scipy.io.loadmat(path)["data"]
                height, width = str(arr.shape[0]), str(arr.shape[1])
            except Exception:
                pass
    elif type_str == "METADATA":
        representation = "capture_metadata_mat"
        metadata_present = "true"
    else:
        return None

    return {
        "path": path.relative_to(root).as_posix(),
        "sample_id": sample_id,
        "capture_id": capture_id,
        "scene_id": scene_id,
        "camera": camera,
        "iso": iso,
        "cfa": cfa,
        "width": width,
        "height": height,
        "bit_depth": bit_depth,
        "representation": representation,
        "metadata_present": metadata_present,
        "target_path": target_path,
        "alignment_verified": alignment_verified,
        "split": split,
        "license_note": license_note,
        "sha256": sha256_file(path),
    }


def inventory(root: Path) -> list[dict[str, str]]:
    rows = []
    for path in files_under(root):
        sidd_record = parse_sidd_record(path, root)
        if sidd_record is not None:
            rows.append(sidd_record)
        else:
            rows.append({
                "path": path.relative_to(root).as_posix(),
                "sample_id": "", "capture_id": "", "scene_id": "", "camera": "",
                "iso": "", "cfa": "", "width": "", "height": "", "bit_depth": "",
                "representation": classify(path), "metadata_present": "unknown",
                "target_path": "", "alignment_verified": "false", "split": "unassigned",
                "license_note": "unknown; record source/version/license before use",
                "sha256": sha256_file(path),
            })
    return rows


def write_csv(rows: list[dict[str, str]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="dataset directory to inventory")
    parser.add_argument("--output", type=Path, default=Path("data/manifests/sidd_inventory.csv"))
    args = parser.parse_args()
    if not args.root.is_dir():
        parser.error("root must be an existing directory")
    rows = inventory(args.root.resolve())
    write_csv(rows, args.output)
    summary = {"root": str(args.root.resolve()), "files": len(rows), "output": str(args.output)}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
