"""Checksum-aware SIDD-CC dataset pipeline for frozen PRISM B1."""
from __future__ import annotations

import csv
import hashlib
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from scipy.io import loadmat
from torch.utils.data import DataLoader, Dataset, get_worker_info

from .b1_packing import pack_bayer, sample_augmentation, transform_paired

FROZEN_SPLIT_SHA256 = "9e93f75071fb3fc4bf5e2c0e0ab127cc90104caafe231dda83f2a2d388ec441c"
SOURCE_FILE_MANIFEST_SHA256 = "42c5c6db7519946e3a73c7060c8f1e84e1f6ac181d47f970ee7052a5367500a8"
SOURCE_GROUP_SPLIT_SHA256 = "1a55e9a3c55f1f08d97f878005a4aeeb75c6a9b1d0e14ef8cbbc53c58d026520"
PAIR_INVENTORY_SHA256 = "8090fa23d8c62f827a5d8b79a08b81c12066d20afef2cb300bc772aa0d8baa01"
SPLIT_COLUMN = "split_frozen_scene_grouped"
SPLITS = {"train", "validation", "internal_test", "official_test"}
PAIR_ROLES = ("raw_noisy", "raw_gt", "raw_metadata", "srgb_noisy", "srgb_gt")
TEST_RAW_NOISY = "data/external/sidd_cc_expanded/Raw/test/BenchmarkNoisyBlocksRaw.mat"
TEST_RAW_GT = "data/external/sidd_cc_expanded/Raw/test/BenchmarkGtBlocksRaw.mat"
TEST_SRGB_NOISY = "data/external/sidd_cc_expanded/SRGB/test/BenchmarkNoisyBlocksSrgb.mat"
TEST_SRGB_GT = "data/external/sidd_cc_expanded/SRGB/test/BenchmarkGtBlocksSrgb.mat"
TEST_COORDS = "data/external/sidd_cc_expanded/Raw/test/BenchmarkBlocks32.mat"
TEST_SCENES = "data/external/sidd_cc_expanded/Raw/test/test_scenes_info.txt"


class B1DataError(RuntimeError):
    """Raised when data contradicts the frozen B1 data contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_csv(path: Path) -> List[dict]:
    try:
        with path.open(newline="", encoding="utf-8") as stream:
            return list(csv.DictReader(stream))
    except OSError as exc:
        raise B1DataError("cannot read manifest {}: {}".format(path, exc)) from exc


def resolve_project_path(root: Path, relative: str) -> Path:
    """Resolve a manifest path and reject absolute paths or traversal outside root."""
    candidate = Path(relative)
    if candidate.is_absolute():
        raise B1DataError("manifest path must be project-relative: {}".format(relative))
    resolved_root = root.resolve()
    resolved = (resolved_root / candidate).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise B1DataError("manifest path escapes project root: {}".format(relative)) from exc
    return resolved


def load_raw_mat(path: Path, expected_shape: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """Read only the verified float32 MAT variable `data`; preserve every value."""
    try:
        contents = loadmat(str(path), variable_names=["data"], squeeze_me=False,
                           struct_as_record=False, verify_compressed_data_integrity=True)
    except Exception as exc:
        raise B1DataError("{}: unable to decode MATLAB MAT file: {}".format(path, exc)) from exc
    if "data" not in contents:
        raise B1DataError("{}: required MAT variable 'data' is missing".format(path))
    raw = np.asarray(contents["data"])
    if raw.ndim != 2:
        raise B1DataError("{}: RAW data must be 2D; got shape {}".format(path, raw.shape))
    if raw.dtype != np.float32:
        raise B1DataError("{}: RAW data must be float32; got {}".format(path, raw.dtype))
    if raw.shape[0] % 2 or raw.shape[1] % 2:
        raise B1DataError("{}: RAW dimensions must be even; got {}".format(path, raw.shape))
    if expected_shape is not None and tuple(raw.shape) != tuple(expected_shape):
        raise B1DataError("{}: RAW shape {} differs from manifest {}".format(path, raw.shape, expected_shape))
    if not np.isfinite(raw).all():
        bad = int(raw.size - np.isfinite(raw).sum())
        raise B1DataError("{}: RAW contains {} non-finite values".format(path, bad))
    return raw


def load_srgb_png(path: Path, expected_shape: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """Read a native 8-bit RGB PNG without gamma or color transformations."""
    try:
        with Image.open(str(path)) as image:
            image.load()
            if image.format != "PNG":
                raise B1DataError("{}: target must be PNG; got {}".format(path, image.format))
            if image.mode != "RGB":
                raise B1DataError("{}: target must have RGB mode; got {}".format(path, image.mode))
            rgb = np.asarray(image)
    except B1DataError:
        raise
    except Exception as exc:
        raise B1DataError("{}: unable to decode RGB target: {}".format(path, exc)) from exc
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise B1DataError("{}: target must be HxWx3 uint8 RGB; got {} {}".format(path, rgb.shape, rgb.dtype))
    if expected_shape is not None and tuple(rgb.shape[:2]) != tuple(expected_shape):
        raise B1DataError("{}: target shape {} differs from manifest {}".format(path, rgb.shape[:2], expected_shape))
    return rgb


def validate_pair(raw: np.ndarray, target: np.ndarray, sample_id: str) -> None:
    if raw.ndim != 2 or target.ndim != 3 or target.shape[2] != 3:
        raise B1DataError("{}: expected 2D Bayer RAW and HxWx3 RGB target".format(sample_id))
    if raw.shape != target.shape[:2]:
        raise B1DataError("{}: RAW/target dimensions differ: {} vs {}".format(sample_id, raw.shape, target.shape[:2]))
    if raw.dtype != np.float32 or target.dtype != np.uint8:
        raise B1DataError("{}: expected RAW float32 and target uint8, got {} and {}".format(sample_id, raw.dtype, target.dtype))
    if not np.isfinite(raw).all():
        raise B1DataError("{}: RAW contains non-finite values".format(sample_id))


def arrays_to_tensors(packed: np.ndarray, rgb_u8: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
    if packed.ndim != 3 or packed.shape[0] != 4:
        raise B1DataError("packed RAW tensor must have shape (4,H/2,W/2); got {}".format(packed.shape))
    if packed.dtype != np.float32 or not np.isfinite(packed).all():
        raise B1DataError("packed RAW must be finite float32")
    if rgb_u8.ndim != 3 or rgb_u8.shape[2] != 3 or rgb_u8.dtype != np.uint8:
        raise B1DataError("RGB target must be HxWx3 uint8")
    x = torch.from_numpy(np.ascontiguousarray(packed))
    y = torch.from_numpy(np.ascontiguousarray(rgb_u8.transpose(2, 0, 1))).to(dtype=torch.float32).div_(255.0)
    if not x.is_contiguous() or not y.is_contiguous():
        raise B1DataError("tensor conversion failed to produce contiguous NCHW samples")
    if not torch.isfinite(x).all() or not torch.isfinite(y).all():
        raise B1DataError("tensor conversion produced non-finite values")
    if float(y.min()) < 0.0 or float(y.max()) > 1.0:
        raise B1DataError("scaled sRGB target is outside [0,1]")
    return x, y


def tensor_sample_stats(raw: np.ndarray, packed: np.ndarray, target_u8: np.ndarray) -> dict:
    def stats(a):
        v = np.asarray(a)
        finite = np.isfinite(v)
        finite_values = v[finite]
        return {"shape": list(v.shape), "dtype": str(v.dtype),
                "min": float(finite_values.min()) if finite_values.size else None,
                "max": float(finite_values.max()) if finite_values.size else None,
                "mean": float(finite_values.mean()) if finite_values.size else None,
                "std": float(finite_values.std()) if finite_values.size else None,
                "finite_count": int(finite.sum()), "nan_count": int(np.isnan(v).sum()),
                "infinite_count": int(np.isinf(v).sum())}
    return {"raw": stats(raw), "packed": stats(packed), "target_uint8": stats(target_u8)}


def _default_root() -> Path:
    return Path(__file__).resolve().parents[3]


class B1Dataset(Dataset):
    """Split-aware SIDD-CC dataset returning packed Bayer and RGB tensors.

    Dataset items are dictionaries with CHW float32 `input`, CHW float32
    `target`, stable `sample_id` / `group_id`, and optionally `raw_mosaic` for
    diagnostics. `crop_size=256` and `augment=True` implement the preregistered
    training crop policy; evaluation defaults to full-resolution samples.
    """
    def __init__(self, split: str, root: Optional[Path] = None,
                 split_manifest: Optional[Path] = None,
                 pair_inventory: Optional[Path] = None,
                 file_manifest: Optional[Path] = None,
                 crop_size: Optional[int] = None, augment: Optional[bool] = None,
                 seed: int = 20261106, verify_hashes: bool = True,
                 include_raw_mosaic: bool = False):
        if split not in SPLITS:
            raise ValueError("split must be one of {}; got {!r}".format(sorted(SPLITS), split))
        self.root = Path(root) if root is not None else _default_root()
        self.split = split
        self.split_manifest_path = Path(split_manifest) if split_manifest else self.root / "experiments/P3/B1_SPLIT_MANIFEST.csv"
        self.pair_inventory_path = Path(pair_inventory) if pair_inventory else self.root / "data/manifests/sidd_cc_expanded_pair_inventory.csv"
        self.file_manifest_path = Path(file_manifest) if file_manifest else self.root / "data/manifests/sidd_cc_expanded_file_manifest.csv"
        self.crop_size = crop_size
        self.augment = (split == "train" and crop_size is not None) if augment is None else bool(augment)
        self.seed = int(seed)
        self.verify_hashes = bool(verify_hashes)
        self.include_raw_mosaic = bool(include_raw_mosaic)
        if crop_size is not None and (crop_size <= 0 or crop_size % 2):
            raise ValueError("crop_size must be a positive even number")
        if self.augment and split != "train":
            raise ValueError("stochastic augmentation is permitted only for the training split")
        if self.augment and crop_size is None:
            raise ValueError("frozen training augmentation requires a paired crop_size")
        self._rng = np.random.default_rng(self.seed)
        self._verified_hashes = set()
        self._test_arrays = None
        self._scene_to_index = None
        self._coordinates = None
        self._split_rows, self._pair_rows, self._file_rows = self._load_and_validate_manifests()
        self._rows = self._build_samples()
        self._audit_references()

    def _load_and_validate_manifests(self):
        for path, expected, label in (
            (self.split_manifest_path, FROZEN_SPLIT_SHA256, "frozen B1 split manifest"),
            (self.file_manifest_path, SOURCE_FILE_MANIFEST_SHA256, "source file manifest")):
            if not path.is_file():
                raise B1DataError("{} is missing: {}".format(label, path))
            observed = sha256_file(path)
            if observed != expected:
                raise B1DataError("{} SHA-256 mismatch: expected {}, got {}".format(label, expected, observed))
        split_rows = _read_csv(self.split_manifest_path)
        if sha256_file(self.pair_inventory_path) != PAIR_INVENTORY_SHA256:
            raise B1DataError("paired sample inventory SHA-256 does not match the audited version")
        pair_rows = _read_csv(self.pair_inventory_path)
        file_rows = _read_csv(self.file_manifest_path)
        source_split_path = self.root / "data/manifests/sidd_cc_expanded_split_manifest.csv"
        if not source_split_path.is_file() or sha256_file(source_split_path) != SOURCE_GROUP_SPLIT_SHA256:
            raise B1DataError("source group split manifest is missing or does not match the audited checksum")
        expected_source_hashes = {r.get("split_manifest_sha256", "") for r in split_rows}
        if expected_source_hashes != {SOURCE_GROUP_SPLIT_SHA256}:
            raise B1DataError("B1 split rows do not bind to the audited source split manifest")
        file_hash = {r["path"]: r for r in file_rows}
        if len(file_hash) != len(file_rows):
            raise B1DataError("source file manifest contains duplicate paths")
        for row in split_rows:
            if row.get("source_file_manifest_sha256") != SOURCE_FILE_MANIFEST_SHA256:
                raise B1DataError("split row {} has unexpected source file manifest checksum".format(row.get("group_id")))
        return split_rows, pair_rows, file_hash

    def _build_samples(self):
        group_rows = {r["group_id"]: r for r in self._split_rows}
        if len(group_rows) != len(self._split_rows):
            raise B1DataError("frozen split manifest has duplicate group IDs")
        labels = {r["group_id"]: r[SPLIT_COLUMN] for r in self._split_rows}
        if any(label not in SPLITS for label in labels.values()):
            raise B1DataError("frozen split manifest contains an unsupported split label")
        samples = []
        seen_pair_ids = set()
        path_split = {}
        path_owner = {}
        for row in self._pair_rows:
            group = row["instance_id"]
            if group not in labels or labels[group] == "official_test":
                raise B1DataError("pair inventory references unknown or official-test group {}".format(group))
            assigned = labels[group]
            sample_id = "{}:{}".format(group, row["pair_index"])
            if sample_id in seen_pair_ids:
                raise B1DataError("duplicate sample identifier {}".format(sample_id))
            seen_pair_ids.add(sample_id)
            for role in PAIR_ROLES:
                rel = row.get(role, "")
                if not rel:
                    raise B1DataError("{}: missing {} manifest reference".format(sample_id, role))
                path = resolve_project_path(self.root, rel)
                if not path.is_file():
                    raise B1DataError("{}: referenced {} file is missing: {}".format(sample_id, role, path))
                prior = path_split.get(rel)
                if prior is not None and prior != assigned:
                    raise B1DataError("cross-split file leakage for {}: {} and {}".format(rel, prior, assigned))
                path_split[rel] = assigned
                owner = path_owner.get(rel)
                if owner is not None and owner != sample_id:
                    raise B1DataError("duplicate file reference {} in samples {} and {}".format(rel, owner, sample_id))
                path_owner[rel] = sample_id
                if rel not in self._file_rows:
                    raise B1DataError("{}: {} is absent from SHA-256 file manifest".format(sample_id, rel))
            if row.get("split") not in {"train", "validation"}:
                raise B1DataError("{}: unexpected historical pair split label {!r}".format(sample_id, row.get("split")))
            if int(row.get("height", 0)) <= 0 or int(row.get("width", 0)) <= 0:
                raise B1DataError("{}: invalid dimensions in pair inventory".format(sample_id))
            samples.append({"sample_id": sample_id, "group_id": group,
                            "split": assigned, "pair_index": row["pair_index"],
                            "height": int(row["height"]), "width": int(row["width"]),
                            "files": {role: row[role] for role in PAIR_ROLES},
                            "camera": row.get("camera", ""), "cfa": row.get("cfa", "")})
        ids = set(labels)
        non_test_ids = {gid for gid, label in labels.items() if label != "official_test"}
        if len(samples) != 230 or non_test_ids != {r["group_id"] for r in self._split_rows if r[SPLIT_COLUMN] != "official_test"}:
            raise B1DataError("expected all 230 audited train RAW/RGB pairs; found {}".format(len(samples)))
        by_group = {}
        for item in samples:
            by_group.setdefault(item["group_id"], []).append(item)
        for group in non_test_ids:
            if len(by_group.get(group, [])) != 2:
                raise B1DataError("{}: expected exactly two indexed RAW/RGB pairs".format(group))
            ordered = sorted(by_group[group], key=lambda item: item["pair_index"])
            group_parts = []
            for item in ordered:
                for role in PAIR_ROLES:
                    rel = item["files"][role]
                    group_parts.append("{}:{}:{}".format(item["pair_index"], rel,
                                                           self._file_rows[rel]["sha256"]))
            # This encoding is shared with scripts/freeze_b1_split_manifest.py.
            checksum_text = group + "|pair_indices=" + ",".join(item["pair_index"] for item in ordered) + "|" + ";".join(group_parts)
            observed = hashlib.sha256(checksum_text.encode("utf-8")).hexdigest()
            frozen_row = group_rows[group]
            if observed != frozen_row.get("group_source_sha256"):
                raise B1DataError("{}: frozen source group checksum mismatch".format(group))
        official_rows = [r for r in self._split_rows if r[SPLIT_COLUMN] == "official_test"]
        if len(official_rows) != 29 or len(ids) != 144:
            raise B1DataError("frozen split counts mismatch: {} groups, {} official test".format(len(ids), len(official_rows)))
        return sorted([s for s in samples if s["split"] == self.split], key=lambda x: x["sample_id"]) if self.split != "official_test" else self._build_test_samples(official_rows)

    def _build_test_samples(self, rows):
        scenes = self._read_scene_ids()
        manifest_scenes = [r["group_id"] for r in rows]
        if scenes != manifest_scenes:
            raise B1DataError("official test scene order differs from frozen split manifest")
        self._scene_to_index = {scene: i for i, scene in enumerate(scenes)}
        coordinate_paths = [TEST_COORDS, "data/external/sidd_cc_expanded/SRGB/test/BenchmarkBlocks32.mat"]
        for rel in coordinate_paths:
            self._verify_file_hash(rel, "official test block coordinate array")
        c0 = loadmat(str(resolve_project_path(self.root, coordinate_paths[0])), variable_names=["BenchmarkBlocks32"])["BenchmarkBlocks32"]
        c1 = loadmat(str(resolve_project_path(self.root, coordinate_paths[1])), variable_names=["BenchmarkBlocks32"])["BenchmarkBlocks32"]
        if c0.shape != (32, 4) or not np.array_equal(c0, c1):
            raise B1DataError("RAW/SRGB official-test block coordinates differ or have wrong shape")
        self._coordinates = c0.copy()
        test_files = (TEST_RAW_NOISY, TEST_RAW_GT, TEST_SRGB_NOISY, TEST_SRGB_GT, TEST_COORDS)
        result = []
        for row in rows:
            gid = row["group_id"]
            if row.get("sample_count") != "32":
                raise B1DataError("{}: official test sample_count is not 32".format(gid))
            # Recompute the group checksum from the pinned file manifest records.
            parts = ["block_indices=0..31; scene order from test_scenes_info.txt"]
            pairs_hash = ";".join("{}:{}".format(p, self._file_rows[p]["sha256"]) for p in test_files)
            expected_group_sha = hashlib.sha256((gid+"|"+parts[0]+"|"+pairs_hash).encode("utf-8")).hexdigest()
            if row.get("group_source_sha256") != expected_group_sha:
                raise B1DataError("{}: official test group source checksum mismatch".format(gid))
            for rel in test_files:
                self._verify_file_hash(rel, "official test source")
            for block in range(32):
                result.append({"sample_id": "{}:block{:02d}".format(gid, block),
                               "group_id": gid, "split": "official_test", "block_index": block,
                               "scene_index": self._scene_to_index[gid], "height": 256, "width": 256,
                               "files": {"raw_noisy": TEST_RAW_NOISY, "raw_gt": TEST_RAW_GT,
                                         "srgb_noisy": TEST_SRGB_NOISY, "srgb_gt": TEST_SRGB_GT,
                                         "coordinates": TEST_COORDS}})
        return result

    def _read_scene_ids(self):
        self._verify_file_hash(TEST_SCENES, "official test scene ordering")
        path = resolve_project_path(self.root, TEST_SCENES)
        if not path.is_file():
            raise B1DataError("official test scene list is missing: {}".format(path))
        rows = []
        with path.open(encoding="utf-8") as f:
            header = f.readline().strip().split("\t")
            try: index = header.index("scene_id")
            except ValueError as exc: raise B1DataError("test_scenes_info.txt has no scene_id column") from exc
            for line in f:
                fields = line.rstrip("\n").split("\t")
                if fields and fields[0]: rows.append(fields[index])
        return rows

    def _audit_references(self):
        dataset_root = self.root / "data/external/sidd_cc_expanded"
        if not dataset_root.is_dir():
            raise B1DataError("expanded SIDD-CC dataset directory is missing: {}".format(dataset_root))
        # Hugging Face keeps lock/metadata files in .cache beside snapshots;
        # these are transport state, not published dataset members.
        on_disk_all = {p.relative_to(self.root).as_posix() for p in dataset_root.rglob("*")
                       if p.is_file() and ".cache" not in p.relative_to(dataset_root).parts}
        manifest_all = set(self._file_rows)
        absent_from_disk = sorted(manifest_all-on_disk_all)
        absent_from_manifest = sorted(on_disk_all-manifest_all)
        if absent_from_disk or absent_from_manifest:
            raise B1DataError("dataset file inventory mismatch; missing on disk={}, absent from manifest={}".format(
                absent_from_disk[:8], absent_from_manifest[:8]))
        # Reportable audit: all MAT/PNG files that represent image data must be referenced.
        referenced = {p for row in self._pair_rows for p in (row[r] for r in PAIR_ROLES)}
        referenced.update((TEST_RAW_NOISY, TEST_RAW_GT, TEST_SRGB_NOISY, TEST_SRGB_GT,
                           TEST_COORDS, "data/external/sidd_cc_expanded/SRGB/test/BenchmarkBlocks32.mat"))
        train_references = {p for row in self._pair_rows for p in (row[r] for r in PAIR_ROLES)}
        official_references = {TEST_RAW_NOISY, TEST_RAW_GT, TEST_SRGB_NOISY, TEST_SRGB_GT,
                               TEST_COORDS, "data/external/sidd_cc_expanded/SRGB/test/BenchmarkBlocks32.mat"}
        if train_references & official_references:
            raise B1DataError("training and official-test manifests reference the same files")
        roots = [self.root/"data/external/sidd_cc_expanded/Raw/train/Data",
                 self.root/"data/external/sidd_cc_expanded/SRGB/train/Data",
                 self.root/"data/external/sidd_cc_expanded/Raw/test",
                 self.root/"data/external/sidd_cc_expanded/SRGB/test"]
        actual = set()
        for base in roots:
            if not base.is_dir(): raise B1DataError("dataset directory is missing: {}".format(base))
            for path in base.rglob("*"):
                if path.is_file() and path.suffix.lower() in {".mat", ".png"}:
                    actual.add(path.relative_to(self.root).as_posix())
        extra = sorted(actual-referenced)
        missing = sorted(referenced-actual)
        if extra or missing:
            raise B1DataError("dataset image-file reference audit failed; unreferenced={}, missing={}".format(extra[:8], missing[:8]))

    def __len__(self): return len(self._rows)

    def _verify_file_hash(self, relative: str, sample_id: str):
        if not self.verify_hashes or relative in self._verified_hashes: return
        record = self._file_rows.get(relative)
        if record is None: raise B1DataError("{}: {} absent from source SHA manifest".format(sample_id, relative))
        path = resolve_project_path(self.root, relative)
        if not path.is_file(): raise B1DataError("{}: missing file {}".format(sample_id, path))
        if path.stat().st_size != int(record["size_bytes"]):
            raise B1DataError("{}: file size mismatch for {}".format(sample_id, relative))
        observed = sha256_file(path)
        if observed != record["sha256"]:
            raise B1DataError("{}: SHA-256 mismatch for {} (expected {}, got {})".format(sample_id, relative, record["sha256"], observed))
        self._verified_hashes.add(relative)

    def verify_all_source_hashes(self) -> dict:
        """Verify every file named by the pinned 1,163-entry source manifest."""
        total_bytes = 0
        for relative, record in self._file_rows.items():
            self._verify_file_hash(relative, "source file manifest audit")
            total_bytes += int(record["size_bytes"])
        return {"file_count": len(self._file_rows), "total_bytes": total_bytes}

    def _load_train_arrays(self, row):
        sample_id = row["sample_id"]
        for rel in row["files"].values(): self._verify_file_hash(rel, sample_id)
        raw_path = resolve_project_path(self.root, row["files"]["raw_noisy"])
        target_path = resolve_project_path(self.root, row["files"]["srgb_gt"])
        shape = (row["height"], row["width"])
        raw = load_raw_mat(raw_path, shape)
        target = load_srgb_png(target_path, shape)
        validate_pair(raw, target, sample_id)
        return raw, target

    def _load_test_arrays(self):
        if self._test_arrays is not None: return self._test_arrays
        paths = [TEST_RAW_NOISY, TEST_SRGB_GT]
        for rel in paths: self._verify_file_hash(rel, "official test tensor")
        try:
            raw_all = loadmat(str(resolve_project_path(self.root, TEST_RAW_NOISY)), variable_names=["data"], squeeze_me=False)["data"]
            srgb_all = loadmat(str(resolve_project_path(self.root, TEST_SRGB_GT)), variable_names=["data"], squeeze_me=False)["data"]
        except Exception as exc:
            raise B1DataError("unable to load official test block tensors: {}".format(exc)) from exc
        if raw_all.shape != (29, 32, 256, 256) or raw_all.dtype != np.float32:
            raise B1DataError("official test noisy RAW tensor shape/dtype mismatch: {} {}".format(raw_all.shape, raw_all.dtype))
        if srgb_all.shape != (29, 32, 256, 256, 3) or srgb_all.dtype != np.uint8:
            raise B1DataError("official test GT sRGB tensor shape/dtype mismatch: {} {}".format(srgb_all.shape, srgb_all.dtype))
        if not np.isfinite(raw_all).all(): raise B1DataError("official test RAW tensor contains non-finite values")
        self._test_arrays = raw_all, srgb_all
        return self._test_arrays

    def __getitem__(self, index):
        row = self._rows[index]
        sample_id = row["sample_id"]
        if self.split == "official_test":
            raw_all, rgb_all = self._load_test_arrays()
            raw = raw_all[row["scene_index"], row["block_index"]]
            target = rgb_all[row["scene_index"], row["block_index"]]
            validate_pair(raw, target, sample_id)
        else:
            raw, target = self._load_train_arrays(row)
        if self.crop_size is not None:
            ch = cw = int(self.crop_size)
            h, w = raw.shape
            if ch > h or cw > w:
                raise B1DataError("{}: requested crop {} exceeds image {}".format(sample_id, (ch, cw), raw.shape))
            max_y = (h-ch)//2
            max_x = (w-cw)//2
            # Crop offsets are multiples of two sensor pixels, preserving the Bayer lattice.
            y = 2*int(self._rng.integers(max_y+1))
            x = 2*int(self._rng.integers(max_x+1))
            raw = raw[y:y+ch, x:x+cw]
            target = target[y:y+ch, x:x+cw]
        if self.augment:
            operation = sample_augmentation(self._rng)
            raw, target = transform_paired(raw, target, operation)
        packed = pack_bayer(raw)
        input_tensor, target_tensor = arrays_to_tensors(packed, target)
        expected = (4, raw.shape[0]//2, raw.shape[1]//2)
        if tuple(input_tensor.shape) != expected or tuple(target_tensor.shape) != (3, raw.shape[0], raw.shape[1]):
            raise B1DataError("{}: tensor shape contract violated: input {}, target {}".format(sample_id, tuple(input_tensor.shape), tuple(target_tensor.shape)))
        item = {"input": input_tensor, "target": target_tensor,
                "sample_id": sample_id, "group_id": row["group_id"], "split": self.split}
        if self.include_raw_mosaic:
            item["raw_mosaic"] = torch.from_numpy(np.ascontiguousarray(raw))
            item["target_uint8"] = torch.from_numpy(np.array(target, copy=True, order="C"))
        return item

    def reseed_worker(self, seed: int):
        self._rng = np.random.default_rng(int(seed))

    def close(self):
        self._test_arrays = None
        self._coordinates = None
        self._verified_hashes.clear()


def seed_b1_worker(worker_id: int):
    """Seed Python/NumPy and the worker's dataset augmentation RNG."""
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)
    info = get_worker_info()
    if info is not None and hasattr(info.dataset, "reseed_worker"):
        info.dataset.reseed_worker(seed)


def make_b1_dataloader(dataset: B1Dataset, batch_size: Optional[int] = None,
                       num_workers: int = 0, pin_memory: Optional[bool] = None,
                       persistent_workers: Optional[bool] = None,
                       shuffle: Optional[bool] = None, seed: int = 20261106) -> DataLoader:
    """Create a reproducible loader; default train batches use 256px paired crops."""
    if batch_size is None: batch_size = 8 if dataset.split == "train" else 1
    if batch_size <= 0 or num_workers < 0: raise ValueError("invalid batch_size or num_workers")
    if pin_memory is None: pin_memory = bool(torch.cuda.is_available())
    if persistent_workers is None: persistent_workers = num_workers > 0
    if persistent_workers and num_workers == 0:
        raise ValueError("persistent_workers requires num_workers > 0")
    if shuffle is None: shuffle = dataset.split == "train"
    generator = torch.Generator().manual_seed(int(seed))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=pin_memory,
                      persistent_workers=persistent_workers,
                      worker_init_fn=seed_b1_worker, generator=generator,
                      drop_last=False)


def make_training_dataset(**kwargs) -> B1Dataset:
    """Construct the frozen training view: 256px crops plus paired D4 augmentation."""
    kwargs.setdefault("crop_size", 256)
    kwargs.setdefault("augment", True)
    return B1Dataset(split="train", **kwargs)
