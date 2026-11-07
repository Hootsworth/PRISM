"""SIDD-CC Bayer MAT input and metadata validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat


class RawInputError(ValueError):
    """A RAW input is malformed or lacks required, verified metadata."""


SUPPORTED_CFA = {"BGGR", "GRBG", "RGGB"}


@dataclass(frozen=True)
class RawFrame:
    """Validated, unmodified normalized SIDD-CC Bayer mosaic and metadata."""

    mosaic: np.ndarray
    cfa_pattern: str
    as_shot_neutral: np.ndarray
    color_matrix2: np.ndarray
    black_level_original: float
    white_level_original: float
    metadata_path: Path
    input_path: Path
    metadata: dict[str, Any]

    @property
    def saturation_candidate_mask(self) -> np.ndarray:
        """Mark samples at/above normalized white without changing or dropping them.

        SIDD-CC maps the original WhiteLevel to nominal 1.0, but does not ship a
        per-pixel clipping flag. This is therefore a candidate mask, not proof
        that a sensor site clipped.
        """
        return self.mosaic >= np.float32(1.0)

    @property
    def saturation_candidate_count(self) -> int:
        return int(np.count_nonzero(self.mosaic >= np.float32(1.0)))


def _scalar(data: dict[str, Any], key: str) -> float:
    if key not in data:
        raise RawInputError("required metadata field is missing: {}".format(key))
    value = np.asarray(data[key], dtype=np.float64).reshape(-1)
    if value.size != 1 or not np.isfinite(value[0]):
        raise RawInputError("metadata field {} must be one finite number".format(key))
    return float(value[0])


def _string(data: dict[str, Any], key: str) -> str:
    if key not in data:
        raise RawInputError("required metadata field is missing: {}".format(key))
    value = np.asarray(data[key]).reshape(-1)
    if value.size != 1:
        raise RawInputError("metadata field {} must be one string".format(key))
    item = value[0]
    if isinstance(item, bytes):
        item = item.decode("utf-8", errors="strict")
    return str(item).strip().upper()


def load_sidd_cc_mat(input_path: str | Path, metadata_path: str | Path | None = None) -> RawFrame:
    """Load the documented `data` array and paired `METADATA_RAW` MAT file.

    SIDD-CC values are already black-subtracted and normalized. Original level
    tags are retained as provenance only and are never applied to the array.
    """
    source = Path(input_path)
    if not source.is_file():
        raise RawInputError("RAW MAT file does not exist: {}".format(source))
    if metadata_path is None:
        name = source.name
        if "_NOISY_RAW_" in name:
            name = name.replace("_NOISY_RAW_", "_METADATA_RAW_")
        elif "_GT_RAW_" in name:
            name = name.replace("_GT_RAW_", "_METADATA_RAW_")
        else:
            raise RawInputError("cannot derive metadata filename; supply metadata_path")
        meta_path = source.with_name(name)
    else:
        meta_path = Path(metadata_path)
    if not meta_path.is_file():
        raise RawInputError("metadata MAT file does not exist: {}".format(meta_path))

    try:
        raw_mat = loadmat(str(source), variable_names=["data"])
        metadata = loadmat(str(meta_path))
    except Exception as exc:
        raise RawInputError("could not parse MAT input or metadata: {}".format(exc)) from exc
    if "data" not in raw_mat:
        raise RawInputError("SIDD-CC image MAT must contain variable 'data'")
    mosaic = np.asarray(raw_mat["data"])
    if mosaic.ndim != 2 or min(mosaic.shape) < 2:
        raise RawInputError("Bayer mosaic must be a 2D array at least 2x2")
    if mosaic.dtype.kind not in "fiu" or not np.all(np.isfinite(mosaic)):
        raise RawInputError("Bayer mosaic must contain finite numeric values")
    if mosaic.dtype != np.float32:
        raise RawInputError("verified SIDD-CC Bayer MAT representation must be float32")
    # This view is not mutated by any B0 stage. Negative sensor samples survive.
    mosaic.setflags(write=False)

    cfa = _string(metadata, "cfa_str")
    if cfa not in SUPPORTED_CFA:
        raise RawInputError("unsupported CFA pattern {!r}; supported: {}".format(cfa, ", ".join(sorted(SUPPORTED_CFA))))
    if "AsShotNeutral" not in metadata:
        raise RawInputError("required metadata field is missing: AsShotNeutral")
    neutral = np.asarray(metadata["AsShotNeutral"], dtype=np.float64).reshape(-1)
    if neutral.shape != (3,) or not np.all(np.isfinite(neutral)) or np.any(neutral <= 0):
        raise RawInputError("AsShotNeutral must contain three finite positive values")
    if "ColorMatrix2" not in metadata:
        raise RawInputError("required metadata field is missing: ColorMatrix2")
    cm_values = np.asarray(metadata["ColorMatrix2"], dtype=np.float64).reshape(-1)
    if cm_values.size != 9 or not np.all(np.isfinite(cm_values)):
        raise RawInputError("ColorMatrix2 must contain nine finite values")
    # SIDD-CC stores the nine DNG matrix coefficients as a row-major vector.
    color_matrix = cm_values.reshape(3, 3)
    if np.linalg.cond(color_matrix) > 1e8:
        raise RawInputError("ColorMatrix2 is singular or numerically unstable")
    black_level = _scalar(metadata, "BlackLevel")
    white_level = _scalar(metadata, "WhiteLevel")
    if white_level <= black_level:
        raise RawInputError("WhiteLevel must exceed BlackLevel")

    retained = {
        "cfa_str": cfa,
        "AsShotNeutral": neutral.copy(),
        "ColorMatrix2": color_matrix.copy(),
        "BlackLevel": black_level,
        "WhiteLevel": white_level,
        "normalization": "already (raw - BlackLevel) / (WhiteLevel - BlackLevel); do not repeat",
    }
    return RawFrame(
        mosaic=mosaic,
        cfa_pattern=cfa,
        as_shot_neutral=neutral,
        color_matrix2=color_matrix,
        black_level_original=black_level,
        white_level_original=white_level,
        metadata_path=meta_path,
        input_path=source,
        metadata=retained,
    )
