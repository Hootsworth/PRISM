"""Bayer spatial packing, exact inverse packing and paired D4 transforms."""
from __future__ import annotations

from typing import Tuple
import numpy as np


_D4 = {"identity", "rot90", "rot180", "rot270", "flip_horizontal", "flip_vertical", "transpose", "anti_transpose"}


def _check_array(array: np.ndarray, name: str) -> np.ndarray:
    value = np.asarray(array)
    if value.ndim != 2:
        raise ValueError("{} must be a two-dimensional array; got shape {}".format(name, value.shape))
    if value.dtype.kind not in "uif":
        raise TypeError("{} must have a real numeric dtype; got {}".format(name, value.dtype))
    if value.shape[0] % 2 or value.shape[1] % 2:
        raise ValueError("{} dimensions must both be even for Bayer packing; got {}".format(name, value.shape))
    return value


def pack_bayer(mosaic: np.ndarray) -> np.ndarray:
    """Pack a Bayer mosaic into [TL, TR, BL, BR] without CFA-based reordering."""
    raw = _check_array(mosaic, "Bayer mosaic")
    packed = np.stack((raw[0::2, 0::2], raw[0::2, 1::2],
                       raw[1::2, 0::2], raw[1::2, 1::2]), axis=0)
    return np.ascontiguousarray(packed)


def unpack_bayer(packed: np.ndarray) -> np.ndarray:
    """Reconstruct the exact input mosaic from a four-plane [TL,TR,BL,BR] array."""
    value = np.asarray(packed)
    if value.ndim != 3 or value.shape[0] != 4:
        raise ValueError("packed Bayer array must have shape (4,H/2,W/2); got {}".format(value.shape))
    if value.dtype.kind not in "uif":
        raise TypeError("packed Bayer planes must have a real numeric dtype; got {}".format(value.dtype))
    _, h2, w2 = value.shape
    raw = np.empty((h2 * 2, w2 * 2), dtype=value.dtype)
    raw[0::2, 0::2] = value[0]
    raw[0::2, 1::2] = value[1]
    raw[1::2, 0::2] = value[2]
    raw[1::2, 1::2] = value[3]
    return raw


def apply_dihedral(array: np.ndarray, operation: str) -> np.ndarray:
    """Apply a declared D4 spatial transform to the first two (image) axes."""
    value = np.asarray(array)
    if value.ndim < 2:
        raise ValueError("spatial transform needs at least two dimensions")
    if operation not in _D4:
        raise ValueError("unsupported paired augmentation {!r}; supported: {}".format(operation, ", ".join(sorted(_D4))))
    if operation == "identity": result = value
    elif operation == "rot90": result = np.rot90(value, 1, axes=(0, 1))
    elif operation == "rot180": result = np.rot90(value, 2, axes=(0, 1))
    elif operation == "rot270": result = np.rot90(value, 3, axes=(0, 1))
    elif operation == "flip_horizontal": result = np.flip(value, axis=1)
    elif operation == "flip_vertical": result = np.flip(value, axis=0)
    elif operation == "transpose": result = np.swapaxes(value, 0, 1)
    else: result = np.flip(np.swapaxes(value, 0, 1), axis=(0, 1))
    return np.ascontiguousarray(result)


def transform_cfa_pattern(pattern: str, operation: str) -> str:
    """Return the CFA phase produced by a spatial transform (diagnostics only).

    This label is never passed to B1. Returning GBRG is valid for transformed
    samples even though the source capture set's documented patterns differ.
    """
    name = str(pattern).strip().upper()
    if len(name) != 4 or any(ch not in "RGB" for ch in name):
        raise ValueError("CFA pattern must contain four R/G/B letters")
    tile = np.asarray(list(name)).reshape(2, 2)
    return "".join(apply_dihedral(tile, operation).reshape(-1).tolist())


def transform_paired(raw: np.ndarray, target: np.ndarray, operation: str) -> Tuple[np.ndarray, np.ndarray]:
    """Transform full-resolution Bayer and RGB arrays identically, then repack later."""
    mosaic = _check_array(raw, "Bayer mosaic")
    rgb = np.asarray(target)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("target must be HxWx3 RGB; got {}".format(rgb.shape))
    if rgb.shape[:2] != mosaic.shape:
        raise ValueError("paired RAW/target dimensions differ: {} vs {}".format(mosaic.shape, rgb.shape[:2]))
    return apply_dihedral(mosaic, operation), apply_dihedral(rgb, operation)


def augmentation_names():
    """Deterministic order for seeded augmentation sampling."""
    return tuple(sorted(_D4))


def sample_augmentation(rng) -> str:
    """Sample a D4 operation from an explicit NumPy-compatible RNG."""
    return str(rng.choice(augmentation_names()))
