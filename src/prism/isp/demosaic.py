"""Normalized-convolution bilinear Bayer demosaicing."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import convolve

from .raw import RawInputError, SUPPORTED_CFA

_BILINEAR_KERNEL = np.array(
    [[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]], dtype=np.float32
)


def cfa_color_masks(shape: tuple[int, int], pattern: str) -> np.ndarray:
    """Return HxWx3 boolean masks in RGB order for a named Bayer phase."""
    pattern = pattern.upper()
    if pattern not in SUPPORTED_CFA:
        raise RawInputError("unsupported CFA pattern: {}".format(pattern))
    if len(shape) != 2 or shape[0] < 2 or shape[1] < 2:
        raise RawInputError("mosaic shape must be at least 2x2")
    tile = np.array([list(pattern[0:2]), list(pattern[2:4])])
    # Map characters to colors; the two G sites share one output plane.
    h, w = shape
    masks = np.zeros((h, w, 3), dtype=bool)
    color_index = {"R": 0, "G": 1, "B": 2}
    for dy in range(2):
        for dx in range(2):
            masks[dy::2, dx::2, color_index[tile[dy, dx]]] = True
    return masks


def bilinear_demosaic(mosaic: np.ndarray, pattern: str) -> np.ndarray:
    """Deterministic 3x3 bilinear interpolation; preserves measured CFA sites.

    Missing samples are normalized convolutions of each color's observed
    samples with the standard separable [1, 2, 1] tent kernel. Reflective
    boundary extension avoids undefined image borders. No clipping is applied.
    """
    image = np.asarray(mosaic)
    if image.ndim != 2 or image.dtype.kind not in "fiu" or not np.all(np.isfinite(image)):
        raise RawInputError("demosaic expects a finite two-dimensional mosaic")
    masks = cfa_color_masks(image.shape, pattern)
    source = image.astype(np.float32, copy=False)
    output = np.empty((*image.shape, 3), dtype=np.float32)
    for channel in range(3):
        mask = masks[..., channel].astype(np.float32)
        weighted = convolve(source * mask, _BILINEAR_KERNEL, mode="reflect")
        weights = convolve(mask, _BILINEAR_KERNEL, mode="reflect")
        if np.any(weights == 0):
            raise RawInputError("CFA pattern does not sample color channel {}".format(channel))
        output[..., channel] = weighted / weights
        output[..., channel][masks[..., channel]] = source[masks[..., channel]]
    return output
