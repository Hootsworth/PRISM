"""Optional display rendering stages; linear RGB remains a separate output."""

from __future__ import annotations

import numpy as np

from .raw import RawInputError


def tone_map_reinhard(linear_rgb: np.ndarray) -> np.ndarray:
    """Apply a simple luminance Reinhard curve while retaining chroma ratios.

    This display-only stage maps positive luminance Y to Y/(1+Y). Negative
    linear values are preserved until sRGB encoding/export, where display range
    clipping is explicitly applied.
    """
    rgb = np.asarray(linear_rgb, dtype=np.float32)
    if rgb.ndim != 3 or rgb.shape[-1] != 3 or not np.all(np.isfinite(rgb)):
        raise RawInputError("tone mapping expects finite HxWx3 linear RGB")
    luminance = rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722
    mapped = rgb.copy()
    positive = luminance > 0
    scale = np.ones_like(luminance, dtype=np.float32)
    scale[positive] = 1.0 / (1.0 + luminance[positive])
    mapped *= scale[..., None]
    return mapped


def encode_srgb(linear_rgb: np.ndarray) -> np.ndarray:
    """Apply the IEC sRGB OETF to linear sRGB without clipping the result."""
    rgb = np.asarray(linear_rgb, dtype=np.float32)
    if rgb.ndim != 3 or rgb.shape[-1] != 3 or not np.all(np.isfinite(rgb)):
        raise RawInputError("sRGB encoding expects finite HxWx3 linear RGB")
    out = np.empty_like(rgb)
    low = rgb <= 0.0031308
    out[low] = 12.92 * rgb[low]
    out[~low] = 1.055 * np.power(rgb[~low], 1.0 / 2.4) - 0.055
    return out


def to_uint8_srgb(encoded: np.ndarray) -> np.ndarray:
    """Clip only at the 8-bit display-file boundary and quantize deterministically."""
    values = np.asarray(encoded, dtype=np.float32)
    if values.ndim != 3 or values.shape[-1] != 3 or not np.all(np.isfinite(values)):
        raise RawInputError("PNG export expects finite HxWx3 encoded sRGB")
    return np.rint(np.clip(values, 0.0, 1.0) * 255.0).astype(np.uint8)
