"""Classical B0 Bayer RAW to linear sRGB pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .color import (ColorTransform, apply_matrix_rgb, apply_matrix_rgb_reference,
                    build_color_transform)
from .demosaic import bilinear_demosaic
from .raw import RawFrame, RawInputError
from .render import encode_srgb, tone_map_reinhard


@dataclass(frozen=True)
class B0Result:
    camera_rgb_linear: np.ndarray
    xyz_d50: np.ndarray
    linear_srgb: np.ndarray
    display_linear_rgb: np.ndarray
    srgb_encoded: np.ndarray
    transform: ColorTransform


def _process_bayer(frame: RawFrame, tone_map: str | None, optimized: bool) -> B0Result:
    """Shared pipeline with reference and fused matrix implementations."""
    if tone_map not in (None, "reinhard"):
        raise RawInputError("tone_map must be None or 'reinhard'")
    camera_rgb = bilinear_demosaic(frame.mosaic, frame.cfa_pattern)
    transform = build_color_transform(frame.as_shot_neutral, frame.color_matrix2)
    if optimized:
        # Adjacent linear stages are composed once, then applied in two full
        # frame passes. This preserves the exposed XYZ(D50) intermediate.
        camera_to_d50 = transform.source_to_d50 @ transform.camera_to_xyz
        xyz_d50 = apply_matrix_rgb(camera_rgb, camera_to_d50)
        d50_to_linear_srgb = transform.xyz_d65_to_linear_srgb @ transform.d50_to_d65
        linear_srgb = apply_matrix_rgb(xyz_d50, d50_to_linear_srgb)
    else:
        xyz = apply_matrix_rgb_reference(camera_rgb, transform.camera_to_xyz)
        xyz_d50 = apply_matrix_rgb_reference(xyz, transform.source_to_d50)
        del xyz
        xyz_d65 = apply_matrix_rgb_reference(xyz_d50, transform.d50_to_d65)
        linear_srgb = apply_matrix_rgb_reference(xyz_d65, transform.xyz_d65_to_linear_srgb)
        del xyz_d65
    if tone_map == "reinhard":
        display_linear = tone_map_reinhard(linear_srgb)
    else:
        display_linear = linear_srgb.copy()
    srgb = encode_srgb(display_linear)
    for name, array in (("camera RGB", camera_rgb), ("XYZ D50", xyz_d50),
                        ("linear sRGB", linear_srgb), ("rendered sRGB", srgb)):
        if not np.all(np.isfinite(array)):
            raise RawInputError("{} produced non-finite values".format(name))
    return B0Result(camera_rgb, xyz_d50, linear_srgb, display_linear, srgb, transform)


def process_bayer(frame: RawFrame, tone_map: str | None = "reinhard") -> B0Result:
    """Run optimized B0 without changing the source mosaic or clipping values.

    SIDD-CC normalization is already applied. Demosaicing uses normalized
    convolution bilinear interpolation. White balance is incorporated as a
    Bradford adaptation derived from the verified camera neutral, following
    the DNG no-forward-matrix path. Output `linear_srgb` is pre-tone-map,
    pre-OETF and intentionally may contain negative or >1 values.
    """
    return _process_bayer(frame, tone_map, optimized=True)


def process_bayer_reference(frame: RawFrame, tone_map: str | None = "reinhard") -> B0Result:
    """Run the original multi-pass B0 arithmetic for reproducible comparison."""
    return _process_bayer(frame, tone_map, optimized=False)
