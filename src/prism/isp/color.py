"""DNG-convention camera RGB to XYZ(D50), Bradford adaptation, and sRGB."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .raw import RawInputError

D50_XYZ = np.array([0.96422, 1.0, 0.82521], dtype=np.float64)
D65_XYZ = np.array([0.95047, 1.0, 1.08883], dtype=np.float64)
XYZ_D65_TO_LINEAR_SRGB = np.array(
    [[3.2404542, -1.5371385, -0.4985314],
     [-0.9692660, 1.8760108, 0.0415560],
     [0.0556434, -0.2040259, 1.0572252]], dtype=np.float64
)
BRADFORD = np.array(
    [[0.8951, 0.2664, -0.1614],
     [-0.7502, 1.7135, 0.0367],
     [0.0389, -0.0685, 1.0296]], dtype=np.float64
)


def xyz_white_from_camera_neutral(neutral: np.ndarray, color_matrix2: np.ndarray) -> np.ndarray:
    """Infer the XYZ white represented by AsShotNeutral and ColorMatrix2."""
    matrix = np.asarray(color_matrix2, dtype=np.float64)
    camera_neutral = np.asarray(neutral, dtype=np.float64).reshape(-1)
    if matrix.shape != (3, 3) or camera_neutral.shape != (3,):
        raise RawInputError("ColorMatrix2 must be 3x3 and AsShotNeutral must have 3 values")
    if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(camera_neutral)) or np.any(camera_neutral <= 0):
        raise RawInputError("color metadata contains invalid values")
    try:
        xyz = np.linalg.solve(matrix, camera_neutral)
    except np.linalg.LinAlgError as exc:
        raise RawInputError("ColorMatrix2 cannot be inverted") from exc
    if not np.all(np.isfinite(xyz)) or xyz[1] <= 0:
        raise RawInputError("ColorMatrix2 and AsShotNeutral imply an invalid XYZ white")
    return xyz / xyz[1]


def bradford_adaptation(source_white: np.ndarray, target_white: np.ndarray) -> np.ndarray:
    """Return a 3x3 Bradford chromatic adaptation matrix for XYZ whites."""
    source = np.asarray(source_white, dtype=np.float64).reshape(3)
    target = np.asarray(target_white, dtype=np.float64).reshape(3)
    source_lms = BRADFORD @ source
    target_lms = BRADFORD @ target
    if np.any(np.abs(source_lms) < 1e-12):
        raise RawInputError("source white is invalid for Bradford adaptation")
    return np.linalg.inv(BRADFORD) @ np.diag(target_lms / source_lms) @ BRADFORD


@dataclass(frozen=True)
class ColorTransform:
    """Validated transform provenance and matrix stages for one capture."""

    camera_to_xyz: np.ndarray
    source_white_xyz: np.ndarray
    source_to_d50: np.ndarray
    d50_to_d65: np.ndarray
    xyz_d65_to_linear_srgb: np.ndarray
    as_shot_neutral: np.ndarray
    color_matrix2: np.ndarray

    @property
    def camera_to_linear_srgb(self) -> np.ndarray:
        return self.xyz_d65_to_linear_srgb @ self.d50_to_d65 @ self.source_to_d50 @ self.camera_to_xyz

    def metadata(self) -> dict[str, object]:
        return {
            "ColorMatrix2_convention": "XYZ to reference camera-native RGB (DNG ColorMatrix convention)",
            "ColorMatrix2_storage": "9 coefficients reshaped in DNG row-scan order to 3x3",
            "camera_to_XYZ": "inverse(ColorMatrix2)",
            "white_balance": "Bradford adaptation derived from AsShotNeutral and inverse(ColorMatrix2); no second per-channel gain is applied",
            "source_white_XYZ_Y1": self.source_white_xyz.tolist(),
            "XYZ_target_white": "D65 before linear sRGB matrix",
            "camera_to_XYZ_matrix": self.camera_to_xyz.tolist(),
            "source_white_to_D50_matrix": self.source_to_d50.tolist(),
            "D50_to_D65_matrix": self.d50_to_d65.tolist(),
            "XYZ_D65_to_linear_sRGB_matrix": self.xyz_d65_to_linear_srgb.tolist(),
            "AsShotNeutral": self.as_shot_neutral.tolist(),
            "ColorMatrix2": self.color_matrix2.tolist(),
            "caveat": "CalibrationIlluminant2, ColorMatrix1, camera calibration, and forward-matrix tags are absent; ColorMatrix2-only DNG approximation, not a complete camera profile.",
        }


def build_color_transform(neutral: np.ndarray, color_matrix2: np.ndarray) -> ColorTransform:
    matrix = np.asarray(color_matrix2, dtype=np.float64)
    try:
        camera_to_xyz = np.linalg.inv(matrix)
    except np.linalg.LinAlgError as exc:
        raise RawInputError("ColorMatrix2 cannot be inverted") from exc
    source_white = xyz_white_from_camera_neutral(neutral, matrix)
    source_to_d50 = bradford_adaptation(source_white, D50_XYZ)
    d50_to_d65 = bradford_adaptation(D50_XYZ, D65_XYZ)
    return ColorTransform(camera_to_xyz, source_white, source_to_d50, d50_to_d65,
                          XYZ_D65_TO_LINEAR_SRGB.copy(), np.asarray(neutral).copy(), matrix.copy())


def apply_matrix_rgb(rgb: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Apply a 3x3 transform with one float32 output allocation and no clipping."""
    values = np.asarray(rgb)
    transform = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 3 or values.shape[-1] != 3 or transform.shape != (3, 3):
        raise RawInputError("expected HxWx3 RGB and a 3x3 transform")
    if not np.all(np.isfinite(values)):
        raise RawInputError("RGB input contains non-finite values")
    # Matrix and data are float32 for the image operation; the 3x3 metadata
    # transform is tiny. einsum writes directly to the sole output allocation.
    out = np.empty(values.shape, dtype=np.float32)
    np.einsum("...j,ij->...i", values, transform.astype(np.float32), out=out, optimize=False)
    return out


def apply_matrix_rgb_reference(rgb: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Original B0 matrix implementation retained for regression benchmarking."""
    values = np.asarray(rgb)
    transform = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 3 or values.shape[-1] != 3 or transform.shape != (3, 3):
        raise RawInputError("expected HxWx3 RGB and a 3x3 transform")
    if not np.all(np.isfinite(values)):
        raise RawInputError("RGB input contains non-finite values")
    out = np.empty(values.shape, dtype=np.float32)
    for channel in range(3):
        plane = np.zeros(values.shape[:2], dtype=np.float32)
        for source_channel in range(3):
            plane += values[..., source_channel] * np.float32(transform[channel, source_channel])
        out[..., channel] = plane
    return out
