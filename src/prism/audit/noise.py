"""Noise calibration and variance modeling for PRISM sensor-consistency audit."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import math
import re
from pathlib import Path
import numpy as np


@dataclass(frozen=True)
class NoiseModel:
    """Affine noise variance model: Var(n | x) = a * x + b.
    
    Parameters:
        a: Photon shot noise slope (scale factor)
        b: Read noise variance intercept
        name: Profile or camera identifier
        calibrated: True only when calibration has been established from
            independent sensor measurements and its provenance is recorded.
    """
    a: float
    b: float
    name: str = "default"
    calibrated: bool = False
    calibration_reference: Optional[str] = None

    def __post_init__(self):
        if not math.isfinite(float(self.a)) or not math.isfinite(float(self.b)) or self.a < 0 or self.b < 0:
            raise ValueError("noise variance coefficients must be finite and nonnegative")
        if self.calibrated and not (self.calibration_reference and self.calibration_reference.strip()):
            raise ValueError("calibrated noise model requires a calibration reference/provenance")

    def variance(self, signal: np.ndarray) -> np.ndarray:
        """Compute pixel-wise noise variance for given signal level."""
        sig = np.maximum(signal, 0.0)
        return self.a * sig + self.b

    def std_dev(self, signal: np.ndarray, eps: float = 1e-7) -> np.ndarray:
        """Compute pixel-wise standard deviation for normalization."""
        var = self.variance(signal)
        return np.sqrt(np.maximum(var, eps))


# Provisional diagnostic coefficients only. These values have no recorded
# independent calibration source in this project and must not classify tiles
# as sensor-verified or actionable. Keep calibrated=False until a calibration
# report establishes the noise law for the relevant camera/ISO/domain.
PROVISIONAL_NOISE_COEFFICIENTS: Dict[str, NoiseModel] = {
    "IP": NoiseModel(a=0.00045, b=0.00012, name="iPhone7_provisional", calibrated=False),
    "GP": NoiseModel(a=0.00052, b=0.00015, name="GooglePixel_provisional", calibrated=False),
    "S6": NoiseModel(a=0.00048, b=0.00014, name="SamsungS6_provisional", calibrated=False),
    "N6": NoiseModel(a=0.00055, b=0.00018, name="Nexus6_provisional", calibrated=False),
    "G4": NoiseModel(a=0.00050, b=0.00016, name="LGG4_provisional", calibrated=False),
    "GENERIC": NoiseModel(a=0.00050, b=0.00015, name="GenericBayer_provisional", calibrated=False),
}


def parse_path_camera_iso(path: Path | str) -> Tuple[Optional[str], Optional[float]]:
    """Attempt to parse camera code and ISO setting from standard SIDD directory name."""
    s = str(path)
    match = re.search(r'_([A-Z0-9]{2})_(\d{5})_', s)
    if match:
        cam_code = match.group(1)
        iso_val = float(match.group(2))
        return cam_code, iso_val
    for cam in ["IP", "GP", "S6", "N6", "G4"]:
        if f"_{cam}_" in s:
            return cam, None
    return None, None


def get_noise_model(camera_id: Optional[str] = None,
                    iso: Optional[float] = None,
                    path: Optional[Path | str] = None,
                    mosaic: Optional[np.ndarray] = None) -> NoiseModel:
    """Lookup or scale noise model based on camera model, ISO, path, or frame statistics."""
    if (camera_id is None or iso is None) and path is not None:
        p_cam, p_iso = parse_path_camera_iso(path)
        if camera_id is None:
            camera_id = p_cam
        if iso is None:
            iso = p_iso

    if camera_id and camera_id.upper() in PROVISIONAL_NOISE_COEFFICIENTS:
        base = PROVISIONAL_NOISE_COEFFICIENTS[camera_id.upper()]
    elif mosaic is not None:
        return estimate_noise_parameters(mosaic)
    else:
        base = PROVISIONAL_NOISE_COEFFICIENTS["GENERIC"]

    if iso is not None and iso > 0:
        # Scale shot noise and read noise by ISO ratio relative to ISO 100 baseline
        iso_ratio = max(float(iso) / 100.0, 1.0)
        scaled_a = base.a * np.sqrt(iso_ratio)
        scaled_b = base.b * (iso_ratio ** 1.5)
        return NoiseModel(a=scaled_a, b=scaled_b, name=f"{base.name}_ISO{int(iso)}", calibrated=False)

    return base


def estimate_noise_parameters(mosaic: np.ndarray) -> NoiseModel:
    """Estimate a, b from image residual variance in local flat patches."""
    arr = np.asarray(mosaic, dtype=np.float32)
    h, w = arr.shape
    block_size = 16
    means = []
    variances = []
    for y in range(0, h - block_size, block_size):
        for x in range(0, w - block_size, block_size):
            patch = arr[y:y+block_size, x:x+block_size]
            p_mean = float(np.mean(patch))
            p_var = float(np.var(patch))
            if 0.02 < p_mean < 0.8:
                means.append(p_mean)
                variances.append(p_var)
    if len(means) >= 10:
        means = np.array(means)
        variances = np.array(variances)
        p10_var = np.percentile(variances, 25)
        b = max(float(p10_var * 0.5), 1e-5)
        a = max(float(p10_var / (np.median(means) + 1e-4)), 1e-4)
        return NoiseModel(a=a, b=b, name="unvalidated_frame_statistic", calibrated=False)
    return NoiseModel(a=0.0005, b=0.00015, name="unvalidated_generic_estimate", calibrated=False)
