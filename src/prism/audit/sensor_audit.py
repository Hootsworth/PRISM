"""Exploratory sensor residual diagnostics and tile candidates for PRISM."""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from scipy.ndimage import uniform_filter

from ..isp.demosaic import cfa_color_masks
from .noise import NoiseModel, get_noise_model


class TileState(str, Enum):
    LOW_RELATIVE_RESIDUAL = "LOW_RELATIVE_RESIDUAL"
    HIGH_RELATIVE_RESIDUAL_CANDIDATE = "HIGH_RELATIVE_RESIDUAL_CANDIDATE"
    UNCERTAIN = "UNCERTAIN"
    # Compatibility aliases. These names are deprecated; state values and
    # reports deliberately avoid claims of verification or required repair.
    VERIFIED = "LOW_RELATIVE_RESIDUAL"
    NEEDS_REFINEMENT = "HIGH_RELATIVE_RESIDUAL_CANDIDATE"


@dataclass
class TileDiagnostic:
    tile_index: int
    y: int
    x: int
    height: int
    width: int
    state: TileState
    raw_residual_mean: float
    raw_residual_std: float
    norm_residual_energy: float
    clipped_fraction: float
    near_black_fraction: float
    priority_score: float
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AuditResult:
    tile_diagnostics: List[TileDiagnostic]
    grid_rows: int
    grid_cols: int
    tile_sensor_size: int
    state_map: np.ndarray          # 2D array of state strings
    score_map: np.ndarray          # 2D array of float priority scores
    normalized_residual: np.ndarray # H x W normalized residual
    noise_model: NoiseModel
    summary: Dict[str, Any]


def remosaic(camera_rgb: np.ndarray, cfa_pattern: str) -> np.ndarray:
    """Sample camera-native RGB at Bayer CFA site positions: y_hat = S_CFA(x_cam)."""
    masks = cfa_color_masks(camera_rgb.shape[:2], cfa_pattern)
    return np.sum(camera_rgb * masks, axis=2).astype(np.float32)


def phase_aligned_consensus(mosaic_estimate: np.ndarray) -> np.ndarray:
    """Local 3x3 mean within each of the four Bayer phase sublattices.

    Filtering the full-resolution mosaic directly would mix different CFA
    colors in every neighborhood. Filtering each 2x2 phase plane separately
    keeps comparisons on the same sensor/color sublattice.
    """
    if mosaic_estimate.ndim != 2 or mosaic_estimate.shape[0] % 2 or mosaic_estimate.shape[1] % 2:
        raise ValueError("phase-aligned Bayer consensus requires an even HxW mosaic")
    consensus = np.empty_like(mosaic_estimate, dtype=np.float32)
    for row_phase in (0, 1):
        for col_phase in (0, 1):
            plane = mosaic_estimate[row_phase::2, col_phase::2]
            consensus[row_phase::2, col_phase::2] = uniform_filter(plane, size=3, mode="reflect")
    return consensus


def run_sensor_audit(mosaic: np.ndarray,
                     camera_rgb: np.ndarray,
                     cfa_pattern: str,
                     noise_model: Optional[NoiseModel] = None,
                     tile_sensor_size: int = 256,
                     clip_threshold: float = 0.99,
                     near_black_threshold: float = 0.01,
                     uncertain_clip_rate: float = 0.05,
                     uncertain_black_rate: float = 0.15,
                     significance_percentile: float = 85.0) -> AuditResult:
    """Compute exploratory sensor residual diagnostics in camera-linear space.
    
    1. Re-mosaic the camera RGB estimate at sensor CFA sites.
    2. Compute the residual between mosaic and a local spatial consensus.
    3. Normalize with the supplied affine noise model. Uncalibrated models
       force every tile to UNCERTAIN and are never scheduled.
    4. Compute tile-level diagnostics: variance, energy, clipping, near-black.
    5. When calibrated input is explicitly supplied, label relative residual
       states only; the within-frame percentile is not an FDR or error guarantee.
    """
    if noise_model is None:
        noise_model = get_noise_model()

    h, w = mosaic.shape
    y_hat = remosaic(camera_rgb, cfa_pattern)
    
    # Compare each Bayer site only with its own CFA phase; a full-resolution
    # neighborhood would mix red/green/blue sensor samples.
    consensus = phase_aligned_consensus(y_hat)
    residual = mosaic - consensus
    
    # Normalized residual z_i = r_i / sqrt(a * y_hat + b + eps)
    std = noise_model.std_dev(y_hat)
    z = residual / std

    # Divide image into non-overlapping tiles
    step = tile_sensor_size
    n_rows = int(np.ceil(h / step))
    n_cols = int(np.ceil(w / step))

    tile_diagnostics: List[TileDiagnostic] = []
    state_grid = np.empty((n_rows, n_cols), dtype=object)
    score_grid = np.zeros((n_rows, n_cols), dtype=np.float32)

    tile_energies = []
    candidates = []

    tile_idx = 0
    for r in range(n_rows):
        y0 = r * step
        y1 = min(y0 + step, h)
        th = y1 - y0
        for c in range(n_cols):
            x0 = c * step
            x1 = min(x0 + step, w)
            tw = x1 - x0

            tile_z = z[y0:y1, x0:x1]
            tile_raw = mosaic[y0:y1, x0:x1]
            tile_res = residual[y0:y1, x0:x1]

            n_pixels = th * tw
            clip_count = int(np.count_nonzero(tile_raw >= clip_threshold))
            black_count = int(np.count_nonzero(tile_raw <= near_black_threshold))

            clip_rate = clip_count / n_pixels
            black_rate = black_count / n_pixels

            res_mean = float(np.mean(tile_res))
            res_std = float(np.std(tile_res))
            # Energy of normalized residual
            norm_energy = float(np.mean(tile_z ** 2))
            tile_energies.append(norm_energy)

            candidates.append({
                "tile_idx": tile_idx,
                "r": r, "c": c,
                "y": y0, "x": x0,
                "height": th, "width": tw,
                "res_mean": res_mean,
                "res_std": res_std,
                "norm_energy": norm_energy,
                "clip_rate": clip_rate,
                "black_rate": black_rate
            })
            tile_idx += 1

    # Compute a within-frame relative threshold. This is descriptive only and
    # has no independent error-rate or refinement-benefit calibration.
    valid_energies = [
        item["norm_energy"] for item in candidates
        if item["clip_rate"] <= uncertain_clip_rate and item["black_rate"] <= uncertain_black_rate
    ]
    if valid_energies:
        energy_thresh = float(np.percentile(valid_energies, significance_percentile))
    else:
        energy_thresh = 1.5

    low_residual_count = 0
    high_residual_count = 0
    uncertain_count = 0

    for item in candidates:
        r, c = item["r"], item["c"]
        clip_rate = item["clip_rate"]
        black_rate = item["black_rate"]
        energy = item["norm_energy"]
        relative_rank_allowed = clip_rate <= uncertain_clip_rate and black_rate <= uncertain_black_rate

        # Classification rules preserve uncertainty without suppressing a
        # clearly labeled exploratory ranking. No uncalibrated tile is marked
        # verified or actionable by default.
        if clip_rate > uncertain_clip_rate or black_rate > uncertain_black_rate:
            state = TileState.UNCERTAIN
            priority = 0.0
            uncertain_count += 1
        elif not noise_model.calibrated:
            state = TileState.UNCERTAIN
            priority = max(energy - energy_thresh, 0.0)
            uncertain_count += 1
        elif energy > energy_thresh:
            state = TileState.HIGH_RELATIVE_RESIDUAL_CANDIDATE
            priority = energy - energy_thresh
            high_residual_count += 1
        else:
            state = TileState.LOW_RELATIVE_RESIDUAL
            priority = 0.0
            low_residual_count += 1

        diag = TileDiagnostic(
            tile_index=item["tile_idx"],
            y=item["y"],
            x=item["x"],
            height=item["height"],
            width=item["width"],
            state=state,
            raw_residual_mean=item["res_mean"],
            raw_residual_std=item["res_std"],
            norm_residual_energy=energy,
            clipped_fraction=clip_rate,
            near_black_fraction=black_rate,
            priority_score=priority,
            details={"energy_threshold": energy_thresh,
                     "relative_rank_allowed": relative_rank_allowed,
                     "relative_rank_calibrated": noise_model.calibrated}
        )
        tile_diagnostics.append(diag)
        state_grid[r, c] = state.value
        score_grid[r, c] = priority

    summary = {
        "total_tiles": len(tile_diagnostics),
        "low_relative_residual_tiles": low_residual_count,
        "high_relative_residual_candidates": high_residual_count,
        "uncertain_tiles": uncertain_count,
        "high_relative_residual_fraction": high_residual_count / max(len(tile_diagnostics), 1),
        "noise_model": noise_model.name,
        "noise_model_calibrated": noise_model.calibrated,
        "heuristic_rank_eligible_tiles": sum(bool(d.details.get("relative_rank_allowed"))
                                             for d in tile_diagnostics),
        "above_relative_threshold_tiles": sum(d.priority_score > 0 for d in tile_diagnostics),
        "energy_threshold": energy_thresh,
        "threshold_method": "within-frame percentile of non-extreme tile energies",
        "threshold_validation": "not calibrated against independent tile-error labels; no FDR guarantee"
    }

    return AuditResult(
        tile_diagnostics=tile_diagnostics,
        grid_rows=n_rows,
        grid_cols=n_cols,
        tile_sensor_size=tile_sensor_size,
        state_map=state_grid,
        score_map=score_grid,
        normalized_residual=z,
        noise_model=noise_model,
        summary=summary
    )
