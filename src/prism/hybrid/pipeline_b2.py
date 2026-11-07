"""PRISM Hybrid (B2) Pipeline: Classical ISP + Sensor Audit + Budgeted Neural Refinement."""
from __future__ import annotations
import time
from dataclasses import dataclass
from typing import Optional, Dict, Any
import numpy as np
import torch

from ..isp.raw import RawFrame
from ..isp.pipeline import process_bayer, B0Result
from ..audit.noise import NoiseModel, get_noise_model
from ..audit.sensor_audit import run_sensor_audit, AuditResult
from ..scheduler.scheduler import schedule_tiles, ScheduleResult
from ..refinement.tile_refiner import refine_tiles


@dataclass
class B2Result:
    final_srgb: np.ndarray
    b0_result: B0Result
    initial_audit: AuditResult
    schedule: ScheduleResult
    post_audit: Optional[AuditResult]
    stage_timings: Dict[str, float]
    stats: Dict[str, Any]


def run_b2(frame: RawFrame,
           model: Optional[torch.nn.Module] = None,
           device: Optional[torch.device] = None,
           budget_fraction: float = 0.5,
           tile_sensor_size: int = 256,
           overlap_sensor: int = 32,
           noise_model: Optional[NoiseModel] = None,
           re_audit: bool = False,
           post_camera_rgb_linear: Optional[np.ndarray] = None,
           allow_uncalibrated_selection: bool = False) -> B2Result:
    """Execute complete PRISM Hybrid (B2) reconstruction pipeline.
    
    1. Classical fast path: B0 linear and Reinhard sRGB rendering.
    2. Linear sensor-space consistency audit: re-mosaic and normalize residual.
    3. Tile diagnostics and exploratory relative-residual ranking.
    4. Adaptive scheduling under explicit compute budget [0.0, 1.0].
    5. Local learned tile refinement on scheduled tiles with feather blending.
       Uncalibrated candidates are excluded unless explicit exploratory selection is enabled.
    6. Report the initial audit. A post-audit is only possible when a caller
       supplies a separately derived camera-linear RGB estimate. B2's display
       sRGB output cannot be sent directly to the sensor-domain audit.
    """
    if not 0.0 <= float(budget_fraction) <= 1.0:
        raise ValueError("budget_fraction must be in [0,1] of total sensor tiles")
    total_start = time.perf_counter()
    timings: Dict[str, float] = {}

    # 1. Classical B0 fast path
    t0 = time.perf_counter()
    b0_result = process_bayer(frame, tone_map="reinhard")
    timings["b0_seconds"] = time.perf_counter() - t0

    # 2. Sensor-consistency audit in camera-native linear space
    t0 = time.perf_counter()
    if noise_model is None:
        camera_id = frame.metadata.get("camera_str") or frame.metadata.get("camera")
        iso = frame.metadata.get("iso")
        noise_model = get_noise_model(camera_id=camera_id, iso=iso,
                                      path=frame.input_path, mosaic=frame.mosaic)

    audit_initial = run_sensor_audit(
        mosaic=frame.mosaic,
        camera_rgb=b0_result.camera_rgb_linear,
        cfa_pattern=frame.cfa_pattern,
        noise_model=noise_model,
        tile_sensor_size=tile_sensor_size
    )
    timings["audit_seconds"] = time.perf_counter() - t0

    # 3. Schedule tiles based on audit evidence and budget
    t0 = time.perf_counter()
    schedule = schedule_tiles(
        audit_result=audit_initial,
        budget_fraction=budget_fraction,
        include_uncertain=allow_uncalibrated_selection
    )
    timings["scheduler_seconds"] = time.perf_counter() - t0

    if schedule.scheduled_count > 0 and model is None:
        raise ValueError("scheduled refinement tiles require a trained B1 model")

    # 4. Local learned tile refinement
    t0 = time.perf_counter()
    base_display = np.clip(b0_result.srgb_encoded, 0.0, 1.0)
    if schedule.scheduled_count > 0 and model is not None:
        if device is None:
            device = next(model.parameters()).device
        final_srgb, ref_meta = refine_tiles(
            mosaic=frame.mosaic,
            base_srgb=base_display,
            selected_tiles=schedule.selected_tiles,
            model=model,
            device=device,
            overlap=overlap_sensor
        )
        timings["refinement_seconds"] = ref_meta["refinement_seconds"]
        refined_count = int(ref_meta["tiles_processed"])
    else:
        final_srgb = base_display.copy()
        timings["refinement_seconds"] = 0.0
        refined_count = 0

    # 5. Optional re-audit. The rendered B2 result is display sRGB and cannot
    # be remosaiced as camera-linear sensor data. Never substitute the original
    # B0 estimate here and label it a post-refinement audit.
    post_audit = None
    if re_audit and post_camera_rgb_linear is None:
        raise ValueError("post-audit requires an independently derived camera-linear RGB estimate")
    if re_audit and schedule.scheduled_count > 0:
        t0 = time.perf_counter()
        post_audit = run_sensor_audit(
            mosaic=frame.mosaic,
            camera_rgb=post_camera_rgb_linear,
            cfa_pattern=frame.cfa_pattern,
            noise_model=noise_model,
            tile_sensor_size=tile_sensor_size
        )
        timings["re_audit_seconds"] = time.perf_counter() - t0
    else:
        timings["re_audit_seconds"] = 0.0

    timings["total_seconds"] = time.perf_counter() - total_start

    stats = {
        "budget_fraction_requested": budget_fraction,
        "budget_fraction_of_total_tiles": budget_fraction,
        "total_tiles": schedule.total_tiles,
        "eligible_tiles": schedule.eligible_count,
        "eligible_refinement_fraction_actual": (refined_count / max(schedule.eligible_count, 1)),
        "total_tile_budget_used_fraction": (refined_count / max(schedule.total_tiles, 1)),
        "tiles_scheduled": schedule.scheduled_count,
        "tiles_refined": refined_count,
        "refinement_percentage": (refined_count / max(schedule.total_tiles, 1)) * 100.0,
        "post_audit_status": ("computed_from_supplied_camera_linear_estimate"
                              if post_audit is not None else "not_computed_no_valid_camera_linear_b2_estimate"),
        "selection_mode": ("uncalibrated_relative_residual_experiment"
                           if allow_uncalibrated_selection and not noise_model.calibrated
                           else ("calibrated_relative_residual_candidates" if noise_model.calibrated
                                 else "no_uncalibrated_selection")),
        "noise_model": noise_model.name,
        "tile_size": tile_sensor_size,
        "overlap_sensor": overlap_sensor,
        "device": str(device) if device is not None else "cpu",
    }

    return B2Result(
        final_srgb=final_srgb,
        b0_result=b0_result,
        initial_audit=audit_initial,
        schedule=schedule,
        post_audit=post_audit,
        stage_timings=timings,
        stats=stats
    )
