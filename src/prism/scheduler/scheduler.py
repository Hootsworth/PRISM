"""Adaptive compute scheduler for PRISM hybrid ISP."""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import numpy as np

from ..audit.sensor_audit import AuditResult, TileDiagnostic, TileState


@dataclass
class ScheduleResult:
    selected_tiles: List[TileDiagnostic]
    budget_fraction: float
    eligible_count: int
    scheduled_count: int
    total_tiles: int
    fraction_refined: float
    metadata: Dict[str, Any]


def schedule_tiles(audit_result: AuditResult,
                   budget_fraction: float = 0.5,
                   max_tiles: Optional[int] = None,
                   include_uncertain: bool = False) -> ScheduleResult:
    """Schedule tiles for learned refinement based on audit evidence and compute budget.
    
    Parameters:
        audit_result: Result from sensor-consistency audit.
        budget_fraction: Maximum fraction of all sensor tiles available for
            refinement [0.0, 1.0]. Selection is capped by eligible candidates.
            0.0 schedules none; 1.0 permits all eligible candidates.
        max_tiles: Optional hard cap on the number of tiles to refine.
        include_uncertain: If True, also allow UNCERTAIN tiles (default False).
            This is exploratory selection, not a calibration or safety override.
    """
    total = len(audit_result.tile_diagnostics)
    budget = float(budget_fraction)
    if not 0.0 <= budget <= 1.0:
        raise ValueError("budget_fraction must be in [0,1]")

    # Filter eligible tiles
    allowed_states = {TileState.HIGH_RELATIVE_RESIDUAL_CANDIDATE}
    if include_uncertain:
        allowed_states.add(TileState.UNCERTAIN)

    eligible = [
        d for d in audit_result.tile_diagnostics
        if (d.state in allowed_states and
            (d.state != TileState.UNCERTAIN or d.details.get("relative_rank_allowed", False)))
    ]
    eligible_count = len(eligible)

    if budget <= 0.0 or eligible_count == 0 or (max_tiles is not None and max_tiles <= 0):
        return ScheduleResult(
            selected_tiles=[],
            budget_fraction=budget,
            eligible_count=eligible_count,
            scheduled_count=0,
            total_tiles=total,
            fraction_refined=0.0,
            metadata={"reason": "zero_budget_or_no_eligible"}
        )

    # Sort descending by priority score
    ranked = sorted(eligible, key=lambda d: d.priority_score, reverse=True)

    computed = int(np.round(budget * total))
    target_count = max(1, min(eligible_count, computed))

    if max_tiles is not None:
        target_count = min(target_count, max_tiles)

    selected = ranked[:target_count]

    return ScheduleResult(
        selected_tiles=selected,
        budget_fraction=budget,
        eligible_count=eligible_count,
        scheduled_count=len(selected),
        total_tiles=total,
        fraction_refined=len(selected) / max(total, 1),
        metadata={
            "selection_strategy": "priority_ranked",
            "highest_priority": selected[0].priority_score if selected else 0.0,
            "lowest_selected_priority": selected[-1].priority_score if selected else 0.0
        }
    )
