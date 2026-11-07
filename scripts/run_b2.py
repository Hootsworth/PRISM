#!/usr/bin/env python3
"""Execute and profile PRISM Hybrid (B2) pipeline on a verified Bayer RAW input."""
from __future__ import annotations
import argparse, hashlib, json, platform, sys, time
from pathlib import Path
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from prism.isp.raw import load_sidd_cc_mat
from prism.models.b1 import B1PackedUNet
from prism.hybrid.pipeline_b2 import run_b2
from prism.audit.sensor_audit import TileState


def save_heatmap(path: Path, state_map: np.ndarray, score_map: np.ndarray):
    """Save a colored PNG visualizing tile audit states and priorities."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows, cols = state_map.shape
    # Palette: green=low relative residual, red=elevated residual candidate,
    # yellow=uncertain. These colors do not imply accuracy or repair truth.
    img = np.zeros((rows, cols, 3), dtype=np.uint8)
    for r in range(rows):
        for c in range(cols):
            st = state_map[r, c]
            if st == TileState.LOW_RELATIVE_RESIDUAL.value:
                img[r, c] = [50, 200, 50]  # Green
            elif st == TileState.HIGH_RELATIVE_RESIDUAL_CANDIDATE.value:
                intensity = min(255, int(150 + score_map[r, c] * 50))
                img[r, c] = [intensity, 40, 40]  # Red
            else:
                img[r, c] = [220, 200, 50]  # Yellow (UNCERTAIN)
    # Scale up for visibility
    scale = 16
    scaled = np.repeat(np.repeat(img, scale, axis=0), scale, axis=1)
    Image.fromarray(scaled).save(path)


def main():
    parser = argparse.ArgumentParser(description='Execute PRISM Hybrid (B2) pipeline')
    parser.add_argument('--raw', type=Path, default=None,
                        help='Path to input RAW .mat file')
    parser.add_argument('--metadata', type=Path, default=None,
                        help='Path to paired metadata .mat file')
    parser.add_argument('--checkpoint', type=Path, required=True,
                        help='Explicit B1 refiner checkpoint; no implicit/random fallback')
    parser.add_argument('--budget', type=float, default=0.5,
                        help='Maximum fraction [0,1] of all tiles available for refinement; selection is capped by eligible tiles')
    parser.add_argument('--allow-uncalibrated-selection', action='store_true',
                        help='Experimental residual ranking on non-extreme uncertain tiles; not sensor-calibrated')
    parser.add_argument('--device', default='auto', choices=('auto', 'cpu', 'cuda', 'mps'))
    parser.add_argument('--tile-size', type=int, default=256)
    parser.add_argument('--overlap', type=int, default=32)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'experiments/P3/B2_outputs')
    parser.add_argument('--save-images', action='store_true', help='Save output PNG images')
    args = parser.parse_args()

    # Default RAW if not specified
    if args.raw is None:
        candidates = list((ROOT / 'data/external/sidd_cc_expanded/Raw/train/Data').glob('*/*_NOISY_RAW_*.MAT'))
        if not candidates:
            candidates = list((ROOT / 'data/external/sidd_subset/Raw/train/Data').glob('*/*_NOISY_RAW_*.MAT'))
        if not candidates:
            parser.error('No RAW file specified and none found in data directories')
        args.raw = candidates[0]

    # Select device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    else:
        device = torch.device(args.device)
    if device.type == 'mps' and not torch.backends.mps.is_available():
        parser.error('MPS requested but unavailable in this Python runtime')
    if device.type == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA requested but unavailable')
    if not 0.0 <= args.budget <= 1.0:
        parser.error('--budget must be a fraction in [0,1] of total sensor tiles')

    print('Loading RAW frame from: {}'.format(args.raw), flush=True)
    frame = load_sidd_cc_mat(args.raw, args.metadata)
    print('RAW shape: {}, CFA: {}, Neutral: {}'.format(
        frame.mosaic.shape, frame.cfa_pattern, frame.as_shot_neutral), flush=True)

    # A B2 inference result must identify a real trained checkpoint. Never
    # silently run with random initialization or an implicit fallback.
    model = B1PackedUNet()
    checkpoint_path = args.checkpoint
    if checkpoint_path is None or not checkpoint_path.is_file():
        parser.error('A valid trained B1 checkpoint is required; pass --checkpoint explicitly')
    print('Loading B1 checkpoint: {}'.format(checkpoint_path), flush=True)
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if 'model_state_dict' not in ckpt:
        parser.error('Checkpoint lacks model_state_dict: {}'.format(checkpoint_path))
    model.load_state_dict(ckpt['model_state_dict'], strict=True)

    model.to(device).eval()
    print('Executing B2 hybrid pipeline on device: {} with budget: {:.0%}...'.format(
        device, args.budget), flush=True)

    result = run_b2(
        frame=frame,
        model=model,
        device=device,
        budget_fraction=args.budget,
        tile_sensor_size=args.tile_size,
        overlap_sensor=args.overlap,
        allow_uncalibrated_selection=args.allow_uncalibrated_selection
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Save images if requested
    if args.save_images:
        b0_u8 = np.rint(np.clip(result.b0_result.srgb_encoded, 0, 1) * 255).astype(np.uint8)
        b2_u8 = np.rint(np.clip(result.final_srgb, 0, 1) * 255).astype(np.uint8)
        Image.fromarray(b0_u8).save(args.output_dir / 'b0_classical.png')
        Image.fromarray(b2_u8).save(args.output_dir / 'b2_hybrid.png')
        save_heatmap(args.output_dir / 'audit_heatmap.png',
                     result.initial_audit.state_map,
                     result.initial_audit.score_map)
        print('Saved outputs to: {}'.format(args.output_dir), flush=True)

    report = {
        "pipeline": "B2_PRISM_Hybrid",
        "raw_file": str(args.raw),
        "cfa_pattern": frame.cfa_pattern,
        "device": str(device),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        "checkpoint_epoch": ckpt.get('epoch'),
        "checkpoint_config": ckpt.get('config'),
        "platform": platform.platform(),
        "budget_fraction": args.budget,
        "timings": result.stage_timings,
        "stats": result.stats,
        "audit_summary": result.initial_audit.summary,
        "noise_calibrated": result.initial_audit.noise_model.calibrated,
        "selection_mode": result.stats['selection_mode'],
        "post_audit_status": result.stats['post_audit_status'],
    }

    summary_path = args.output_dir / 'b2_run_summary.json'
    summary_path.write_text(json.dumps(report, indent=2) + '\n')

    print('\n=== B2 Hybrid Execution Report ===')
    print('Total tiles:           {}'.format(result.stats['total_tiles']))
    print('Eligible (refinement): {}'.format(result.stats['eligible_tiles']))
    print('Tiles refined:         {} ({:.1f}%)'.format(
        result.stats['tiles_refined'], result.stats['refinement_percentage']))
    print('B0 fast-path time:     {:.3f} s'.format(result.stage_timings['b0_seconds']))
    print('Audit overhead:        {:.3f} s'.format(result.stage_timings['audit_seconds']))
    print('Scheduler time:        {:.4f} s'.format(result.stage_timings['scheduler_seconds']))
    print('Refinement time:       {:.3f} s'.format(result.stage_timings['refinement_seconds']))
    print('Total pipeline time:   {:.3f} s'.format(result.stage_timings['total_seconds']))
    print('Summary written to:    {}'.format(summary_path))


if __name__ == '__main__':
    main()
