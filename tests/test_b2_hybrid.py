"""Unit and integration tests for PRISM Hybrid (B2) pipeline."""
import unittest
from pathlib import Path
import numpy as np
import torch

from prism.isp.raw import RawFrame
from prism.models.b1 import B1PackedUNet
from prism.audit.noise import NoiseModel, get_noise_model
from prism.audit.sensor_audit import run_sensor_audit, TileState
from prism.scheduler.scheduler import schedule_tiles
from prism.refinement.tile_refiner import refine_tiles
from prism.hybrid.pipeline_b2 import run_b2


class B2HybridTests(unittest.TestCase):
    def setUp(self):
        # Create a synthetic 128x128 Bayer RGGB RawFrame for fast testing
        np.random.seed(20261106)
        h, w = 128, 128
        mosaic = np.random.uniform(0.1, 0.6, (h, w)).astype(np.float32)
        mosaic.setflags(write=False)
        self.frame = RawFrame(
            mosaic=mosaic,
            cfa_pattern="RGGB",
            as_shot_neutral=np.array([1.0, 1.0, 1.0], dtype=np.float64),
            color_matrix2=np.eye(3, dtype=np.float64),
            black_level_original=0.0,
            white_level_original=1.0,
            metadata_path=Path("dummy_meta.mat"),
            input_path=Path("dummy_raw.mat"),
            metadata={"camera": "TEST", "iso": 100}
        )
        self.model = B1PackedUNet().eval()

    def test_zero_budget_gives_exact_b0_output(self):
        result = run_b2(self.frame, model=self.model, budget_fraction=0.0, tile_sensor_size=64)
        self.assertEqual(result.schedule.scheduled_count, 0)
        self.assertEqual(result.stats["tiles_refined"], 0)
        np.testing.assert_array_equal(result.final_srgb, np.clip(result.b0_result.srgb_encoded, 0.0, 1.0))

    def test_full_budget_processes_all_eligible_tiles(self):
        result = run_b2(self.frame, model=self.model, budget_fraction=1.0, tile_sensor_size=64)
        eligible = result.schedule.eligible_count
        self.assertEqual(result.schedule.scheduled_count, eligible)
        self.assertEqual(result.stats["tiles_refined"], eligible)
        self.assertEqual(result.final_srgb.shape, (128, 128, 3))
        self.assertTrue(np.all(np.isfinite(result.final_srgb)))
        self.assertTrue(np.all(result.final_srgb >= 0.0))
        self.assertTrue(np.all(result.final_srgb <= 1.0))

    def test_clipped_and_near_black_tiles_marked_uncertain(self):
        h, w = 128, 128
        mosaic = np.random.uniform(0.2, 0.5, (h, w)).astype(np.float32)
        # Force top-left tile to be clipped (>= 1.0)
        mosaic[0:64, 0:64] = 1.0
        # Force bottom-right tile to be near-black (<= 0.0)
        mosaic[64:128, 64:128] = 0.0
        mosaic.setflags(write=False)

        frame = RawFrame(
            mosaic=mosaic,
            cfa_pattern="RGGB",
            as_shot_neutral=np.array([1.0, 1.0, 1.0]),
            color_matrix2=np.eye(3),
            black_level_original=0.0,
            white_level_original=1.0,
            metadata_path=Path("dummy.mat"),
            input_path=Path("dummy.mat"),
            metadata={}
        )
        result = run_b2(frame, model=self.model, budget_fraction=1.0, tile_sensor_size=64)
        state_map = result.initial_audit.state_map
        # Top-left and bottom-right tiles must be UNCERTAIN
        self.assertEqual(state_map[0, 0], TileState.UNCERTAIN.value)
        self.assertEqual(state_map[1, 1], TileState.UNCERTAIN.value)
        # UNCERTAIN tiles should NOT be scheduled by default
        for tile in result.schedule.selected_tiles:
            self.assertNotEqual(tile.state, TileState.UNCERTAIN)

    def test_uncalibrated_noise_fallback(self):
        # When noise model is explicitly uncalibrated, all tiles should be UNCERTAIN
        uncal_model = NoiseModel(a=0.001, b=0.001, name="uncalibrated", calibrated=False)
        result = run_b2(self.frame, model=self.model, noise_model=uncal_model,
                        budget_fraction=1.0, tile_sensor_size=64)
        for tile in result.initial_audit.tile_diagnostics:
            self.assertEqual(tile.state, TileState.UNCERTAIN)
        # Because no tiles are eligible, schedule count should be 0
        self.assertEqual(result.schedule.scheduled_count, 0)

    def test_b2_gpu_execution_if_available(self):
        if torch.backends.mps.is_available():
            device = torch.device('mps')
        elif torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            self.skipTest('No GPU accelerator available')

        model = B1PackedUNet().to(device)
        result = run_b2(self.frame, model=model, device=device, budget_fraction=0.5, tile_sensor_size=64)
        self.assertEqual(result.final_srgb.shape, (128, 128, 3))
        self.assertTrue(np.all(np.isfinite(result.final_srgb)))
        self.assertIn("refinement_seconds", result.stage_timings)
        self.assertIn("total_seconds", result.stage_timings)


if __name__ == '__main__':
    unittest.main()
