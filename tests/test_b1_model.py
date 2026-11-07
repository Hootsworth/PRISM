"""Unit and regression tests for B1 architecture, inference and device compatibility."""
import unittest
import torch
from prism.models.b1 import B1PackedUNet
from prism.models.b1_inference import predict_tiled


class B1ModelTests(unittest.TestCase):
    def test_parameter_count_exact(self):
        model = B1PackedUNet()
        param_count = sum(p.numel() for p in model.parameters())
        self.assertEqual(param_count, 357276)

    def test_forward_output_shape_and_bounds(self):
        model = B1PackedUNet().eval()
        # Sensor size 64x80 -> packed size 32x40
        x = torch.rand(2, 4, 32, 40, dtype=torch.float32)
        with torch.inference_mode():
            out = model(x)
        self.assertEqual(out.shape, (2, 3, 64, 80))
        self.assertTrue(torch.isfinite(out).all())
        self.assertGreaterEqual(float(out.min()), 0.0)
        self.assertLessEqual(float(out.max()), 1.0)

    def test_rejection_of_invalid_input(self):
        model = B1PackedUNet()
        with self.assertRaises(ValueError):
            model(torch.rand(1, 3, 32, 32))  # wrong channels
        with self.assertRaises(TypeError):
            model(torch.rand(1, 4, 32, 32, dtype=torch.float64))  # wrong dtype

    def test_predict_tiled_matches_direct_on_aligned_crop(self):
        model = B1PackedUNet().eval()
        # 128x128 sensor -> packed 64x64
        x = torch.rand(1, 4, 64, 64, dtype=torch.float32)
        with torch.inference_mode():
            direct = model(x)[0]
            tiled = predict_tiled(model, x[0], tile_sensor=64, overlap_sensor=16, device='cpu')
        diff = (direct - tiled).abs()
        self.assertLess(float(diff.max()), 0.05)
        self.assertLess(float(diff.mean()), 0.005)

    def test_gpu_device_execution_if_available(self):
        if torch.backends.mps.is_available():
            device = torch.device('mps')
        elif torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            self.skipTest('No GPU accelerator available')

        model = B1PackedUNet().to(device).eval()
        x = torch.rand(1, 4, 32, 32, dtype=torch.float32, device=device)
        with torch.inference_mode():
            out = model(x)
        self.assertEqual(out.device.type, device.type)
        self.assertEqual(out.shape, (1, 3, 64, 64))
        self.assertTrue(torch.isfinite(out).all())


if __name__ == '__main__':
    unittest.main()
