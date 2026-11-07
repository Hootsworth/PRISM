"""Unit tests for deterministic B0 demosaic, metadata and render stages."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.io import savemat

from prism.isp import load_sidd_cc_mat, process_bayer, process_bayer_reference
from prism.isp.color import (D50_XYZ, D65_XYZ, XYZ_D65_TO_LINEAR_SRGB, apply_matrix_rgb,
                             bradford_adaptation,
                             apply_matrix_rgb_reference, build_color_transform)
from prism.isp.demosaic import bilinear_demosaic, cfa_color_masks
from prism.isp.raw import RawInputError
from prism.isp.render import encode_srgb, tone_map_reinhard, to_uint8_srgb


class DemosaicTests(unittest.TestCase):
    def test_each_verified_cfa_preserves_measured_sites_and_channels(self):
        constants = {"R": 0.25, "G": 0.5, "B": 0.75}
        for pattern in ("BGGR", "GRBG", "RGGB"):
            with self.subTest(pattern=pattern):
                masks = cfa_color_masks((7, 9), pattern)
                self.assertTrue(np.all(masks.sum(axis=-1) == 1))
                mosaic = sum(masks[..., c].astype(np.float32) * constants["RGB"[c]] for c in range(3))
                output = bilinear_demosaic(mosaic, pattern)
                self.assertEqual(output.shape, (7, 9, 3))
                self.assertTrue(np.all(np.isfinite(output)))
                for channel, code in enumerate("RGB"):
                    np.testing.assert_array_equal(output[..., channel][masks[..., channel]],
                                                  mosaic[masks[..., channel]])
                    self.assertAlmostEqual(float(output[3, 4, channel]), constants[code], places=6)
                # Constant-color fixture also checks every reflective boundary pixel.
                np.testing.assert_allclose(output,
                    np.broadcast_to(np.array([0.25, 0.5, 0.75], dtype=np.float32), output.shape),
                    atol=1e-6)

    def test_unsupported_cfa_rejected(self):
        with self.assertRaises(RawInputError):
            bilinear_demosaic(np.zeros((4, 4), np.float32), "GBRG")

    def test_negative_raw_values_are_not_clipped(self):
        mosaic = np.full((8, 8), -0.05, np.float32)
        rgb = bilinear_demosaic(mosaic, "RGGB")
        self.assertLess(float(rgb.min()), 0.0)


class ColorTests(unittest.TestCase):
    def test_as_shot_neutral_maps_to_d65_white(self):
        neutral = np.array([0.4, 1.0, 0.6])
        matrix = np.array([[1.2, -0.3, -0.2], [-0.4, 1.5, 0.1], [-0.1, 0.2, 0.6]])
        transform = build_color_transform(neutral, matrix)
        rgb = neutral.reshape(1, 1, 3).astype(np.float32)
        output = apply_matrix_rgb(rgb, transform.camera_to_linear_srgb)
        self.assertTrue(np.all(output[0, 0] > 0))
        np.testing.assert_allclose(output[0, 0], np.repeat(output[0, 0, 1], 3), rtol=2e-5, atol=2e-5)

    def test_singular_color_matrix_rejected(self):
        with self.assertRaises(RawInputError):
            build_color_transform(np.ones(3), np.ones((3, 3)))

    def test_transform_preserves_negative_linear_values(self):
        rgb = np.array([[[-0.1, 0.2, 0.3]]], dtype=np.float32)
        out = apply_matrix_rgb(rgb, np.eye(3))
        self.assertLess(float(out[0, 0, 0]), 0.0)

    def test_matrix_inverse_and_vectorized_matches_reference(self):
        matrix = np.array([[1.1, -0.2, 0.1], [-0.4, 1.7, 0.2], [0.03, -0.1, 0.8]])
        inverse = np.linalg.inv(matrix)
        np.testing.assert_allclose(inverse @ matrix, np.eye(3), atol=1e-12)
        rng = np.random.default_rng(42)
        rgb = rng.normal(0, 0.2, (19, 23, 3)).astype(np.float32)
        fast = apply_matrix_rgb(rgb, matrix)
        ref = apply_matrix_rgb_reference(rgb, matrix)
        np.testing.assert_allclose(fast, ref, rtol=2e-6, atol=2e-7)

    def test_xyz_d65_white_maps_to_linear_srgb_white(self):
        xyz = np.broadcast_to(D65_XYZ.astype(np.float32), (1, 1, 3)).copy()
        rgb = apply_matrix_rgb(xyz, XYZ_D65_TO_LINEAR_SRGB)
        np.testing.assert_allclose(rgb[0, 0], np.ones(3), atol=1e-5)

    def test_bradford_maps_d50_white_to_d65_white(self):
        adaptation = bradford_adaptation(D50_XYZ, D65_XYZ)
        np.testing.assert_allclose(adaptation @ D50_XYZ, D65_XYZ, atol=1e-12)

    def test_known_srgb_oetf_values(self):
        linear = np.array([[[0.0, 0.0031308, 1.0]]], dtype=np.float32)
        encoded = encode_srgb(linear)
        np.testing.assert_allclose(encoded[0, 0], [0.0, 0.0404499, 1.0], atol=1e-6)


class RenderTests(unittest.TestCase):
    def test_tone_mapping_and_oetf_are_separate(self):
        linear = np.array([[[-0.1, 0.18, 1.0]]], dtype=np.float32)
        mapped = tone_map_reinhard(linear)
        encoded = encode_srgb(mapped)
        self.assertFalse(np.array_equal(mapped, encoded))
        self.assertLess(float(mapped[0, 0, 0]), 0.0)
        self.assertEqual(to_uint8_srgb(encoded).shape, (1, 1, 3))


class InputAndPipelineTests(unittest.TestCase):
    def _write_mat_pair(self, root: Path, cfa="RGGB", metadata=None):
        raw = np.tile(np.array([[0.25, 0.5], [0.5, 0.75]], dtype=np.float32), (4, 4))
        raw[0, 0] = -0.02
        raw[0, 1] = 1.25
        raw_path = root / "0001_NOISY_RAW_001.MAT"
        meta_path = root / "0001_METADATA_RAW_001.MAT"
        savemat(str(raw_path), {"data": raw})
        fields = {
            "BlackLevel": np.array([[64.0]]), "WhiteLevel": np.array([[1023.0]]),
            "AsShotNeutral": np.array([[0.5, 1.0, 0.75]]),
            "ColorMatrix2": np.array([[1.2, -0.3, -0.2, -0.4, 1.5, 0.1, -0.1, 0.2, 0.6]]),
            "cfa_str": np.array([cfa]),
        }
        if metadata is not None:
            fields.update(metadata)
        savemat(str(meta_path), fields)
        return raw_path, meta_path, raw

    def test_sidd_mat_load_preserves_normalization_and_negative_values(self):
        with tempfile.TemporaryDirectory() as temp:
            raw_path, _, raw = self._write_mat_pair(Path(temp))
            frame = load_sidd_cc_mat(raw_path)
            self.assertEqual(frame.mosaic.dtype, np.float32)
            self.assertEqual(frame.mosaic.shape, raw.shape)
            self.assertAlmostEqual(float(frame.mosaic[0, 0]), -0.02, places=7)
            self.assertAlmostEqual(float(frame.mosaic[0, 1]), 1.25, places=7)
            self.assertEqual(frame.saturation_candidate_count, 1)
            self.assertTrue(frame.saturation_candidate_mask[0, 1])
            self.assertEqual(frame.black_level_original, 64.0)
            self.assertFalse(frame.mosaic.flags.writeable)

    def test_missing_metadata_and_unsupported_cfa_fail_clearly(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw_path, meta_path, _ = self._write_mat_pair(root, cfa="GBRG")
            with self.assertRaisesRegex(RawInputError, "unsupported CFA"):
                load_sidd_cc_mat(raw_path)
            meta_path.unlink()
            with self.assertRaisesRegex(RawInputError, "metadata MAT file does not exist"):
                load_sidd_cc_mat(raw_path)

    def test_missing_required_metadata_field_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw_path, meta_path, _ = self._write_mat_pair(root)
            savemat(str(meta_path), {"cfa_str": np.array(["RGGB"])})
            with self.assertRaisesRegex(RawInputError, "AsShotNeutral"):
                load_sidd_cc_mat(raw_path)

    def test_nonfinite_raw_and_malformed_color_matrix_fail(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw_path, meta_path, raw = self._write_mat_pair(root)
            bad_raw = raw.copy(); bad_raw[2, 2] = np.nan
            savemat(str(raw_path), {"data": bad_raw})
            with self.assertRaisesRegex(RawInputError, "finite"):
                load_sidd_cc_mat(raw_path)
            savemat(str(raw_path), {"data": raw})
            savemat(str(meta_path), {"BlackLevel": 0.0, "WhiteLevel": 1.0,
                "AsShotNeutral": [[0.5, 1.0, 0.75]], "ColorMatrix2": [[1, 2, 3]],
                "cfa_str": np.array(["RGGB"])})
            with self.assertRaisesRegex(RawInputError, "nine finite values"):
                load_sidd_cc_mat(raw_path)

    def test_pipeline_preserves_dark_and_overrange_measurements(self):
        with tempfile.TemporaryDirectory() as temp:
            raw_path, _, raw = self._write_mat_pair(Path(temp))
            frame = load_sidd_cc_mat(raw_path)
            before = frame.mosaic.copy()
            self.assertLess(float(before[0, 0]), 0.0)
            self.assertGreater(float(before[0, 1]), 1.0)
            result = process_bayer(frame, tone_map=None)
            np.testing.assert_array_equal(frame.mosaic, before)
            self.assertTrue(np.isfinite(result.linear_srgb).all())

    def test_full_pipeline_renders_neutral_gray_neutral(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw_path, meta_path, _ = self._write_mat_pair(root)
            meta = load_sidd_cc_mat(raw_path)
            masks = cfa_color_masks(meta.mosaic.shape, meta.cfa_pattern)
            neutral = meta.as_shot_neutral.astype(np.float32)
            mosaic = np.zeros(meta.mosaic.shape, dtype=np.float32)
            for channel in range(3):
                mosaic[masks[..., channel]] = neutral[channel]
            savemat(str(raw_path), {"data": mosaic})
            frame = load_sidd_cc_mat(raw_path, meta_path)
            result = process_bayer(frame, tone_map=None)
            center = result.linear_srgb[3:-3, 3:-3]
            spread = np.max(center, axis=-1) - np.min(center, axis=-1)
            self.assertLess(float(np.max(spread)), 2e-5)

    def test_complete_pipeline_shape_finite_and_channel_order(self):
        with tempfile.TemporaryDirectory() as temp:
            raw_path, _, _ = self._write_mat_pair(Path(temp))
            frame = load_sidd_cc_mat(raw_path)
            before = frame.mosaic.copy()
            result = process_bayer(frame, tone_map=None)
            reference = process_bayer_reference(frame, tone_map=None)
            self.assertEqual(result.linear_srgb.shape, (*frame.mosaic.shape, 3))
            self.assertTrue(np.all(np.isfinite(result.linear_srgb)))
            np.testing.assert_allclose(result.camera_rgb_linear, reference.camera_rgb_linear, atol=0)
            np.testing.assert_allclose(result.xyz_d50, reference.xyz_d50, atol=1e-6, rtol=2e-6)
            np.testing.assert_allclose(result.linear_srgb, reference.linear_srgb, atol=2e-6, rtol=2e-6)
            np.testing.assert_array_equal(frame.mosaic, before)
            self.assertEqual(result.camera_rgb_linear.shape[-1], 3)


if __name__ == "__main__":
    unittest.main()
