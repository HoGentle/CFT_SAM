"""验证去洞地面面积、外边界平滑及提示／无效像元约束。"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from affine import Affine
from core.sam_postprocess import process_regions, validate_postprocess, _ring_area_m2, _rasterize_regions
from core.sam_roi import mask_to_regions


class PostprocessTests(unittest.TestCase):
    def setUp(self):
        self.mask = np.zeros((80, 80), dtype=bool)
        self.mask[5:75, 5:75] = True
        self.mask[20:22, 20:22] = False
        self.mask[40:46, 40:46] = False
        self.coords = np.array([[10, 10]], dtype=float)
        self.labels = np.array([1])
        self.valid = np.ones_like(self.mask)
        self.affine = Affine(2, 0, 500000, 0, -2, 3300000)

    def process(self, settings, excluded=None):
        regions = mask_to_regions(self.mask, self.coords, self.labels, excluded)
        result = process_regions(regions, self.affine, "EPSG:32649", self.coords, self.labels,
                                 self.valid, excluded or [], validate_postprocess(settings))
        return _rasterize_regions(result, self.mask.shape), result

    def test_disabled_options_preserve_geometry(self):
        actual, _ = self.process({})
        np.testing.assert_array_equal(actual, self.mask)

    def test_holes_strictly_below_ground_area_are_filled(self):
        actual, regions = self.process({"fill_holes": True, "max_hole_area_m2": 16.1})
        expected = self.mask.copy()
        expected[20:22, 20:22] = True
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(len(regions[0]["holes"]), 1)
        actual, _ = self.process({"fill_holes": True, "max_hole_area_m2": 16})
        np.testing.assert_array_equal(actual, self.mask)

    def test_hole_with_negative_prompt_is_kept(self):
        self.coords = np.array([[10, 10], [20, 20]])
        self.labels = np.array([1, 0])
        actual, _ = self.process({"fill_holes": True, "max_hole_area_m2": 17})
        np.testing.assert_array_equal(actual, self.mask)

    def test_invalid_pixels_and_completed_regions_are_not_filled(self):
        self.valid[20, 20] = False
        actual, _ = self.process({"fill_holes": True, "max_hole_area_m2": 17})
        np.testing.assert_array_equal(actual, self.mask)
        self.valid[:] = True
        excluded = [{"hull": [[20, 20], [22, 20], [22, 22], [20, 22]], "holes": []}]
        actual, _ = self.process({"fill_holes": True, "max_hole_area_m2": 17}, excluded)
        np.testing.assert_array_equal(actual, self.mask)

    def test_smoothing_reduces_zigzags_and_preserves_holes(self):
        self.mask[5:9, 8:72:4] = False
        actual, _ = self.process({"smooth_boundary": True})
        perimeter = lambda mask: np.count_nonzero(mask[1:] != mask[:-1]) + np.count_nonzero(mask[:, 1:] != mask[:, :-1])
        self.assertLess(perimeter(actual), perimeter(self.mask))
        np.testing.assert_array_equal(actual[20:22, 20:22], False)
        np.testing.assert_array_equal(actual[40:46, 40:46], False)
        self.assertTrue(actual[10, 10])

    def test_smoothing_and_filling_can_be_combined(self):
        actual, _ = self.process({"smooth_boundary": True, "fill_holes": True, "max_hole_area_m2": 17})
        self.assertTrue(actual[20:22, 20:22].all())
        self.assertFalse(actual[40:46, 40:46].any())

    def test_smoothing_protects_prompts_at_edge_and_completed_area(self):
        self.coords = np.array([[5, 5], [4, 4]])
        self.labels = np.array([1, 0])
        self.valid[5:7, 60:65] = False
        self.mask &= self.valid
        excluded = [{"hull": [[70, 5], [74, 5], [74, 10], [70, 10]], "holes": []}]
        actual, _ = self.process({"smooth_boundary": True, "fill_holes": True, "max_hole_area_m2": 1000}, excluded)
        self.assertTrue(actual[5, 5])
        self.assertFalse(actual[4, 4])
        self.assertFalse(actual[~self.valid].any())
        self.assertFalse(actual[5:10, 70:74].any())

    def test_area_respects_rotated_grid_and_latitude(self):
        ring = [[0, 0], [2, 0], [2, 2], [0, 2]]
        affine = Affine(2, 0.2, 500000, 0.1, -2, 3300000)
        self.assertAlmostEqual(_ring_area_m2(ring, affine, "EPSG:32649"), 16.08, places=7)
        equator = _ring_area_m2(ring, Affine(0.00001, 0, 111, 0, -0.00001, 0), "EPSG:4326")
        north = _ring_area_m2(ring, Affine(0.00001, 0, 111, 0, -0.00001, 60), "EPSG:4326")
        self.assertTrue(0.45 < north / equator < 0.55)

    def test_bad_settings_and_missing_crs(self):
        for value in ([], {"fill_holes": 1}, {"smooth_boundary": "true"},
                      {"max_hole_area_m2": 0}, {"max_hole_area_m2": float("nan")},
                      {"max_hole_area_m2": True}, {"max_hole_area_m2": "10"}):
            with self.assertRaises(ValueError):
                validate_postprocess(value)
        with self.assertRaisesRegex(ValueError, "坐标系"):
            _ring_area_m2([[0, 0], [2, 0], [2, 2]], self.affine, None)


if __name__ == "__main__":
    unittest.main(verbosity=2)
