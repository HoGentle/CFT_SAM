"""验证细长凹槽修补、完整面积筛选、内部洞及提示约束。"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.sam_postprocess import process_regions, validate_postprocess, _rasterize_regions
from core.sam_roi import mask_to_regions
import numpy as np
from affine import Affine


class NotchTests(unittest.TestCase):
    def setUp(self):
        self.mask = np.zeros((100, 120), dtype=bool)
        self.mask[10:90, 10:110] = True
        self.mask[10:35, 48:51] = False
        self.mask[10:45, 75:100] = False
        self.mask[50:55, 60:65] = False
        self.valid = np.ones_like(self.mask)
        self.coords = np.array([[20, 60]], dtype=float)
        self.labels = np.array([1])
        self.transform = Affine(0.1, 0, 500000, 0, -0.1, 3300000)
        self.settings = {"fill_small_notches": True, "max_notch_width_m": 0.45, "max_notch_area_m2": 0.8}

    def process(self, options=None, excluded=None):
        regions = mask_to_regions(self.mask, self.coords, self.labels)
        processed = process_regions(regions, self.transform, "EPSG:32649", self.coords, self.labels,
                                    self.valid, excluded or [], validate_postprocess(options or self.settings))
        return _rasterize_regions(processed, self.mask.shape)

    def test_fills_narrow_notch_but_keeps_large_bay_and_hole(self):
        expected = self.mask.copy()
        expected[10:35, 48:51] = True
        np.testing.assert_array_equal(self.process(), expected)

    def test_disabled_width_and_area_limits(self):
        for change in ({"fill_small_notches": False}, {"max_notch_width_m": 0.15}, {"max_notch_area_m2": 0.7}):
            with self.subTest(change=change):
                np.testing.assert_array_equal(self.process({**self.settings, **change}), self.mask)
        self.assertTrue(self.process({**self.settings, "max_notch_area_m2": 0.75})[10:35, 48:51].all())

    def test_bottle_neck_and_chamber_are_screened_together(self):
        self.mask[10:35, 48:51] = True
        self.mask[10:20, 48:51] = False
        self.mask[20:35, 43:56] = False
        np.testing.assert_array_equal(self.process({**self.settings, "max_notch_area_m2": 1}), self.mask)
        actual = self.process({**self.settings, "max_notch_area_m2": 2.3})
        expected = self.mask.copy()
        expected[10:20, 48:51] = True
        expected[20:35, 43:56] = True
        np.testing.assert_array_equal(actual, expected)

    def test_negative_prompt_protects_entire_notch(self):
        self.coords = np.array([[20, 60], [49, 25]])
        self.labels = np.array([1, 0])
        np.testing.assert_array_equal(self.process(), self.mask)

    def test_notches_on_each_side_and_non_square_ground_pixels(self):
        original = self.mask.copy()
        expected = original.copy()
        expected[10:35, 48:51] = True
        marker = np.zeros_like(original)
        marker[60, 20] = True
        for turn in (1, 2, 3):
            with self.subTest(turn=turn):
                self.mask = np.rot90(original, turn).copy()
                self.valid = np.ones_like(self.mask)
                self.coords = np.argwhere(np.rot90(marker, turn))[:, ::-1]
                np.testing.assert_array_equal(self.process(), np.rot90(expected, turn))
        self.mask, self.valid, self.coords = original, np.ones_like(original), np.array([[20, 60]])
        self.transform = Affine(0.1, 0, 500000, 0, -0.2, 3300000)
        np.testing.assert_array_equal(self.process(), original)
        np.testing.assert_array_equal(self.process({**self.settings, "max_notch_area_m2": 1.6}), expected)

    def test_invalid_pixels_and_completed_regions_protect_entire_notch(self):
        self.valid[25, 49] = False
        np.testing.assert_array_equal(self.process(), self.mask)
        self.valid[:] = True
        excluded = [{"hull": [[48, 20], [51, 20], [51, 22], [48, 22]], "holes": []}]
        np.testing.assert_array_equal(self.process(excluded=excluded), self.mask)

    def test_different_regions_are_not_joined(self):
        self.mask[10:45, 75:100] = True
        self.mask[10:35, 48:51] = True
        self.mask[10:70, 47:52] = False
        self.mask[40:55, 48:51] = True
        self.coords = np.array([[20, 60], [49, 45]])
        self.labels = np.array([1, 1])
        np.testing.assert_array_equal(self.process({**self.settings, "max_notch_width_m": 0.65, "max_notch_area_m2": 5}), self.mask)

    def test_smoothing_and_hole_filling_can_be_combined(self):
        actual = self.process({**self.settings, "smooth_boundary": True, "fill_holes": True, "max_hole_area_m2": 0.3})
        self.assertTrue(actual[20:35, 48:51].all())
        self.assertTrue(actual[50:55, 60:65].all())
        self.assertFalse(actual[20:35, 80:95].any())

    def test_invalid_settings(self):
        for change in ({"fill_small_notches": 1}, {"max_notch_width_m": 0}, {"max_notch_width_m": 6},
                       {"max_notch_width_m": float("nan")}, {"max_notch_area_m2": 0},
                       {"max_notch_area_m2": True}, {"max_notch_area_m2": float("inf")}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_postprocess({**self.settings, **change})


if __name__ == "__main__":
    unittest.main()
