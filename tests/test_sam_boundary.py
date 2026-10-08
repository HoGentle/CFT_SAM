"""验证按宽度合并细缝、原图米制换算与连通区域筛选顺序。"""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from affine import Affine
from core.sam_boundary import close_small_gaps, gap_kernel_for_width
from core.sam_postprocess import validate_postprocess, _rasterize_regions
from core.sam_roi import SamRoiService, mask_to_regions


class BoundaryTests(unittest.TestCase):
    def setUp(self):
        self.coords = np.array([[10, 20]])
        self.labels = np.array([1])
        self.mask = np.zeros((40, 60), dtype=bool)
        self.mask[5:35, 5:55] = True

    def close(self, mask, kernel=(3, 3), **kwargs):
        return close_small_gaps(mask, kernel, self.coords, self.labels, **kwargs)

    def test_thin_gap_reconnects_but_wider_gap_remains(self):
        mask = self.mask.copy()
        mask[5:35, 25:27] = False
        mask[5:35, 40:44] = False
        result = self.close(mask)
        self.assertTrue(result[20, 25:27].all())
        self.assertFalse(result[20, 40:44].any())
        self.assertTrue(result[mask].all())
        regions = mask_to_regions(result, self.coords, self.labels)
        raster = _rasterize_regions(regions, mask.shape)
        self.assertTrue(raster[20, 35])
        self.assertFalse(raster[20, 50])

    def test_even_kernel_does_not_translate_outline(self):
        mask = self.mask.copy()
        mask[5:35, 25] = False
        result = self.close(mask, (2, 2))
        np.testing.assert_array_equal(result, self.mask)
        result = self.close(self.mask, (2, 4))
        np.testing.assert_array_equal(result, self.mask)

    def test_diagonal_wire_can_be_closed(self):
        mask = self.mask.copy()
        y, x = np.indices(mask.shape)
        mask[x == y + 10] = False
        result = self.close(mask)
        self.assertTrue(result[20, 30])
        regions = mask_to_regions(result, self.coords, self.labels)
        self.assertTrue(_rasterize_regions(regions, mask.shape)[20, 45])

    def test_negative_points_invalid_pixels_and_exclusions_survive(self):
        mask = self.mask.copy()
        mask[5:35, 25:27] = False
        self.coords = np.array([[10, 20], [25, 20]])
        self.labels = np.array([1, 0])
        result = self.close(mask)
        self.assertFalse(result[20, 25])
        self.assertTrue(result[21, 25])
        valid = np.ones_like(mask)
        valid[:, 25:27] = False
        result = self.close(mask, valid=valid)
        self.assertFalse(result[~valid].any())
        excluded = [{"hull": [[25, 0], [27, 0], [27, 40], [25, 40]], "holes": []}]
        result = self.close(mask, excluded=excluded)
        self.assertFalse(result[:, 25:27].any())

    def test_width_uses_native_pixel_sizes_including_anisotropic_grid(self):
        affine = Affine(0.1, 0, 500000, 0, -0.2, 3300000)
        self.assertEqual(gap_kernel_for_width(0.4, affine, "EPSG:32649", (40, 60)), (5, 3))
        self.assertEqual(gap_kernel_for_width(0.05, affine, "EPSG:32649", (40, 60)), (1, 1))
        np.testing.assert_array_equal(self.close(self.mask, (1, 1)), self.mask)
        with self.assertRaisesRegex(ValueError, "坐标系"):
            gap_kernel_for_width(0.4, affine, None, (40, 60))

    def test_gap_is_closed_before_component_and_positive_selection(self):
        mask = self.mask.copy()
        mask[5:35, 25] = False
        predictor = SimpleNamespace(set_image=lambda image: None,
                                    predict=lambda **kwargs: (mask[None].copy(), np.array([0.9]), None))
        service = SamRoiService("unused")
        service._predictor = predictor
        service._loaded_model_id = "sam1_vit_h"
        service.device = "cpu"
        image = np.zeros((40, 60, 3), dtype=np.uint8)
        points = [{"x": 10, "y": 20, "label": 1}]
        raw = service.predict(image, "test", points)
        self.assertFalse(_rasterize_regions(raw["regions"], mask.shape)[20, 40])
        filtered = service.predict(image, "test", points, gap_kernel=(2, 2))
        self.assertTrue(_rasterize_regions(filtered["regions"], mask.shape)[20, 40])
        # 保留点恰落在电线缝隙上时，应在闭缝后校验它是否被覆盖。
        filtered = service.predict(image, "test", [{"x": 25, "y": 20, "label": 1}], gap_kernel=(2, 2))
        self.assertTrue(_rasterize_regions(filtered["regions"], mask.shape)[20, 25])

    def test_negative_constraints_cannot_be_bypassed_when_selecting_candidate(self):
        masks = np.stack([self.mask.copy(), self.mask.copy()])
        masks[1, 5:35, 25] = False
        predictor = SimpleNamespace(set_image=lambda image: None,
                                    predict=lambda **kwargs: (masks.copy(), np.array([0.99, 0.7]), None))
        service = SamRoiService("unused")
        service._predictor = predictor
        service._loaded_model_id = "sam1_vit_h"
        service.device = "cpu"
        result = service.predict(np.zeros((40, 60, 3), dtype=np.uint8), "test",
                                 [{"x": 10, "y": 20, "label": 1}, {"x": 25, "y": 20, "label": 0}],
                                 gap_kernel=(3, 3))
        self.assertAlmostEqual(result["score"], 0.7)
        self.assertFalse(_rasterize_regions(result["regions"], self.mask.shape)[20, 25])

    def test_boundary_settings_validation(self):
        for setting in ({"ignore_small_boundaries": 1}, {"max_gap_width_m": 0},
                        {"max_gap_width_m": 6}, {"max_gap_width_m": float("inf")},
                        {"max_gap_width_m": "0.2"}, {"max_gap_width_m": True}):
            with self.assertRaises(ValueError):
                validate_postprocess(setting)
        self.assertFalse(validate_postprocess(None)["ignore_small_boundaries"])
        self.assertEqual(validate_postprocess(None)["max_gap_width_m"], 0.2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
