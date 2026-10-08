"""验证原图局部读取、扩窗及提示／轮廓坐标转换，无需加载权重。"""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import rasterio
from affine import Affine
from core.preview import generate_input_preview
from core.sam_window import predict_tiff_roi


class WindowPredictor:
    def __init__(self, box, holes=None):
        self.box = box
        self.holes = holes or []
        self.calls = []

    def predict(self, image, key, points, excluded, **kwargs):
        self.calls.append((image.copy(), key, points, excluded, kwargs))
        _, ox, oy, width, height = key
        x0, y0, x1, y1 = self.box
        x0, x1 = max(0, x0 - ox), min(width, x1 - ox)
        y0, y1 = max(0, y0 - oy), min(height, y1 - oy)
        holes = [(np.asarray(hole) - [ox, oy]).tolist() for hole in self.holes]
        return {"regions": [{"hull": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]], "holes": holes}],
                "model_id": kwargs["model_id"], "model_label": "测试模型"}


class WindowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.path = self.root / "original.tif"
        self.transform = Affine(0.2, 0.03, 500000, 0.01, -0.2, 3300000)
        y, x = np.indices((600, 800))
        self.rgb = np.stack([x % 251, y % 249, (x + y) % 247]).astype(np.uint8)
        self.write(self.path, self.rgb)
        self.preview = {"preview_id": "source-1", "mode": "result_preview",
                        "source_paths": {"preview": str(self.path)}, "georef": {
                            "preview_width": 80, "preview_height": 60,
                            "source_width": 800, "source_height": 600,
                            "transform": list(self.transform[:6]), "crs": "EPSG:32649"}}
        self.options = {"window_size": 128, "padding": 16, "max_window_size": 512}
        self.points = [{"x": 40, "y": 30, "label": 1}]

    def tearDown(self):
        self.temp.cleanup()

    def write(self, path, array, nodata=None):
        with rasterio.open(path, "w", driver="GTiff", width=array.shape[2], height=array.shape[1],
                           count=array.shape[0], dtype=array.dtype, transform=self.transform,
                           crs="EPSG:32649", nodata=nodata) as dst:
            dst.write(array)

    def predict(self, model, points=None, excluded=None):
        return predict_tiff_roi(model, self.preview, points or self.points, excluded or [],
                                "hq_sam_vit_l", self.options)

    def test_native_pixels_prompts_exclusions_and_holes(self):
        hole = [[410, 320], [415, 320], [415, 330], [410, 330]]
        model = WindowPredictor((380, 280, 440, 350), [hole])
        excluded = [{"hull": [[38, 28], [39, 28], [39, 29], [38, 29]],
                     "holes": [[[38.1, 28.1], [38.2, 28.1], [38.2, 28.2]]]}]
        result = self.predict(model, self.points + [{"x": 43, "y": 29, "label": 0}], excluded)
        image, key, prompts, blocked, kwargs = model.calls[0]
        _, ox, oy, width, height = key
        np.testing.assert_array_equal(image, self.rgb[:, oy:oy + height, ox:ox + width].transpose(1, 2, 0))
        self.assertEqual(prompts[0], {"x": 405 - ox, "y": 305 - oy, "label": 1})
        self.assertEqual(prompts[1], {"x": 435 - ox, "y": 295 - oy, "label": 0})
        np.testing.assert_allclose(blocked[0]["hull"], np.asarray(excluded[0]["hull"]) * 10 - [ox, oy])
        np.testing.assert_allclose(blocked[0]["holes"][0], np.asarray(excluded[0]["holes"][0]) * 10 - [ox, oy])
        self.assertTrue(kwargs["valid_mask"].all())
        self.assertEqual(kwargs["model_id"], "hq_sam_vit_l")
        np.testing.assert_allclose(result["regions"][0]["hull"], [[38, 28], [44, 28], [44, 35], [38, 35]])
        np.testing.assert_allclose(result["regions"][0]["holes"][0], np.asarray(hole) / 10)
        self.assertEqual(result["inference_source"], "original_tiff_window")
        self.assertFalse((self.root / "input_preview.jpg").exists())

    def test_expands_until_outline_is_complete(self):
        model = WindowPredictor((280, 180, 540, 430))
        result = self.predict(model)
        self.assertEqual(len(model.calls), 3)
        self.assertEqual(result["inference_window"]["width"], 512)
        self.assertEqual(result["inference_window"]["attempts"], 3)
        np.testing.assert_allclose(result["regions"][0]["hull"], [[28, 18], [54, 18], [54, 43], [28, 43]])
        self.assertNotEqual(model.calls[0][1], model.calls[1][1])

    def test_rejects_outline_clipped_at_limit(self):
        model = WindowPredictor((0, 0, 800, 600))
        with self.assertRaisesRegex(ValueError, "仍触及"):
            self.predict(model)
        self.assertEqual(len(model.calls), 3)

    def test_true_image_boundary_does_not_expand(self):
        model = WindowPredictor((0, 0, 80, 80))
        result = self.predict(model, [{"x": 1, "y": 1, "label": 1}])
        self.assertEqual(len(model.calls), 1)
        self.assertEqual(result["regions"][0]["hull"][0], [0, 0])

    def test_composite_uses_original_band_order_and_masks(self):
        paths = {}
        for role, value in (("B4", 11), ("B3", 22), ("B2", 33)):
            paths[role] = str(self.root / f"{role}.tif")
            band = np.full((1, 600, 800), value, dtype=np.uint8)
            band[:, 300:302, 400:402] = 0
            self.write(paths[role], band, nodata=0)
        self.preview.update(mode="composite", source_paths=paths)
        model = WindowPredictor((380, 280, 440, 350))
        self.predict(model)
        image, key, _, _, kwargs = model.calls[0]
        np.testing.assert_array_equal(image[0, 0], [11, 22, 33])
        _, ox, oy, _, _ = key
        self.assertFalse(kwargs["valid_mask"][300 - oy, 400 - ox])
        np.testing.assert_array_equal(image[300 - oy, 400 - ox], [255, 255, 255])

    def test_invalid_positive_and_missing_source_do_not_predict(self):
        self.write(self.path, np.zeros((3, 600, 800), dtype=np.uint8), nodata=0)
        model = WindowPredictor((380, 280, 440, 350))
        with self.assertRaisesRegex(ValueError, "无效像元"):
            self.predict(model)
        self.assertFalse(model.calls)
        self.preview.pop("source_paths")
        with self.assertRaisesRegex(ValueError, "原始 TIFF 路径"):
            self.predict(model)

    def test_source_grid_change_and_scattered_prompts_are_rejected(self):
        model = WindowPredictor((380, 280, 440, 350))
        with self.assertRaisesRegex(ValueError, "覆盖范围过大"):
            self.predict(model, self.points + [{"x": 1, "y": 1, "label": 0},
                                              {"x": 79, "y": 59, "label": 0}])
        self.preview["georef"]["source_width"] = 801
        with self.assertRaisesRegex(ValueError, "网格与预览不一致"):
            self.predict(model)
        self.assertFalse(model.calls)

    def test_preview_records_exact_source_for_both_modes(self):
        jpg = self.root / "preview.jpg"
        info = generate_input_preview(self.path, {}, jpg, max_long_side=80)
        self.assertEqual(info["source_paths"], {"preview": str(self.path.resolve())})
        bands = {role: self.path for role in ("B4", "B3", "B2")}
        info = generate_input_preview(None, bands, jpg, max_long_side=80)
        self.assertEqual(info["mode"], "composite")
        self.assertEqual(set(info["source_paths"]), set(bands))

    def test_alpha_and_last_pixel_prompt(self):
        from rasterio.enums import ColorInterp
        rgba = np.concatenate([self.rgb, np.full((1, 600, 800), 255, dtype=np.uint8)])
        rgba[3, 590:592, 790:792] = 0
        self.write(self.path, rgba)
        with rasterio.open(self.path, "r+") as dst:
            dst.colorinterp = (ColorInterp.red, ColorInterp.green, ColorInterp.blue, ColorInterp.alpha)
        model = WindowPredictor((750, 550, 800, 600))
        result = self.predict(model, [{"x": 79.9, "y": 59.9, "label": 1}])
        self.assertEqual(len(model.calls), 1)
        _, key, points, _, kwargs = model.calls[0]
        _, ox, oy, width, height = key
        self.assertLess(points[0]["x"], width)
        self.assertLess(points[0]["y"], height)
        self.assertFalse(kwargs["valid_mask"][590 - oy, 790 - ox])
        self.assertEqual(result["regions"][0]["hull"][2], [80, 60])

    def test_window_postprocess_uses_original_pixel_ground_area(self):
        hole = [[410, 320], [415, 320], [415, 330], [410, 330]]
        model = WindowPredictor((380, 280, 440, 350), [hole])
        result = predict_tiff_roi(model, self.preview, self.points, [], "sam3", self.options,
                                 {"fill_holes": True, "max_hole_area_m2": 3})
        # 原图洞为 50 像元，旋转网格每像元约 0.0403 平方米。
        self.assertEqual(result["regions"][0]["holes"], [])
        self.assertTrue(result["postprocess"]["fill_holes"])
        model = WindowPredictor((380, 280, 440, 350), [hole])
        result = predict_tiff_roi(model, self.preview, self.points, [], "sam3", self.options,
                                 {"fill_holes": True, "max_hole_area_m2": 1})
        self.assertEqual(len(result["regions"][0]["holes"]), 1)

    def test_scribbles_are_model_prompts_and_never_painted(self):
        model = WindowPredictor((380, 280, 440, 350))
        points = self.points + [{"x": 43, "y": 29, "label": 0}]
        # 故意让线超出模型返回轮廓，确认没有按线强行填补模型掩膜。
        result = predict_tiff_roi(model, self.preview, points, [], "sam3", self.options,
                                 scribbles=[{"points": [[39, 30], [46, 30]]}])
        _, key, prompts, _, kwargs = model.calls[0]
        _, ox, oy, _, _ = key
        self.assertEqual(prompts[:2], [{"x": 405 - ox, "y": 305 - oy, "label": 1},
                                      {"x": 435 - ox, "y": 295 - oy, "label": 0}])
        self.assertEqual(prompts[2], {"x": 390 - ox, "y": 300 - oy, "label": 1})
        self.assertEqual(prompts[-1], {"x": 460 - ox, "y": 300 - oy, "label": 1})
        self.assertEqual(result["scribble_count"], 1)
        self.assertEqual(result["scribble_prompt_count"], 4)
        self.assertEqual(kwargs["model_id"], "sam3")
        np.testing.assert_allclose(result["regions"][0]["hull"], [[38, 28], [44, 28], [44, 35], [38, 35]])

    def test_line_conflicts_and_invalid_pixels_do_not_predict(self):
        model = WindowPredictor((380, 280, 440, 350))
        with self.assertRaisesRegex(ValueError, "冲突"):
            predict_tiff_roi(model, self.preview, self.points + [{"x": 43, "y": 29, "label": 0}],
                             [], "sam3", self.options,
                             scribbles=[{"points": [[40, 30], [43.5, 29.5]]}])
        excluded = [{"hull": [[42, 28], [45, 28], [45, 33], [42, 33]]}]
        with self.assertRaisesRegex(ValueError, "已确认区域"):
            predict_tiff_roi(model, self.preview, self.points, excluded, "sam3", self.options,
                             scribbles=[{"points": [[40, 30], [44, 30]]}])
        data = self.rgb.copy()
        data[:, 300, 440] = 0
        self.write(self.path, data, nodata=0)
        with self.assertRaisesRegex(ValueError, "无效像元"):
            predict_tiff_roi(model, self.preview, self.points, [], "sam3", self.options,
                             scribbles=[{"points": [[40, 30], [44, 30]]}])
        self.assertEqual(model.calls, [])

    def test_window_includes_curve_vertices_even_with_two_sample_budget(self):
        model = WindowPredictor((380, 280, 440, 350))
        result = predict_tiff_roi(model, self.preview, self.points * 126, [], "sam3", self.options,
                                 scribbles=[{"points": [[38, 30], [64, 30], [38, 31]]}])
        window = result["inference_window"]
        self.assertLessEqual(window["x"], 380)
        self.assertGreater(window["x"] + window["width"], 640)
        self.assertEqual(result["scribble_prompt_count"], 2)
        self.assertEqual(len(model.calls[0][2]), 128)

    def test_gap_width_is_converted_on_original_window_grid(self):
        model = WindowPredictor((380, 280, 440, 350))
        result = predict_tiff_roi(model, self.preview, self.points, [], "sam3", self.options,
                                 {"ignore_small_boundaries": True, "max_gap_width_m": 0.4})
        self.assertEqual(model.calls[0][4]["gap_kernel"], (2, 2))
        self.assertTrue(result["postprocess"]["ignore_small_boundaries"])
        self.assertEqual(result["postprocess"]["max_gap_width_m"], 0.4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
