"""验证大田块的粗定位、原图边界细化、重叠提示传递与完整性检查。"""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import rasterio
from affine import Affine
from rasterio.features import rasterize
from shapely.geometry import GeometryCollection, LineString, Polygon, box, mapping

from core.sam_multiscale import _geometry, _polygonal
from core.sam_roi import mask_to_regions
from core.sam_window import predict_tiff_roi


class ImagePredictor:
    """从测试影像的颜色识别田块，粗阶段可提供带定位偏差的轮廓。"""

    def __init__(self, coarse_geometry=None, inconsistent=False):
        self.coarse_geometry = coarse_geometry
        self.inconsistent = inconsistent
        self.calls = []

    def predict(self, image, key, points, excluded, **kwargs):
        self.calls.append((image.shape, key, points, excluded, kwargs))
        if key[1] == "coarse" and self.coarse_geometry is not None:
            _, _, x, y, width, height, oh, ow = key
            mask = rasterize([(mapping(self.coarse_geometry), 1)], out_shape=(oh, ow),
                             transform=Affine.translation(x, y) * Affine.scale(width / ow, height / oh)) > 0
        else:
            mask = (image[:, :, 0] > 180) & (image[:, :, 1] < 100)
        mask &= kwargs["valid_mask"]
        coords = np.array([[p["x"], p["y"]] for p in points])
        labels = np.array([p["label"] for p in points])
        ix, iy = coords.astype(int).T
        if not np.array_equal(mask[iy, ix], labels.astype(bool)):
            raise ValueError("测试候选未能满足提示点")
        if self.inconsistent and key[1] == "fine" and len(self.calls) > 2:
            # 保留提示点，却故意把其余可靠重叠区删去，用来检验合并保护。
            mask[:] = False
            mask[iy[labels == 1], ix[labels == 1]] = True
        return {"regions": mask_to_regions(mask, coords, labels, excluded), "score": 0.9,
                "model_id": kwargs["model_id"], "model_label": "测试模型", "device": "cpu"}


class MultiscaleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.path = self.root / "field.tif"
        self.width, self.height = 900, 700
        self.transform = Affine(0.2, 0.03, 500000, 0.01, -0.2, 3300000)
        self.preview = {"preview_id": "large-field", "mode": "result_preview",
                        "source_paths": {"preview": str(self.path)}, "georef": {
                            "preview_width": 90, "preview_height": 70,
                            "source_width": self.width, "source_height": self.height,
                            "transform": list(self.transform[:6]), "crs": "EPSG:32649"}}
        self.points = [{"x": 20, "y": 20, "label": 1}]
        self.options = {"strategy": "coarse_to_fine", "window_size": 128,
                        "max_window_size": 256, "padding": 16, "coarse_long_side": 128,
                        "overlap": 0.7, "max_refinement_windows": 1000}

    def tearDown(self):
        self.temp.cleanup()

    def write(self, geometry, invalid=None):
        self.expected = rasterize([(mapping(geometry), 1)], out_shape=(self.height, self.width)) > 0
        rgb = np.full((3, self.height, self.width), 30, dtype=np.uint8)
        rgb[:, self.expected] = np.array([220, 60, 90])[:, None]
        if invalid is not None:
            rgb[:, invalid] = 0
            self.expected &= ~invalid
        with rasterio.open(self.path, "w", driver="GTiff", width=self.width, height=self.height,
                           count=3, dtype="uint8", transform=self.transform, crs="EPSG:32649", nodata=0) as dst:
            dst.write(rgb)

    def predict(self, model=None, points=None, excluded=None, postprocess=None):
        return predict_tiff_roi(model or ImagePredictor(), self.preview, points or self.points,
                                excluded or [], "sam1_vit_h", self.options, postprocess=postprocess)

    def mask(self, result):
        geometry = _geometry(result["regions"], scale=(10, 10))
        return rasterize([(mapping(geometry), 1)], out_shape=self.expected.shape) > 0

    def test_field_larger_than_native_limit_is_complete_and_native_refined(self):
        field = box(90, 80, 810, 610)
        self.write(field)
        model = ImagePredictor()
        result = self.predict(model)
        np.testing.assert_array_equal(self.mask(result), self.expected)
        self.assertEqual(result["inference_source"], "original_tiff_coarse_to_fine")
        self.assertGreater(result["multiscale"]["coarse_attempts"], 1)
        self.assertTrue(result["multiscale"]["boundary_complete"])
        self.assertEqual(result["multiscale"]["overlap"], 0.7)
        coarse_calls = [call for call in model.calls if call[1][1] == "coarse"]
        fine_calls = [call for call in model.calls if call[1][1] == "fine"]
        self.assertTrue(all(max(c[0][:2]) <= 128 for c in coarse_calls))
        self.assertTrue(all(c[0] == (128, 128, 3) for c in fine_calls))
        self.assertTrue(any(not any(p["x"] == 205 - c[1][2] and p["y"] == 205 - c[1][3]
                                    for p in c[2]) for c in fine_calls))
        self.assertTrue(all(any(p["label"] == 1 for p in c[2]) for c in fine_calls))

    def test_tangent_window_lines_are_removed_before_boundary_queries(self):
        collection = GeometryCollection([box(0, 0, 10, 10), LineString([(10, 15), (20, 15)])])
        polygon = _polygonal(collection)
        self.assertEqual(polygon.geom_type, "Polygon")
        self.assertIsNotNone(polygon.boundary)
        self.assertEqual(polygon.area, 100)

    def test_inaccurate_coarse_boundary_is_corrected_by_local_translation(self):
        field = box(90, 80, 810, 610)
        self.write(field)
        model = ImagePredictor(box(110, 100, 790, 590))
        result = self.predict(model)
        np.testing.assert_array_equal(self.mask(result), self.expected)

    def test_generated_prompt_conflicts_retry_without_dropping_user_prompts(self):
        class HintSensitivePredictor(ImagePredictor):
            def predict(self, image, key, points, excluded, **kwargs):
                if key[1] == "fine" and len(points) > 2:
                    raise ValueError("当前候选未能同时满足保留和排除点，请调整提示点")
                return super().predict(image, key, points, excluded, **kwargs)

        self.write(box(90, 80, 810, 610))
        points = self.points + [{"x": 2, "y": 2, "label": 0}]
        result = self.predict(HintSensitivePredictor(), points=points)
        np.testing.assert_array_equal(self.mask(result), self.expected)
        self.assertGreater(result["multiscale"]["prompt_retries"], 0)

    def test_concavity_holes_and_original_invalid_pixels_survive(self):
        field = Polygon([[90, 80], [810, 80], [810, 300], [600, 300], [600, 610], [90, 610]],
                        [[[330, 250], [470, 250], [470, 350], [330, 350]]])
        invalid = np.zeros((self.height, self.width), dtype=bool)
        invalid[180:183, 400:403] = True
        self.write(field, invalid)
        result = self.predict()
        np.testing.assert_array_equal(self.mask(result), self.expected)
        self.assertGreaterEqual(sum(len(r["holes"]) for r in result["regions"]), 2)

    def test_original_prompts_exclusions_and_scribbles_are_enforced(self):
        self.write(box(90, 80, 810, 610))
        excluded = [{"hull": [[36, 36], [42, 36], [42, 42], [36, 42]]}]
        points = self.points + [{"x": 2, "y": 2, "label": 0}, {"x": 70, "y": 50, "label": 1}]
        result = predict_tiff_roi(ImagePredictor(), self.preview, points, excluded, "sam1_vit_h", self.options,
                                 scribbles=[{"points": [[20, 20], [70, 50]]}])
        expected = self.expected.copy()
        expected[360:420, 360:420] = False
        np.testing.assert_array_equal(self.mask(result), expected)
        self.assertEqual(result["scribble_count"], 1)
        self.assertGreater(result["scribble_prompt_count"], 1)

    def test_window_budget_never_returns_partial_boundary(self):
        self.write(box(90, 80, 810, 610))
        self.options["max_refinement_windows"] = 2
        with self.assertRaisesRegex(ValueError, "数量上限"):
            self.predict()

    def test_inconsistent_overlap_is_rejected(self):
        self.write(box(90, 80, 810, 610))
        with self.assertRaisesRegex(ValueError, "不一致"):
            self.predict(ImagePredictor(inconsistent=True))

    def test_original_invalid_positive_is_rejected_before_coarse_model(self):
        invalid = np.zeros((self.height, self.width), dtype=bool)
        invalid[205, 205] = True
        self.write(box(90, 80, 810, 610), invalid)
        model = ImagePredictor()
        with self.assertRaisesRegex(ValueError, "无效像元"):
            self.predict(model)
        self.assertFalse(model.calls)

    def test_coarse_prompt_collision_is_rejected_with_actionable_error(self):
        self.write(box(90, 80, 810, 610))
        points = self.points + [{"x": 20.001, "y": 20.001, "label": 0}]
        with self.assertRaisesRegex(ValueError, "降采样后.*过近"):
            self.predict(points=points)

    def test_composite_multiscale_uses_original_band_order(self):
        self.write(box(90, 80, 810, 610))
        paths = {}
        with rasterio.open(self.path) as src:
            rgb = src.read()
        for index, role in enumerate(("B4", "B3", "B2")):
            path = self.root / f"{role}.tif"
            with rasterio.open(path, "w", driver="GTiff", width=self.width, height=self.height,
                               count=1, dtype="uint8", transform=self.transform, crs="EPSG:32649", nodata=0) as dst:
                dst.write(rgb[index], 1)
            paths[role] = str(path)
        self.preview.update(mode="composite", source_paths=paths)
        result = self.predict()
        np.testing.assert_array_equal(self.mask(result), self.expected)

    def test_api_preserves_geographic_coordinates_and_multiscale_metadata(self):
        import app as web
        from rasterio.warp import transform as project

        self.write(box(90, 80, 810, 610))
        with patch.object(web.SESSION, "input_preview", self.preview), \
                patch.dict(web.CONFIG, {"sam_roi": self.options}), \
                patch.object(web, "_sam_service", ImagePredictor()):
            response = web.app.test_client().post("/api/roi/sam", json={
                "preview_id": self.preview["preview_id"], "points": self.points})
        self.assertEqual(response.status_code, 200, response.json)
        self.assertTrue(response.json["multiscale"]["boundary_complete"])
        self.assertEqual(response.json["inference_source"], "original_tiff_coarse_to_fine")
        for point in response.json["regions"][0]["hull"]:
            x, y = self.transform * (point["x"] * 10, point["y"] * 10)
            lons, lats = project("EPSG:32649", "EPSG:4326", [x], [y])
            self.assertAlmostEqual(point["lon"], lons[0], places=10)
            self.assertAlmostEqual(point["lat"], lats[0], places=10)

    def test_true_image_edges_are_allowed(self):
        self.write(box(0, 0, self.width, self.height))
        result = self.predict()
        np.testing.assert_array_equal(self.mask(result), self.expected)

    def test_hole_filling_uses_full_original_area_and_protects_invalid_holes(self):
        field = Polygon([[90, 80], [810, 80], [810, 610], [90, 610]],
                        [[[330, 250], [350, 250], [350, 270], [330, 270]]])
        invalid = np.zeros((self.height, self.width), dtype=bool)
        invalid[180:183, 400:403] = True
        self.write(field, invalid)
        result = self.predict(postprocess={"fill_holes": True, "max_hole_area_m2": 20})
        expected = self.expected.copy()
        expected[250:270, 330:350] = True
        np.testing.assert_array_equal(self.mask(result), expected)
        result = self.predict(postprocess={"fill_holes": True, "max_hole_area_m2": 10})
        np.testing.assert_array_equal(self.mask(result), self.expected)

    def test_tiled_smoothing_preserves_prompts_exclusions_and_invalid_pixels(self):
        field = Polygon([[90, 80], [810, 80], [810, 610], [90, 610]],
                        [[[330, 250], [350, 250], [350, 270], [330, 270]]])
        invalid = np.zeros((self.height, self.width), dtype=bool)
        invalid[100:103, 100:103] = True
        invalid[79:81, 89:91] = True
        self.write(field, invalid)
        excluded = [{"hull": [[36, 36], [42, 36], [42, 42], [36, 42]]}]
        points = self.points + [{"x": 33, "y": 25, "label": 0}]
        result = self.predict(points=points, excluded=excluded,
                              postprocess={"smooth_boundary": True, "fill_holes": True,
                                           "max_hole_area_m2": 20})
        mask = self.mask(result)
        self.assertFalse(mask[invalid].any())
        self.assertFalse(mask[360:420, 360:420].any())
        self.assertFalse(mask[255, 335])
        self.assertTrue(mask[205, 205])


if __name__ == "__main__":
    unittest.main(verbosity=2)
