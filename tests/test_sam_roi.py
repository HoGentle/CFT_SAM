"""点提示区域的几何、坐标和接口回归测试；无需加载大型权重。"""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.sam_roi import SamRoiService, mask_to_regions, validate_prompts
from core.roi import polygons_lonlat_to_pixel, rasterize_polygons_in_window
from core.stats import lonlat_polygons_area

import numpy as np
from affine import Affine
from rasterio.windows import Window


class GeometryTests(unittest.TestCase):
    def test_prompt_validation(self):
        for points in ([], [{"x": 1, "y": 1, "label": 0}],
                       [{"x": float("nan"), "y": 1, "label": 1}],
                       [{"x": 10, "y": 1, "label": 1}],
                       [{"x": 1, "y": 1, "label": 2}]):
            with self.assertRaises(ValueError):
                validate_prompts(points, 10, 10)

    def test_concavity_holes_and_components(self):
        mask = np.zeros((16, 20), dtype=bool)
        mask[1:12, 1:12] = True
        mask[1:5, 8:12] = False
        mask[6:8, 4:6] = False
        mask[3:6, 15:18] = True
        coords, labels = validate_prompts([{"x": 2, "y": 2, "label": 1}], 20, 16)
        regions = mask_to_regions(mask, coords, labels)
        self.assertEqual(len(regions), 1)
        self.assertEqual(len(regions[0]["holes"]), 1)
        result = rasterize_polygons_in_window(
            [{"hull": np.array(r["hull"]), "holes": [np.array(h) for h in r["holes"]]} for r in regions],
            Window(0, 0, 20, 16),
        )
        expected = mask.copy()
        expected[3:6, 15:18] = False
        np.testing.assert_array_equal(result, expected)
        np.testing.assert_array_equal(
            rasterize_polygons_in_window([{"hull": np.array(regions[0]["hull"]),
                                          "holes": [np.array(h) for h in regions[0]["holes"]]}],
                                         Window(3, 4, 6, 5)), expected[4:9, 3:9])

    def test_exclusion_with_hole(self):
        mask = np.ones((12, 12), dtype=bool)
        excluded = [{"hull": [[0, 0], [12, 0], [12, 12], [0, 12]],
                     "holes": [[[3, 3], [8, 3], [8, 8], [3, 8]]]}]
        regions = mask_to_regions(mask, np.array([[4, 4]]), np.array([1]), excluded)
        result = rasterize_polygons_in_window([{"hull": np.array(r["hull"]), "holes": []} for r in regions],
                                             Window(0, 0, 12, 12))
        expected = np.zeros_like(mask)
        expected[3:8, 3:8] = True
        np.testing.assert_array_equal(result, expected)

    def test_geographic_conversion_preserves_sam_outline(self):
        outline = [[0, 0], [4, 0], [4, 1], [1, 1], [1, 4], [0, 4]]
        sam = {"source": "sam", "hull": outline, "holes": []}
        poly = polygons_lonlat_to_pixel([sam], Affine.identity(), "EPSG:4326")[0]
        np.testing.assert_array_equal(poly["hull"], outline)
        manual = polygons_lonlat_to_pixel([outline], Affine.identity())[0]
        self.assertEqual(len(manual), 5)

    def test_projected_grid(self):
        from rasterio.warp import transform
        lonlat = [[113.5, 30.5], [113.501, 30.5], [113.501, 30.501], [113.5, 30.501]]
        xs, ys = transform("EPSG:4326", "EPSG:32649", *zip(*lonlat))
        affine = Affine(1, 0, xs[0], 0, -1, ys[0])
        result = polygons_lonlat_to_pixel([{"source": "sam", "hull": lonlat}], affine, "EPSG:32649")[0]
        np.testing.assert_allclose(result["hull"], np.column_stack([np.array(xs) - xs[0], ys[0] - np.array(ys)]))

    def test_area_subtracts_holes(self):
        outer = [[113.5, 30.5], [113.502, 30.5], [113.502, 30.502], [113.5, 30.502]]
        hole = [[113.5005, 30.5005], [113.5015, 30.5005], [113.5015, 30.5015], [113.5005, 30.5015]]
        area = lonlat_polygons_area([{"hull": outer, "holes": [hole]}])["area_m2"]
        expected = lonlat_polygons_area([outer])["area_m2"] - lonlat_polygons_area([hole])["area_m2"]
        self.assertAlmostEqual(area, expected, places=1)

    def test_candidate_selection_and_embedding_cache(self):
        class Predictor:
            calls = 0

            def set_image(self, image):
                self.calls += 1

            def predict(self, **kwargs):
                masks = np.zeros((3, 10, 10), dtype=bool)
                masks[0] = True
                masks[1, 1:5, 1:5] = True
                masks[2, 1:4, 1:4] = True
                return masks, np.array([0.99, 0.8, 0.7]), None

        service = SamRoiService("unused")
        service._predictor = Predictor()
        service._loaded_model_id = "sam1_vit_h"
        service.device = "cpu"
        points = [{"x": 2, "y": 2, "label": 1}, {"x": 8, "y": 8, "label": 0}]
        image = np.zeros((10, 10, 3), dtype=np.uint8)
        result = service.predict(image, "first", points)
        self.assertEqual(result["score"], 0.8)
        service.predict(image, "first", points)
        self.assertEqual(service._predictor.calls, 1)
        service.predict(image, "second", points)
        self.assertEqual(service._predictor.calls, 2)

    def test_catalog_and_invalid_model(self):
        with tempfile.TemporaryDirectory() as temp:
            Path(temp, "sam_hq_vit_l.pth").touch()
            Path(temp, "unknown.pth").touch()
            service = SamRoiService(temp)
            models = service.list_models()
            self.assertEqual([m["id"] for m in models], ["hq_sam_vit_l"])
            with self.assertRaises(ValueError):
                service.predict(np.zeros((10, 10, 3), dtype=np.uint8), "image",
                                [{"x": 2, "y": 2, "label": 1}], model_id="../checkpoint")

    def test_invalid_pixels_are_excluded_from_model_mask(self):
        from types import SimpleNamespace
        service = SamRoiService("unused")
        service._loaded_model_id = "sam1_vit_h"
        service.device = "cpu"
        service._predictor = SimpleNamespace(
            set_image=lambda image: None,
            predict=lambda **kwargs: (np.ones((1, 10, 10), dtype=bool), np.array([0.9]), None))
        valid = np.ones((10, 10), dtype=bool)
        valid[5:7, 5:7] = False
        result = service.predict(np.zeros((10, 10, 3), dtype=np.uint8), "masked",
                                 [{"x": 2, "y": 2, "label": 1}], valid_mask=valid)
        self.assertEqual(len(result["regions"][0]["holes"]), 1)
        regions = [{"hull": np.array(r["hull"]), "holes": [np.array(h) for h in r["holes"]]}
                   for r in result["regions"]]
        np.testing.assert_array_equal(rasterize_polygons_in_window(regions, Window(0, 0, 10, 10)), valid)

    def test_sam3_checkpoint_mapping_is_strict(self):
        import torch
        from core.sam3_roi import load_point_weights

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.zeros(1))

        with tempfile.TemporaryDirectory() as temp:
            checkpoint = Path(temp) / "weights.pt"
            for family, prefix in (("sam3", "tracker."), ("sam3_1", "tracker.model.")):
                torch.save({prefix + "weight": torch.ones(1)}, checkpoint)
                model = Model()
                load_point_weights(model, checkpoint, family)
                self.assertEqual(model.weight.item(), 1)
            torch.save({"unrelated.weight": torch.ones(1)}, checkpoint)
            with self.assertRaises(RuntimeError):
                load_point_weights(Model(), checkpoint, "sam3_1")

    def test_sam3_builder_restores_precision_context(self):
        import torch
        from types import SimpleNamespace
        from core.sam3_roi import build_point_predictor

        class Model:
            def __init__(self):
                self.bf16_context = torch.autocast("cpu", dtype=torch.bfloat16)
                self.bf16_context.__enter__()

            def to(self, **kwargs):
                return self

            def eval(self):
                return self

        prior = torch.is_autocast_enabled("cpu")
        modules = {
            "sam3.model_builder": SimpleNamespace(build_tracker=lambda **kwargs: Model(),
                                                   build_sam3_multiplex_video_model=None),
            "sam3.model.sam1_task_predictor": SimpleNamespace(SAM3InteractiveImagePredictor=lambda model, **kwargs: model),
        }
        with patch.dict(sys.modules, modules), patch("core.sam3_roi.load_point_weights"):
            build_point_predictor("unused", "sam3", "cpu")
        self.assertEqual(torch.is_autocast_enabled("cpu"), prior)

    def test_diagnosis_uses_outline_across_blocks(self):
        import json
        import rasterio
        from core.diagnose import run_diagnosis
        config = json.loads((Path(__file__).resolve().parents[1] / "config.json").read_text(encoding="utf-8"))
        mask = np.zeros((80, 80), dtype=bool)
        mask[2:78, 2:78] = True
        mask[2:20, 50:78] = False
        mask[60:70, 60:70] = False
        regions = mask_to_regions(mask, np.array([[30, 30]]), np.array([1]))
        polygons = [{"hull": np.array(r["hull"]), "holes": [np.array(h) for h in r["holes"]]} for r in regions]
        with tempfile.TemporaryDirectory() as temp:
            paths = {}
            for role, value in (("B1", 0.3), ("B2", 0.2), ("B3", 0.1), ("B4", 0.5)):
                paths[role] = Path(temp) / f"{role}.tif"
                with rasterio.open(paths[role], "w", driver="GTiff", width=80, height=80, count=1,
                                   dtype="float32", crs="EPSG:4326",
                                   transform=Affine(0.00001, 0, 113.5, 0, -0.00001, 30.5)) as dst:
                    dst.write(np.full((80, 80), value, dtype=np.float32), 1)
            result = run_diagnosis(paths, config["stages"]["1"], Path(temp) / "output",
                                   {**config["defaults"], "processing_block_size": 64}, roi_polygons=polygons)
            self.assertEqual(result["metadata"]["valid_pixel_count"], int(mask.sum()))
            with rasterio.open(result["class_raster"]) as src:
                np.testing.assert_array_equal(src.read(1) != 255, mask)
            with rasterio.open(result["value_raster"]) as src:
                np.testing.assert_array_equal(np.isfinite(src.read(1)), mask)


class ApiTests(unittest.TestCase):
    def setUp(self):
        import app as web
        self.web = web
        self.client = web.app.test_client()
        self.previous = web.SESSION.input_preview
        self.preview = {"preview_id": "test", "georef": {
            "preview_width": 10, "preview_height": 10, "source_width": 100, "source_height": 100,
            "transform": [0.0001, 0, 113.5, 0, -0.0001, 30.5], "crs": "EPSG:4326"},
            "mode": "result_preview"}
        web.SESSION.input_preview = self.preview

    def tearDown(self):
        self.web.SESSION.input_preview = self.previous

    def test_stale_preview_and_invalid_prompts(self):
        resp = self.client.post("/api/roi/sam", json={"preview_id": "stale", "points": []})
        self.assertEqual(resp.status_code, 409)
        resp = self.client.post("/api/roi/sam", json={"preview_id": "test", "points": []})
        self.assertEqual(resp.status_code, 400)

    def test_prediction_coordinates_and_metadata(self):
        import rasterio
        prediction = {"regions": [{"hull": [[10, 10], [40, 10], [40, 40], [10, 40]], "holes": []}],
                      "score": 0.9, "device": "cpu"}
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "source.tif"
            self.preview["source_paths"] = {"preview": str(path)}
            with rasterio.open(path, "w", driver="GTiff", width=100, height=100, count=3,
                               dtype="uint8", crs="EPSG:4326",
                               transform=Affine(*self.preview["georef"]["transform"])) as dst:
                dst.write(np.full((3, 100, 100), 73, dtype=np.uint8))
            # 不创建 JPG：接口必须直接从 TIFF 读取，不依赖预览文件。
            with patch.object(self.web, "PREVIEW_DIR", Path(temp)), \
                    patch.object(self.web._sam_service, "predict", return_value=prediction) as predict:
                resp = self.client.post("/api/roi/sam", json={"preview_id": "test",
                                        "model_id": "hq_sam_vit_l",
                                        "points": [{"x": 2, "y": 2, "label": 1}]})
                self.assertEqual(predict.call_args.kwargs["model_id"], "hq_sam_vit_l")
                self.assertEqual(predict.call_args.args[0].shape, (100, 100, 3))
                self.assertEqual(predict.call_args.args[2][0]["x"], 25)
                self.assertTrue((predict.call_args.args[0] == 73).all())
        self.assertEqual(resp.status_code, 200, resp.json)
        self.assertAlmostEqual(resp.json["regions"][0]["hull"][0]["lon"], 113.501)
        self.assertAlmostEqual(resp.json["points"][0]["lon"], 113.5025)
        self.assertEqual(resp.json["points"][0]["label"], 1)
        self.assertEqual(resp.json["preview_id"], "test")
        self.assertEqual(resp.json["inference_source"], "original_tiff_window")

    def test_postprocess_request_validation_and_metadata(self):
        from core.sam_roi import SamRoiService
        settings = {"fill_holes": True, "max_hole_area_m2": 5, "smooth_boundary": True}
        with patch.object(self.web, "predict_tiff_roi", return_value={"regions": [], "postprocess": settings}) as predict:
            resp = self.client.post("/api/roi/sam", json={"preview_id": "test", "postprocess": settings,
                                    "points": [{"x": 2, "y": 2, "label": 1}]})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(predict.call_args.kwargs["postprocess"], settings)
        self.assertEqual(resp.json["postprocess"], settings)
        # 非法选项应在读取 TIFF 或调用大型模型前拒绝。
        with patch.object(SamRoiService, "predict") as model:
            resp = self.client.post("/api/roi/sam", json={"preview_id": "test",
                                    "points": [{"x": 2, "y": 2, "label": 1}],
                                    "postprocess": {"fill_holes": True, "max_hole_area_m2": -1}})
        self.assertEqual(resp.status_code, 400)
        model.assert_not_called()
        cleaned = self.web.clean_roi_region({"source": "sam", "hull": [[113, 30], [113.1, 30], [113, 30.1]],
                                            "points": [], "point_labels": [], "postprocess": settings})
        from core.sam_postprocess import validate_postprocess
        self.assertEqual(cleaned["postprocess"], validate_postprocess(settings))

    def test_area_accepts_legacy_and_sam_regions(self):
        hull = [[113.5, 30.5], [113.501, 30.5], [113.501, 30.501], [113.5, 30.501]]
        hole = [[113.5002, 30.5002], [113.5008, 30.5002], [113.5008, 30.5008], [113.5002, 30.5008]]
        old = self.client.post("/api/roi/area", json={"regions": [{"hull": hull}]}).json
        new = self.client.post("/api/roi/area", json={"regions": [{"source": "sam", "hull": hull, "holes": [hole]}]}).json
        self.assertLess(new["area_m2"], old["area_m2"])

    def test_projected_window_returns_original_geographic_outline(self):
        import rasterio
        from rasterio.warp import transform as project
        from core.preview import _preview_georef
        affine = Affine(0.2, 0.03, 500000, 0.01, -0.2, 3300000)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "source.tif"
            with rasterio.open(path, "w", driver="GTiff", width=800, height=600, count=3,
                               dtype="uint8", crs="EPSG:32649", transform=affine) as dst:
                dst.write(np.full((3, 600, 800), 73, dtype=np.uint8))
                self.preview["georef"] = _preview_georef(dst, 60, 80)
            self.preview["source_paths"] = {"preview": str(path)}
            prediction = {"regions": [{"hull": [[20, 20], [90, 20], [90, 90], [20, 90]],
                                       "holes": [[[60, 60], [70, 60], [70, 70], [60, 70]]]}]}
            with patch.dict(self.web.CONFIG, {"sam_roi": {"window_size": 128, "max_window_size": 512,
                                                         "padding": 16}}), \
                    patch.object(self.web._sam_service, "predict", return_value=prediction) as predict:
                resp = self.client.post("/api/roi/sam", json={"preview_id": "test",
                                        "points": [{"x": 40, "y": 30, "label": 1}]})
        self.assertEqual(resp.status_code, 200, resp.json)
        _, ox, oy, _, _ = predict.call_args.args[1]
        self.assertGreater(ox, 0)
        self.assertGreater(oy, 0)
        region = resp.json["regions"][0]
        expected_x, expected_y = affine * (ox + 20, oy + 20)
        lons, lats = project("EPSG:32649", "EPSG:4326", [expected_x], [expected_y])
        self.assertAlmostEqual(region["hull"][0]["lon"], lons[0], places=10)
        self.assertAlmostEqual(region["hull"][0]["lat"], lats[0], places=10)
        self.assertAlmostEqual(region["hull"][0]["x"], (ox + 20) / 10)
        self.assertAlmostEqual(region["holes"][0][0]["x"], (ox + 60) / 10)
        self.assertEqual(resp.json["points"][0]["x"], 40)


if __name__ == "__main__":
    unittest.main(verbosity=2)
