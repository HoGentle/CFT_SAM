"""系统模式、单张 RGB 预览、默认导出名称及投影坐标的回归验证。"""

import sys
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import app as web
import numpy as np
import rasterio
from affine import Affine
from core.inputs import InputBundle, InputFile, scan_folder, validate_bundle
from core.jobs import JobManager


class SystemModeTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.folder = Path(self.directory.name)
        self.rgb = self.folder / "result..tif"
        with rasterio.open(self.rgb, "w", driver="GTiff", width=30, height=20,
                           count=3, dtype="uint8", crs="EPSG:4326",
                           transform=Affine(0.00001, 0, 113.5, 0, -0.00001, 30.5)) as dst:
            dst.write(np.full((3, 20, 30), 120, dtype=np.uint8))
        self.bundle = InputBundle(files={"preview": InputFile(self.rgb, self.rgb.name, self.rgb.stat().st_size, "local")})
        self.session = web.SessionState()
        self.session.bundle = self.bundle
        self.manager = JobManager()
        for name, value in (("SESSION", self.session), ("_job_manager", self.manager),
                            ("PREVIEW_DIR", self.folder), ("CURRENT_DIR", self.folder / "current")):
            patcher = patch.object(web, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        web.CURRENT_DIR.mkdir()
        self.client = web.app.test_client()

    def test_rgb_only_validation_and_real_preview(self):
        self.assertEqual(self.client.post("/api/mode", json={"mode": "annotation"}).status_code, 200)
        response = self.client.post("/api/validate", json={})
        self.assertEqual(response.json["errors"], [])
        deadline = time.monotonic() + 8
        while self.manager.is_running() and time.monotonic() < deadline:
            time.sleep(0.02)
        self.assertFalse(self.manager.is_running())
        self.assertIsNone(self.manager.current_status()["error"])
        session = self.client.get("/api/session").json
        self.assertEqual(session["missing"], [])
        self.assertTrue(session["has_input_preview"])
        self.assertEqual(session["georef"]["source_width"], 30)
        self.assertGreater(session["input_area"]["area_m2"], 0)
        self.assertIsNone(self.session.band_paths)
        self.assertEqual(self.client.post("/api/run", json={}).status_code, 400)

    def test_prescription_still_requires_four_channels(self):
        with patch("app.start_preview_job") as start:
            response = self.client.post("/api/validate", json={})
        self.assertEqual(len(response.json["errors"]), 4)
        start.assert_not_called()

    def test_mode_switch_invalidates_previous_validation(self):
        self.session.validation = {"errors": []}
        self.session.input_preview = {"preview_id": "旧预览"}
        self.session.band_paths = {"B2": self.rgb}
        response = self.client.post("/api/mode", json={"mode": "annotation"})
        self.assertEqual(response.json["mode"], "annotation")
        self.assertIsNone(self.session.validation)
        self.assertIsNone(self.session.input_preview)
        self.assertIsNone(self.session.band_paths)
        self.assertEqual(self.client.post("/api/mode", json={"mode": "无效"}).status_code, 400)

    def test_default_shp_name_from_rgb(self):
        region = {"source": "manual", "hull": [[113, 30], [114, 30], [114, 31]]}
        with patch("app.choose_shapefile_path", return_value=None) as choose:
            response = self.client.post("/api/roi/export-shp", json={"regions": [region]})
        self.assertTrue(response.json["cancelled"])
        choose.assert_called_once_with("result")

    def test_rgb_with_channel_keyword_and_repeated_upload(self):
        self.session.mode = "annotation"
        for _ in range(2):
            part = self.folder / "upload.part"
            part.write_bytes(self.rgb.read_bytes())
            with patch.dict(web._UPLOADS, {"test": {"name": "Green田块.tif", "size": part.stat().st_size, "path": part}}):
                response = self.client.post("/api/upload/finish", json={"upload_id": "test"})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json["role"], "preview")
            self.assertTrue(Path(self.session.bundle.files["preview"].path).is_file())

    def test_annotation_folder_ignores_single_band(self):
        single = self.folder / "result_Green.tif"
        with rasterio.open(single, "w", driver="GTiff", width=2, height=2, count=1,
                           dtype="uint8", crs="EPSG:4326", transform=Affine(1, 0, 113, 0, -1, 31)) as dst:
            dst.write(np.ones((1, 2, 2), dtype=np.uint8))
        bundle, skipped = scan_folder(self.folder, web.CONFIG["recognition"], mode="annotation")
        self.assertEqual(set(bundle.files), {"preview"})
        self.assertIn(single.name, skipped)
        self.assertEqual(validate_bundle(bundle, mode="annotation")[0], [])

    def test_projected_manual_region_is_converted_to_lonlat(self):
        self.session.input_preview = {"preview_id": "投影预览", "georef": {
            "preview_width": 100, "preview_height": 100, "source_width": 100, "source_height": 100,
            "crs": "EPSG:32649", "transform": [1, 0, 500000, 0, -1, 3400000]}}
        region = web.clean_roi_region({"source": "manual", "preview_id": "投影预览",
                                       "pixel_hull": [[0, 0], [50, 0], [50, 50]], "pixel_points": []})
        self.assertTrue(110 < region["hull"][0][0] < 112)
        self.assertTrue(30 < region["hull"][0][1] < 32)
        with self.assertRaises(ValueError):
            web.clean_roi_region({"source": "manual", "preview_id": "过期预览", "pixel_hull": [[0, 0]]})


if __name__ == "__main__":
    unittest.main()
