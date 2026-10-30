"""验证区域文件直接保存、凹边界及空洞保留，以及取消和覆盖处理。"""

import sys
import unittest
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import shapefile
from core.roi_export import export_roi_shapefile

from rasterio.crs import CRS
from rasterio.features import rasterize
from affine import Affine

class RoiExportTests(unittest.TestCase):
    def setUp(self):
        from app import app
        self.client = app.test_client()
        self.regions = [
            {"source": "manual", "hull": [[0, 0], [5, 4], [10, 0], [10, 10], [0, 10]]},
            {"source": "sam", "hull": [[20, 0], [30, 0], [30, 10], [20, 10]],
             "holes": [[[23, 3], [27, 3], [27, 7], [23, 7]]]},
        ]

    def test_saved_files_and_geometry(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "田块.规划.shp"
            with patch("app.choose_shapefile_path", return_value=path):
                response = self.client.post("/api/roi/export-shp", json={"regions": self.regions})
            self.assertEqual(response.status_code, 200)
            self.assertFalse(response.json["cancelled"])
            self.assertEqual(response.json["directory"], directory)
            self.assertEqual(set(Path(directory).iterdir()), {
                path.with_suffix(f".{suffix}") for suffix in ("shp", "shx", "dbf", "prj", "cpg")})
            self.assertEqual(path.with_suffix(".cpg").read_bytes(), b"UTF-8")
            self.assertEqual(CRS.from_wkt(path.with_suffix(".prj").read_text()).to_epsg(), 4326)
            with shapefile.Reader(str(path), encoding="utf-8") as reader:
                self.assertEqual(len(reader), 2)
                self.assertEqual(reader.record(0).as_dict(), {
                    "region_id": 1, "name": "区域1", "source": "manual"})
                first, second = reader.shapes()
                self.assertEqual(len(first.points), 6)
                self.assertEqual(set(map(tuple, first.points)), set(map(tuple, self.regions[0]["hull"])))
                self.assertEqual(len(second.parts), 2)
                geometry = [shape.__geo_interface__ for shape in (first, second)]
                self.assertEqual(len(geometry[1]["coordinates"]), 2)
                mask = rasterize([(g, 1) for g in geometry], out_shape=(10, 30), transform=Affine.identity())
                self.assertFalse(mask[1, 5])
                self.assertTrue(mask[6, 5])
                self.assertFalse(mask[4, 24])
                self.assertTrue(mask[1, 24])

    def test_invalid_requests(self):
        for payload in ({}, {"regions": []}, {"regions": "无效"}, [1],
                        {"regions": [{"hull": [[0, 0], [1, 1]]}]},
                        {"regions": [{"hull": [[0, 0], [1, 1], [2, 2]]}]},
                        {"regions": [{"hull": [[0, 0], [1, 1], [None, 2]]}]}):
            with self.subTest(payload=payload):
                response = self.client.post("/api/roi/export-shp", json=payload)
                self.assertEqual(response.status_code, 400)
                self.assertIn("error", response.json)

    def test_closed_rings(self):
        outline = [[113, 30], [114, 30], [114, 31], [113, 31], [113, 30]]
        files = export_roi_shapefile([{"source": "manual", "hull": outline}])
        with shapefile.Reader(shp=BytesIO(files["shp"]), dbf=BytesIO(files["dbf"])) as reader:
            self.assertEqual(len(reader.shape(0).points), 5)

    def test_cancel_does_not_write(self):
        with patch("app.choose_shapefile_path", return_value=None), patch.object(Path, "write_bytes") as write:
            response = self.client.post("/api/roi/export-shp", json={"regions": self.regions})
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json["cancelled"])
        write.assert_not_called()

    def test_save_failure(self):
        with patch("app.choose_shapefile_path", side_effect=OSError("无法打开保存窗口")):
            response = self.client.post("/api/roi/export-shp", json={"regions": self.regions})
        self.assertEqual(response.status_code, 500)
        self.assertIn("无法打开保存窗口", response.json["error"])

    def test_existing_companion_requires_confirmation(self):
        from core.roi_save_dialog import choose_path
        with TemporaryDirectory() as directory:
            path = Path(directory) / "田块.shp"
            path.with_suffix(".dbf").write_bytes("原有属性".encode("utf-8"))
            with patch("core.roi_save_dialog.tk.Tk"), \
                    patch("core.roi_save_dialog.filedialog.asksaveasfilename", side_effect=[str(path), ""]), \
                    patch("core.roi_save_dialog.messagebox.askyesno", return_value=False) as confirm:
                self.assertIsNone(choose_path())
                confirm.assert_called_once()
            self.assertEqual(path.with_suffix(".dbf").read_bytes(), "原有属性".encode("utf-8"))


if __name__ == "__main__":
    unittest.main()
