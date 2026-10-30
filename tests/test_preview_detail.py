"""原图细节、透明度、请求限制和超大影像窗口读取的回归验证。"""

import sys
import unittest
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import app as web
import numpy as np
import rasterio
from affine import Affine
from PIL import Image
from rasterio.windows import Window
from core import preview_detail as detail
from core.preview import generate_input_preview


class PreviewDetailTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "rgb.tif"
        y, x = np.indices((256, 512))
        self.data = np.stack([x % 256, y, (x + y) % 256]).astype("uint8")
        with rasterio.open(self.path, "w", driver="GTiff", width=512, height=256,
                           count=3, dtype="uint8", transform=Affine(1, 0, 10, 0, -1, 20)) as dst:
            dst.write(self.data)
        self.preview = generate_input_preview(self.path, {}, Path(self.tmp.name) / "small.jpg", 64)
        self.preview["preview_id"] = "当前影像"
        session = web.SessionState()
        session.input_preview = self.preview
        patcher = patch.object(web, "SESSION", session)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.client = web.app.test_client()
        self.params = dict(preview_id="当前影像", x=10, y=5, w=8, h=8, width=64, height=64)

    def test_original_pixels_and_coordinate_alignment(self):
        response = self.client.get("/api/preview/detail", query_string=self.params)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["Cache-Control"], "no-store")
        pixels = np.asarray(Image.open(BytesIO(response.data)))
        np.testing.assert_array_equal(pixels[..., :3], self.data[:, 40:104, 80:144].transpose(1, 2, 0))
        self.assertTrue((pixels[..., 3] == 255).all())

    def test_invalid_and_unbounded_requests_rejected_before_read(self):
        for overrides in ({"x": -1}, {"w": 1000}, {"x": "nan"}, {"h": "inf"},
                          {"width": 100000}, {"width": 2048, "height": 2048}, {"height": 0}):
            with self.subTest(overrides=overrides), patch.object(detail, "render_detail") as render:
                response = self.client.get("/api/preview/detail", query_string={**self.params, **overrides})
                self.assertEqual(response.status_code, 400)
                render.assert_not_called()

    def test_stale_image_and_concurrency(self):
        response = self.client.get("/api/preview/detail", query_string={**self.params, "preview_id": "旧影像"})
        self.assertEqual(response.status_code, 409)
        detail.READ_SLOT.acquire()
        try:
            self.assertEqual(self.client.get("/api/preview/detail", query_string=self.params).status_code, 429)
        finally:
            detail.READ_SLOT.release()

    def test_failure_releases_read_slot(self):
        with patch.object(detail, "render_detail", side_effect=OSError("读取失败")):
            self.assertEqual(self.client.get("/api/preview/detail", query_string=self.params).status_code, 422)
        self.assertTrue(detail.READ_SLOT.acquire(blocking=False))
        detail.READ_SLOT.release()

    def test_image_replaced_during_read_discards_response(self):
        def replace_image(*args):
            web.SESSION.input_preview = dict(self.preview, preview_id="新影像")
            return BytesIO("过期图片".encode("utf-8"))
        with patch.object(detail, "render_detail", side_effect=replace_image):
            self.assertEqual(self.client.get("/api/preview/detail", query_string=self.params).status_code, 409)
        self.assertTrue(detail.READ_SLOT.acquire(blocking=False))
        detail.READ_SLOT.release()

    def test_composite_uses_same_global_stretch(self):
        info = dict(self.preview, source_paths={role: str(self.path) for role in ("B4", "B3", "B2")})
        info["georef"] = dict(info["georef"], display_limits=[(0, 510)] * 3)
        pixels = np.asarray(Image.open(detail.render_detail(info, [10, 5, 8, 8], [64, 64])))
        expected = self.data[0, 40:104, 80:144] // 2
        for channel in range(3):
            np.testing.assert_array_equal(pixels[..., channel], expected)

    def test_alpha_preserved(self):
        path = Path(self.tmp.name) / "rgba.tif"
        with rasterio.open(path, "w", driver="GTiff", width=512, height=256, count=4,
                           dtype="uint8", transform=Affine(1, 0, 10, 0, -1, 20)) as dst:
            dst.write(self.data, indexes=[1, 2, 3])
            alpha = np.full((256, 512), 255, dtype="uint8")
            alpha[40:60, 80:100] = 0
            dst.write(alpha, 4)
        info = dict(self.preview, source_paths={"preview": str(path)})
        pixels = np.asarray(Image.open(detail.render_detail(info, [10, 5, 8, 8], [64, 64])))
        self.assertTrue((pixels[:20, :20, 3] == 0).all())
        self.assertEqual(pixels[-1, -1, 3], 255)

    def test_thirty_gigabyte_logical_raster_reads_small_window(self):
        # 稀疏 TIFF 表示约 30 GB 原始像素，测试本身不分配整图。
        path = Path(self.tmp.name) / "huge.tif"
        with rasterio.open(path, "w", driver="GTiff", width=100000, height=100000,
                           count=3, dtype="uint8", tiled=True, blockxsize=256, blockysize=256,
                           BIGTIFF="YES", SPARSE_OK="YES", transform=Affine(1, 0, 10, 0, -1, 20)) as dst:
            dst.write(np.full((3, 256, 256), 123, dtype="uint8"), window=Window(0, 0, 256, 256))
        info = {"source_paths": {"preview": str(path)}, "georef": {
            "source_width": 100000, "source_height": 100000, "preview_width": 1000, "preview_height": 1000}}
        pixels = np.asarray(Image.open(detail.render_detail(info, [0, 0, 0.64, 0.64], [64, 64])))
        self.assertEqual(pixels.shape, (64, 64, 4))
        self.assertTrue((pixels[..., :3] == 123).all())


if __name__ == "__main__":
    unittest.main()
