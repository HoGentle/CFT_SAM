"""区域合并、滑窗分割、引导边界、空洞和多部分导出的回归测试。"""

import sys
import unittest
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.roi_edit import edit_regions
from core.roi_export import export_roi_shapefile
from core.roi import polygons_lonlat_to_pixel, rasterize_polygons_in_window
from core.stats import lonlat_polygons_area
import app as web
import numpy as np
import rasterio
import shapefile
from affine import Affine
from rasterio.windows import Window
from shapely.geometry import Polygon
from shapely.ops import unary_union


def geometry(region):
    return unary_union([Polygon([(p["x"], p["y"]) for p in part["hull"]],
                                [[(p["x"], p["y"]) for p in h] for h in part.get("holes", [])])
                        for part in region.get("parts") or [region]])


class RoiEditTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "rgb.tif"
        self.transform = Affine(0.00001, 0, 113.5, 0, -0.00001, 30.5)
        self.create_image()
        self.parent = {"hull": [[10, 10], [170, 10], [170, 130], [10, 130]],
                       "holes": [[[25, 40], [35, 40], [35, 50], [25, 50]]]}

    def create_image(self, width=180, height=140, boundary=True):
        image = np.zeros((3, height, width), dtype="uint8")
        image[:] = np.array([90, 150, 65])[:, None, None]
        if boundary:
            image[:, :, width // 2 - 4:width // 2 + 4] = 15
        with rasterio.open(self.path, "w", driver="GTiff", width=width, height=height,
                           count=3, dtype="uint8", crs="EPSG:4326", transform=self.transform) as dst:
            dst.write(image)
        self.preview = {"preview_id": "test", "mode": "result_preview", "source_paths": {"preview": str(self.path)},
                        "georef": {"preview_width": width, "preview_height": height, "source_width": width,
                                   "source_height": height, "crs": "EPSG:4326", "transform": list(self.transform[:6])}}

    def assert_partition(self, result, parent=None):
        polygons = [geometry(r) for r in result["regions"]]
        target = parent or Polygon(self.parent["hull"], self.parent["holes"])
        self.assertGreaterEqual(len(polygons), 2)
        self.assertLess(unary_union(polygons).symmetric_difference(target).area, 1e-6)
        self.assertAlmostEqual(sum(p.area for p in polygons), target.area, places=6)
        self.assertTrue(all(r["points"] == [] for r in result["regions"]))

    def test_merge_connects_gap_preserves_hole_and_exports_one_polygon(self):
        a = {"hull": [[10, 10], [40, 10], [40, 70], [10, 70]],
             "holes": [[[20, 20], [30, 20], [30, 30], [20, 30]]]}
        b = {"hull": [[42, 10], [72, 10], [72, 70], [42, 70]]}
        result = edit_regions(self.preview, [a, b], "merge")
        merged = result["regions"][0]
        self.assertNotIn("parts", merged)
        self.assertEqual(len(merged["holes"]), 1)
        self.assertEqual(geometry(merged).geom_type, "Polygon")
        self.assertTrue(geometry(merged).is_valid)
        self.assertGreater(geometry(merged).area, 3500)
        cleaned = web.clean_roi_region({**merged,
            "hull": [[p["lon"], p["lat"]] for p in merged["hull"]],
            "holes": [[[p["lon"], p["lat"]] for p in h] for h in merged["holes"]],
        })
        pixels = polygons_lonlat_to_pixel([cleaned], self.transform, "EPSG:4326")
        mask = rasterize_polygons_in_window(pixels, Window(0, 0, 180, 140))
        self.assertGreater(int(mask.sum()), 3500)
        self.assertFalse(mask[25, 25])
        self.assertTrue(mask[40, 41])
        whole = lonlat_polygons_area([cleaned])["area_m2"]
        self.assertGreater(whole, 0)
        files = export_roi_shapefile([cleaned])
        with shapefile.Reader(shp=BytesIO(files["shp"]), dbf=BytesIO(files["dbf"])) as reader:
            self.assertEqual(len(reader), 1)
            self.assertEqual(len(reader.shape(0).parts), 2)
            self.assertEqual(reader.shape(0).__geo_interface__["type"], "Polygon")

    @staticmethod
    def rectangle(x, y, width=20, height=50):
        return {"hull": [[x,y],[x+width,y],[x+width,y+height],[x,y+height]]}

    def metric_preview(self):
        return {"georef": {**self.preview["georef"],
                "crs": "+proj=aeqd +lat_0=30.5 +lon_0=113.5 +datum=WGS84 +units=m +no_defs",
                "transform": [1,0,-90,0,-1,70]}}

    def test_merge_distance_default_custom_limit_and_unlimited(self):
        regions = [self.rectangle(10,10), self.rectangle(36,10)]
        with self.assertRaisesRegex(ValueError, "距离过远无法合并"):
            edit_regions(self.metric_preview(), regions, "merge")
        for limit in (6.01, -1):
            result = edit_regions(self.metric_preview(), regions, "merge", merge_max_distance_m=limit)
            self.assertEqual(geometry(result["regions"][0]).geom_type, "Polygon")
            self.assertAlmostEqual(result["merge_distances_m"][0], 6, places=5)
        # 恰好五米允许；超过五米拒绝，失败不修改输入。
        regions[1] = self.rectangle(35,10)
        self.assertEqual(len(edit_regions(self.metric_preview(), regions, "merge")["regions"]), 1)
        regions[1] = self.rectangle(35.01,10)
        with self.assertRaisesRegex(ValueError, "5.010"):
            edit_regions(self.metric_preview(), regions, "merge")
        self.assertEqual(regions[1]["hull"][0], [35.01,10])

    def test_merge_measures_edges_not_centers_or_vertices(self):
        regions = [self.rectangle(10,10,20,100), self.rectangle(34,40,20,20)]
        result = edit_regions(self.metric_preview(), regions, "merge")
        self.assertAlmostEqual(result["merge_distances_m"][0], 4, places=5)
        self.assertEqual(geometry(result["regions"][0]).geom_type, "Polygon")

    def test_merge_chain_checks_original_edges_and_rejects_far_cluster(self):
        regions = [self.rectangle(x,10) for x in (10,34,58)]
        result = edit_regions(self.metric_preview(), regions, "merge")
        self.assertEqual(len(result["merge_distances_m"]), 2)
        self.assertEqual(geometry(result["regions"][0]).geom_type, "Polygon")
        regions.append(self.rectangle(90,10))
        with self.assertRaisesRegex(ValueError, "距离过远"):
            edit_regions(self.metric_preview(), regions, "merge")

    def test_merge_point_contact_and_shared_edge(self):
        a = self.rectangle(10,10,20,20)
        for b in (self.rectangle(30,30,20,20), self.rectangle(30,10,20,20)):
            result = edit_regions(self.metric_preview(), [a,b], "merge", merge_max_distance_m=0)
            polygon = geometry(result["regions"][0])
            self.assertEqual(polygon.geom_type, "Polygon")
            self.assertTrue(polygon.is_valid)
            self.assertGreaterEqual(polygon.area, 800 - 1e-7)

    def test_merge_geographic_latitude_and_projected_feet_use_meters(self):
        regions = [self.rectangle(10,10), self.rectangle(40,10)]
        for latitude, allowed in ((75,True),(0,False)):
            preview = {"georef": {**self.preview["georef"], "transform": [0.00001,0,113.5,0,-0.00001,latitude]}}
            if allowed:
                edit_regions(preview, regions, "merge")
            else:
                with self.assertRaisesRegex(ValueError, "距离过远"):
                    edit_regions(preview, regions, "merge")
        feet = {"georef": {**self.preview["georef"], "crs":"EPSG:2263", "transform":[1,0,980000,0,-1,200000]}}
        result = edit_regions(feet, regions, "merge")
        self.assertAlmostEqual(result["merge_distances_m"][0], 3.048, places=3)
        with self.assertRaisesRegex(ValueError, "距离过远"):
            edit_regions(feet, [regions[0],self.rectangle(50,10)], "merge")

    def test_merge_source_scale_rotation_and_obstacle(self):
        preview = self.metric_preview()
        preview["georef"].update(source_width=1800,source_height=1400,
            transform=list((Affine.translation(-90,70)*Affine.rotation(30)*Affine.scale(0.2,-0.8))[:6]))
        with self.assertRaisesRegex(ValueError, "距离过远"):
            edit_regions(preview,[self.rectangle(10,10),self.rectangle(33,10)],"merge")
        with self.assertRaisesRegex(ValueError, "未选中"):
            edit_regions(self.metric_preview(),[self.rectangle(10,10),self.rectangle(34,10)],"merge",
                         excluded_regions=[self.rectangle(31,0,2,100)])

    def test_merge_rejects_invalid_distance_values(self):
        regions = [self.rectangle(10,10), self.rectangle(34,10)]
        for value in (-2,-0.5,float("nan"),float("inf"),None,True,"非法"):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError,"最大合并距离"):
                edit_regions(self.metric_preview(),regions,"merge",merge_max_distance_m=value)

    def test_merge_overlaps_is_true_union(self):
        first = {"hull": [[10, 10], [50, 10], [50, 50], [10, 50]]}
        second = {"hull": [[30, 10], [70, 10], [70, 50], [30, 50]]}
        result = edit_regions(self.preview, [first, second], "merge")
        self.assertAlmostEqual(geometry(result["regions"][0]).area, 2400)

    def test_automatic_split_uses_image_boundary_and_keeps_full_parent(self):
        result = edit_regions(self.preview, [self.parent], "split")
        self.assert_partition(result)
        self.assertEqual(len(result["regions"]), 2)
        self.assertGreater(result["scan_windows"], 0)

    def test_polyline_guidance_follows_nearby_boundary(self):
        result = edit_regions(self.preview, [self.parent], "split", [[90, 30], [90, 110]])
        self.assert_partition(result)
        self.assertEqual(len(result["regions"]), 2)

    def test_automatic_split_can_find_three_regions(self):
        with rasterio.open(self.path, "r+") as dst:
            image = dst.read()
            image[:, 68:76, 90:] = 15
            dst.write(image)
        result = edit_regions(self.preview, [self.parent], "split")
        self.assert_partition(result)
        self.assertEqual(len(result["regions"]), 3)

    def test_multiple_native_windows(self):
        self.create_image(2200, 240)
        parent = {"hull": [[10, 10], [2190, 10], [2190, 230], [10, 230]], "holes": []}
        result = edit_regions(self.preview, [parent], "split")
        self.assert_partition(result, Polygon(parent["hull"]))
        self.assertGreaterEqual(result["scan_windows"], 3)

    def test_guided_split_can_reach_image_edges(self):
        parent = {"hull": [[0,0],[180,0],[180,140],[0,140]], "holes": []}
        result = edit_regions(self.preview, [parent], "split", [[90,30],[90,110]])
        self.assert_partition(result, Polygon(parent["hull"]))

    def test_flat_image_does_not_invent_boundaries(self):
        self.create_image(boundary=False)
        for guide in (None, [[86, 30], [86, 110]]):
            with self.subTest(guide=guide), self.assertRaises(ValueError):
                edit_regions(self.preview, [self.parent], "split", guide)

    def test_validation_and_stale_preview(self):
        for operation, regions, guide in (("merge", [self.parent], None), ("split", [self.parent]*2, None),
                                           ("split", [self.parent], [[50, 50]]), ("split", [self.parent], [[-1,0],[5,6]])):
            with self.subTest(operation=operation, guide=guide), self.assertRaises(ValueError):
                edit_regions(self.preview, regions, operation, guide)
        session = web.SessionState()
        session.input_preview = self.preview
        with patch.object(web, "SESSION", session):
            response = web.app.test_client().post("/api/roi/edit", json={"preview_id":"old", "operation":"split", "regions":[self.parent]})
            self.assertEqual(response.status_code, 409)
            response = web.app.test_client().post("/api/roi/edit", json={"preview_id":"test", "operation":"split", "regions":[self.parent]})
            self.assertEqual(response.status_code, 200)
            self.assert_partition(response.json)
            regions = [self.rectangle(10,10),self.rectangle(50,10)]
            response = web.app.test_client().post("/api/roi/edit", json={"preview_id":"test", "operation":"merge", "regions":regions})
            self.assertEqual(response.status_code,400)
            self.assertIn("距离过远",response.json["error"])
            response = web.app.test_client().post("/api/roi/edit", json={"preview_id":"test", "operation":"merge", "regions":regions,
                                                     "merge_max_distance_m":-1})
            self.assertEqual(response.status_code,200)
            self.assertEqual(geometry(response.json["regions"][0]).geom_type,"Polygon")
            response = web.app.test_client().post("/api/roi/edit", json={"preview_id":"test", "operation":"merge", "regions":regions,
                                                     "merge_max_distance_m":-2})
            self.assertEqual(response.status_code,400)


if __name__ == "__main__":
    unittest.main()
