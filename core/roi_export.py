"""生成感兴趣区域的 Shapefile 配套文件，并调用本地保存窗口。"""

from io import BytesIO
import json
from pathlib import Path
import subprocess
import sys

import shapefile


def export_roi_shapefile(regions):
    """每个区域对应一个面，保留凹边界及内部空洞，坐标为 WGS84 经纬度。"""
    def ring(points, clockwise):
        coords = [list(p) for p in points]
        if coords[-1] == coords[0]:
            coords.pop()
        # 平移后计算方向，减少经纬度数值相减带来的精度损失。
        x0, y0 = coords[0]
        area = sum((a[0] - x0) * (b[1] - y0) - (b[0] - x0) * (a[1] - y0)
                   for a, b in zip(coords, coords[1:] + coords[:1]))
        if len(coords) < 3 or area == 0:
            raise ValueError("区域边界不能退化为点或直线")
        # Shapefile 外环顺时针，内环逆时针。
        if (area < 0) != clockwise:
            coords.reverse()
        return coords + [coords[0]]

    shp, shx, dbf = BytesIO(), BytesIO(), BytesIO()
    with shapefile.Writer(shp=shp, shx=shx, dbf=dbf,
                          shapeType=shapefile.POLYGON, encoding="utf-8") as writer:
        writer.field("region_id", "N", size=10, decimal=0)
        writer.field("name", "C", size=80)
        writer.field("source", "C", size=10)
        for index, region in enumerate(regions, 1):
            parts = []
            for part in region.get("parts") or [region]:
                parts.append(ring(part["hull"], True))
                parts.extend(ring(hole, False) for hole in part.get("holes", []))
            writer.poly(parts)
            writer.record(index, f"区域{index}", region["source"])

    return {
        "shp": shp.getvalue(), "shx": shx.getvalue(), "dbf": dbf.getvalue(),
        "cpg": b"UTF-8",
        "prj": (
            'GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",'
            'SPHEROID["WGS_1984",6378137,298.257223563]],'
            'PRIMEM["Greenwich",0],UNIT["Degree",0.0174532925199433]]'
        ).encode("ascii"),
    }


def choose_shapefile_path(default_name="感兴趣区域"):
    """在独立进程中显示保存窗口，避免后台请求线程与图形界面线程冲突。"""
    result = subprocess.run(
        [sys.executable, str(Path(__file__).with_name("roi_save_dialog.py")), default_name],
        capture_output=True, text=True, encoding="utf-8", check=True,
        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
    )
    selected = json.loads(result.stdout)
    return Path(selected) if selected else None
