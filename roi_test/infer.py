"""使用本地地块模型测试带地理坐标的无人机影像。"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import rasterio
import torch
from affine import Affine
from PIL import Image, ImageDraw
from pyproj import CRS, Geod, Transformer
from rasterio.enums import ColorInterp, Resampling
from rasterio.features import rasterize, shapes
from shapely.geometry import mapping, shape
from shapely.ops import transform as transform_geometry
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "result.tif", help="输入影像路径")
    parser.add_argument("--weights", type=Path,
                        default=ROOT / "checkpoint" / "DelineateAnythingv2.pt", help="本地权重路径")
    parser.add_argument("--output", type=Path, default=ROOT / "roi_test" / "outputs", help="输出目录")
    parser.add_argument("--resolution", type=float, default=0.25, help="识别影像地面采样间距，单位米")
    parser.add_argument("--imgsz", type=int, default=1536, help="模型输入长边，向上取整为32的倍数")
    parser.add_argument("--conf", type=float, default=0.15, help="最低置信度")
    parser.add_argument("--iou", type=float, default=0.5, help="检测去重的交并比阈值")
    parser.add_argument("--min-area", type=float, default=10, help="最小地块面积，单位平方米")
    parser.add_argument("--device", default="0" if torch.cuda.is_available() else "cpu", help="显卡编号或cpu")
    parser.add_argument("--max-det", type=int, default=1000, help="最多候选数量")
    args = parser.parse_args()
    if not args.input.is_file() or not args.weights.is_file():
        parser.error("输入影像或权重文件不存在")
    if (not math.isfinite(args.resolution) or args.resolution <= 0
            or not math.isfinite(args.min_area) or args.min_area < 0
            or args.imgsz < 32 or args.max_det < 1
            or not 0 < args.conf < 1 or not 0 < args.iou < 1):
        parser.error("分辨率、面积、输入尺寸或阈值无效")
    return args


def read_image(path, resolution):
    with rasterio.open(path) as src:
        if not src.crs or src.count < 3:
            raise ValueError("输入必须具有坐标参考系和至少三个颜色波段")
        crs = CRS.from_wkt(src.crs.to_wkt())
        to_lonlat = Transformer.from_crs(crs, CRS.from_epsg(4326), always_xy=True)
        center = np.array(src.transform * (src.width / 2, src.height / 2))
        x_next = np.array(src.transform * (src.width / 2 + 1, src.height / 2))
        y_next = np.array(src.transform * (src.width / 2, src.height / 2 + 1))
        lon, lat = to_lonlat.transform(*center)
        geod = Geod(ellps="WGS84")
        pixel_x = geod.inv(lon, lat, *to_lonlat.transform(*x_next))[2]
        pixel_y = geod.inv(lon, lat, *to_lonlat.transform(*y_next))[2]
        width = max(1, round(src.width * pixel_x / resolution))
        height = max(1, round(src.height * pixel_y / resolution))
        if max(width, height) > 8192:
            raise ValueError("降采样影像长边超过8192，请增大 --resolution；此脚本用于整幅影像测试")
        if width > src.width or height > src.height:
            raise ValueError("目标采样间距小于源影像，请增大 --resolution")
        print(f"读取影像：{src.width}×{src.height}，原始采样间距约 {pixel_x:.4f} 米", flush=True)
        print(f"生成识别影像：{width}×{height}，目标采样间距 {resolution} 米", flush=True)
        colors = list(src.colorinterp)
        if all(color in colors for color in (ColorInterp.red, ColorInterp.green, ColorInterp.blue)):
            indexes = [colors.index(color) + 1 for color in
                       (ColorInterp.red, ColorInterp.green, ColorInterp.blue)]
        else:
            indexes = [1, 2, 3]
        rgb = src.read(indexes, out_shape=(3, height, width), resampling=Resampling.average)
        if ColorInterp.alpha in colors:
            valid = src.read(colors.index(ColorInterp.alpha) + 1,
                             out_shape=(height, width), resampling=Resampling.nearest) > 0
        else:
            valid = src.dataset_mask(out_shape=(height, width), resampling=Resampling.nearest) > 0
        valid &= np.isfinite(rgb).all(axis=0)
        if not valid.any():
            raise ValueError("影像没有有效像元")
        if rgb.dtype != np.uint8:
            channels = []
            for band in rgb:
                low, high = np.percentile(band[valid], [2, 98])
                channels.append(np.clip((band - low) / max(float(high - low), 1e-6) * 255,
                                        0, 255).astype(np.uint8))
            rgb = np.stack(channels)
        rgb[:, ~valid] = 0
        transform = src.transform * Affine.scale(src.width / width, src.height / height)
        info = {"source_width": src.width, "source_height": src.height,
                "source_pixel_m": [pixel_x, pixel_y], "width": width, "height": height,
                "actual_pixel_m": [pixel_x * src.width / width, pixel_y * src.height / height],
                "source_crs": crs.to_wkt()}
        return np.moveaxis(rgb, 0, -1), valid, transform, crs, to_lonlat, info


def main():
    args = parse_args()
    started = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=True)
    rgb, valid, affine, crs, to_lonlat, info = read_image(args.input, args.resolution)
    height, width = valid.shape
    profile = dict(driver="GTiff", width=width, height=height,
                   transform=affine, crs=crs.to_wkt(), compress="deflate")
    with rasterio.open(args.output / "inference_rgba.tif", "w", count=4, dtype="uint8", **profile) as dst:
        dst.write(np.moveaxis(rgb, -1, 0), indexes=[1, 2, 3])
        dst.write(valid.astype(np.uint8) * 255, 4)
        dst.colorinterp = (ColorInterp.red, ColorInterp.green, ColorInterp.blue, ColorInterp.alpha)
    Image.fromarray(rgb).save(args.output / "input_preview.jpg")
    print("开始加载模型并推理", flush=True)
    model = YOLO(str(args.weights))
    # 数组输入采用蓝绿红顺序；高分辨率掩膜与降采样栅格一一对应。
    result = model.predict(source=np.ascontiguousarray(rgb[:, :, ::-1]),
                           imgsz=math.ceil(args.imgsz / 32) * 32,
                           conf=args.conf, iou=args.iou, max_det=args.max_det,
                           device=args.device, retina_masks=True, verbose=False)[0]
    labels = np.zeros((height, width), dtype=np.uint32)
    features = []
    preview = Image.fromarray(rgb)
    draw = ImageDraw.Draw(preview)
    geod = Geod(ellps="WGS84")
    confidences = result.boxes.conf.cpu().numpy() if result.boxes is not None else np.array([])
    if result.masks is not None:
        if tuple(result.masks.data.shape[1:]) != (height, width):
            raise RuntimeError("模型掩膜尺寸与识别影像不一致，无法正确转换地理坐标")
        for index in np.argsort(-confidences):
            mask = (result.masks.data[int(index)].cpu().numpy() > 0.5) & valid & (labels == 0)
            for geometry, _ in shapes(mask.astype(np.uint8), mask=mask, transform=affine):
                polygon = shape(geometry)
                geographic = transform_geometry(to_lonlat.transform, polygon)
                area = abs(geod.geometry_area_perimeter(geographic)[0])
                if area < args.min_area:
                    continue
                field_id = len(features) + 1
                features.append({"type": "Feature", "geometry": mapping(geographic),
                                 "properties": {"field_id": field_id,
                                                "confidence": round(float(confidences[index]), 6),
                                                "area_m2": round(area, 3), "candidate_id": int(index)}})
                inverse = ~affine
                rings = [polygon.exterior, *polygon.interiors]
                pixels = [np.round([inverse * point for point in ring.coords]).astype(np.int32)
                          for ring in rings]
                # 按栅格化结果写编号，避免绘图取整改变地理边界。
                piece = rasterize([(geometry, 1)], out_shape=(height, width), transform=affine).astype(bool)
                labels[piece] = field_id
                color = ((field_id * 71) % 180 + 75, (field_id * 113) % 180 + 75,
                         (field_id * 149) % 180 + 75)
                for ring in pixels:
                    draw.line([tuple(p) for p in ring], fill=color, width=2)
                point = inverse * tuple(polygon.representative_point().coords)[0]
                draw.text(point, str(field_id), fill="white", stroke_width=2, stroke_fill="black")
    with rasterio.open(args.output / "field_ids.tif", "w", count=1, dtype="uint32", nodata=0, **profile) as dst:
        dst.write(labels, 1)
    collection = {"type": "FeatureCollection", "features": features}
    (args.output / "fields.geojson").write_text(json.dumps(collection, ensure_ascii=False), encoding="utf-8")
    preview.save(args.output / "overlay.png")
    info.update({"input": str(args.input.resolve()), "weights": str(args.weights.resolve()),
                 "parameters": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                 "candidate_count": len(confidences), "field_count": len(features),
                 "total_area_m2": sum(f["properties"]["area_m2"] for f in features),
                 "elapsed_seconds": round(time.perf_counter() - started, 3),
                 "torch_version": torch.__version__})
    (args.output / "summary.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"完成：{len(confidences)} 个候选，保留 {len(features)} 个区域，耗时 {info['elapsed_seconds']} 秒", flush=True)
    print(f"输出目录：{args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
