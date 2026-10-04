"""界面预览图生成（全部为降采样小图，避免向浏览器传大文件）。

* 输入预览：优先用上传的 result.tif（RGBA 整体预览图）；未上传时用
  B4/B3/B2 合成假彩色近似预览。仅作整体浏览。
* 诊断图预览：分级图按固定配色渲染（红->绿 表示长势差->好）。
* 处方图预览：施肥量按渐变色带渲染，NaN 透明。

注意：通道图 B1-B4 不生成任何界面预览，仅参与计算。
"""

from pathlib import Path
import os

import numpy as np
import rasterio
from rasterio.enums import Resampling
from PIL import Image, ImageDraw, ImageFont

# 分级配色（红=差/低 -> 绿=好/高）
CLASS_PALETTE_5 = ["#d73027", "#fc8d59", "#fee08b", "#d9ef8b", "#1a9850"]
CLASS_PALETTE_4 = ["#d73027", "#fee08b", "#a6d96a", "#1a9850"]

# 处方图连续色带（值低 -> 值高）
PRESCRIPTION_STOPS = [
    (0.00, "#313695"),
    (0.25, "#4575b4"),
    (0.40, "#74add1"),
    (0.55, "#abd9e9"),
    (0.70, "#fee090"),
    (0.85, "#f46d43"),
    (1.00, "#a50026"),
]


def _hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip("#")
    return np.array([int(hex_color[i:i + 2], 16) for i in (0, 2, 4)], dtype=np.uint8)


def build_class_legend(level_count):
    """按级数生成分级配色（红=差/低 -> 绿=好/高）。

    级数 4/5 沿用原固定配色；其它级数在红->绿渐变上等距插值，
    保证任意分级数量（2–20）都有颜色且首尾固定为红/绿。
    """
    level_count = max(2, int(level_count))
    if level_count == 4:
        return list(CLASS_PALETTE_4)
    if level_count == 5:
        return list(CLASS_PALETTE_5)

    stops_pos = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    stop_colors = np.stack([_hex_to_rgb(c) for c in CLASS_PALETTE_5]).astype(np.float32)
    positions = np.linspace(0.0, 1.0, level_count)
    palette = []
    for ch in range(3):
        channel = np.interp(positions, stops_pos, stop_colors[:, ch])
        palette.append(np.clip(channel, 0, 255).astype(np.uint8))
    return ["#{:02x}{:02x}{:02x}".format(*color) for color in np.stack(palette, axis=1)]


def build_prescription_gradient_css():
    stops = ", ".join(f"{color} {int(pos * 100)}%" for pos, color in PRESCRIPTION_STOPS)
    return f"linear-gradient(90deg, {stops})"


def _out_shape(height, width, max_long_side, allow_upscale=False):
    scale = max_long_side / float(max(height, width))
    if scale >= 1.0:
        if not allow_upscale:
            return int(height), int(width)
        return max(1, int(round(height * scale))), max(1, int(round(width * scale)))
    return max(1, int(round(height * scale))), max(1, int(round(width * scale)))


def _stretch_uint8_or_float(array):
    if array.dtype == np.uint8:
        return array
    data = array.astype(np.float32)
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return np.zeros_like(data, dtype=np.uint8)
    low, high = np.percentile(finite, (2, 98))
    if high <= low:
        high = low + 1.0
    return np.clip((data - low) / (high - low) * 255.0, 0, 255).astype(np.uint8)


def generate_input_preview(preview_tif, band_paths, out_jpg, max_long_side=2048, jpeg_quality=88):
    """生成输入影像整体预览（result.tif 优先，否则由波段合成假彩色）。

    返回预览图的地理参考信息，供前端把标点像素坐标换算为经纬度。
    """
    from PIL import Image as PILImage

    out_jpg = Path(out_jpg)
    out_jpg.parent.mkdir(parents=True, exist_ok=True)

    if preview_tif is not None:
        try:
            rgb, mask, georef = _read_rgb_preview(preview_tif, max_long_side)
        except Exception:  # noqa: BLE001 - 预览图损坏时退回波段合成
            rgb, mask, georef = None, None, None
        if rgb is not None:
            _save_jpg(rgb, mask, out_jpg, jpeg_quality)
            return {"source": str(preview_tif), "mode": "result_preview", "georef": georef}

    rgb, mask, georef = _read_composite_preview(band_paths, max_long_side)
    _save_jpg(rgb, mask, out_jpg, jpeg_quality)
    return {"source": "B4/B3/B2 合成假彩色", "mode": "composite", "georef": georef}


def _preview_georef(src, out_h, out_w):
    """读取时的降采样参数 -> 预览像素与源影像像素的对应关系。"""
    return {
        "preview_width": int(out_w),
        "preview_height": int(out_h),
        "source_width": int(src.width),
        "source_height": int(src.height),
        "transform": [float(v) for v in src.transform[:6]],
    }


def _read_rgb_preview(path, max_long_side):
    with rasterio.open(path) as src:
        out_h, out_w = _out_shape(src.height, src.width, max_long_side)
        if src.count >= 3:
            rgb = src.read([1, 2, 3], out_shape=(3, out_h, out_w), resampling=Resampling.average)
        else:
            band = src.read(1, out_shape=(1, out_h, out_w), resampling=Resampling.average)
            rgb = np.repeat(band, 3, axis=0)
        mask = src.dataset_mask(out_shape=(out_h, out_w), resampling=Resampling.average) > 0
        georef = _preview_georef(src, out_h, out_w)

    channels = [_stretch_uint8_or_float(rgb[i]) for i in range(3)]
    return np.stack(channels, axis=0), mask, georef


def _read_composite_preview(band_paths, max_long_side):
    with rasterio.open(band_paths["B4"]) as nir, \
            rasterio.open(band_paths["B3"]) as red, \
            rasterio.open(band_paths["B2"]) as green:
        out_h, out_w = _out_shape(nir.height, nir.width, max_long_side)
        shape = (1, out_h, out_w)
        b4 = nir.read(1, out_shape=shape, resampling=Resampling.average).astype(np.float32)
        b3 = red.read(1, out_shape=shape, resampling=Resampling.average).astype(np.float32)
        b2 = green.read(1, out_shape=shape, resampling=Resampling.average).astype(np.float32)
        mask = nir.dataset_mask(out_shape=(out_h, out_w), resampling=Resampling.average) > 0
        georef = _preview_georef(nir, out_h, out_w)

    rgb = np.stack([b4, b3, b2], axis=0)
    channels = [_stretch_uint8_or_float(rgb[i]) for i in range(3)]
    return np.stack(channels, axis=0), mask, georef


def _save_jpg(rgb, mask, out_jpg, jpeg_quality):
    rgb = np.stack(rgb, axis=-1)  # H W 3
    mask = mask[..., None]
    white = np.full_like(rgb, 255, dtype=np.uint8)
    alpha = np.clip(mask.astype(np.float32), 0, 1)
    composited = (rgb * alpha + white * (1 - alpha)).astype(np.uint8)
    Image.fromarray(composited, mode="RGB").save(out_jpg, quality=jpeg_quality)


def _downsampled_band(path, band_index, max_long_side, allow_upscale=False):
    with rasterio.open(path) as src:
        out_h, out_w = _out_shape(src.height, src.width, max_long_side, allow_upscale)
        data = src.read(band_index, out_shape=(out_h, out_w), resampling=Resampling.nearest)
    return data


def generate_class_preview(class_raster, level_count, out_png, max_long_side=2048, palette=None, allow_upscale=False):
    """诊断分级图预览：可选自定义调色板（默认按级数取固定配色），无效像元透明。"""
    data = _downsampled_band(class_raster, 1, max_long_side, allow_upscale).astype(np.int32)
    valid = (data >= 1) & (data <= level_count)

    palette = list(palette) if palette and len(palette) >= level_count else build_class_legend(level_count)
    lut = np.zeros((256, 3), dtype=np.uint8)
    for index, color in enumerate(palette, start=1):
        lut[index] = _hex_to_rgb(color)

    rgb = lut[np.clip(data, 0, 255)]
    alpha = np.where(valid, 255, 0).astype(np.uint8)
    rgba = np.dstack([rgb, alpha])
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgba, mode="RGBA").save(out_png)
    return {"path": str(out_png)}


def generate_prescription_preview(prescription_raster, out_png, max_long_side=2048, allow_upscale=False):
    """处方图预览：按有效值范围渐变渲染，NaN 透明。返回图例范围。"""
    data = _downsampled_band(prescription_raster, 1, max_long_side, allow_upscale).astype(np.float32)
    valid = np.isfinite(data)
    if np.any(valid):
        low, high = np.percentile(data[valid], (1, 99))
        if high <= low:
            high = low + 1.0
    else:
        low, high = 0.0, 1.0

    normalized = np.clip((data - low) / max(high - low, 1e-9), 0.0, 1.0)
    stops_pos = np.array([pos for pos, _ in PRESCRIPTION_STOPS])
    stop_colors = np.stack([_hex_to_rgb(color) for _, color in PRESCRIPTION_STOPS]).astype(np.float32)

    rgb = np.zeros(data.shape + (3,), dtype=np.uint8)
    for ch in range(3):
        rgb[..., ch] = np.interp(normalized, stops_pos, stop_colors[:, ch]).astype(np.uint8)
    alpha = np.where(valid, 255, 0).astype(np.uint8)

    rgba = np.dstack([rgb, alpha])
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgba, mode="RGBA").save(out_png)
    return {"path": str(out_png), "min": float(low), "max": float(high)}


# --------------------------------------------------------------------------
# 图面注记渲染：把注记与图例模块绘制到预览图上（供嵌入/下载导出）
# --------------------------------------------------------------------------

def image_max_side(path):
    """返回图片的最长边像素数（用于判断预览图是否过小）。"""
    with Image.open(path) as im:
        return int(max(im.size))


_ANNOTATION_FONT_CANDIDATES = ("simsun.ttc", "msyh.ttc", "simhei.ttf")


def _load_cjk_font(size_px):
    """加载注记字体（宋体优先），失败时回退 PIL 默认字体。"""
    windir = os.environ.get("WINDIR", r"C:\Windows")
    for name in _ANNOTATION_FONT_CANDIDATES:
        path = Path(windir) / "Fonts" / name
        if path.is_file():
            try:
                return ImageFont.truetype(str(path), int(size_px))
            except OSError:
                continue
    return ImageFont.load_default()


def _fmt_num(v):
    """与前端 formatNum 一致的数值文本。"""
    v = float(v)
    if abs(v) >= 1000:
        return f"{v:.0f}"
    if abs(v) >= 1:
        return f"{v:.2f}"
    return f"{v:.3g}"


def _gradient_bar(stops, width_px, height_px):
    """按色带断点生成渐变条图片（与网页 CSS 渐变同源）。"""
    positions = np.array([float(p) for p, _ in stops], dtype="float64")
    colors = np.stack([_hex_to_rgb(c) for _, c in stops]).astype("float64")
    xs = np.linspace(0.0, 1.0, max(2, int(width_px)))
    row = np.stack([np.interp(xs, positions, colors[:, ch]) for ch in range(3)], axis=1)
    bar = np.repeat(row.reshape(1, row.shape[0], 3).astype(np.uint8), int(height_px), axis=0)
    return Image.fromarray(bar, mode="RGB")


def annotate_preview_png(src_path, out_path, view, annotations, legend):
    """把注记与图例模块绘制到预览图的白色边框区域内（不遮挡图面）。

    view: "class"（四周加白色边框，注记与长势等级图例放在边框内）
          或 "prescription"（加白色下边框，注记与推荐施肥量色温条放在下边框内）
    annotations: {"area": "77.6亩", "unit": "扬州大学", "date": "2026年9月9日"}
    legend: class   -> {"title": "长势等级", "items": [{"color": "#rrggbb", "label": "1级 最差"}, ...]}
            prescription -> {"title": "推荐施肥量", "min": 9.51, "max": 10.4,
                             "stops": [(pos, "#hex"), ...], "unit": "kg/亩"}
    字体为宋体；不绘制指南针/标题/比例尺。
    """
    src_path, out_path = Path(src_path), Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.open(src_path)
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        background = Image.new("RGB", img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[-1])
        img = background
    else:
        img = img.convert("RGB")

    measure = ImageDraw.Draw(img)
    img_w, img_h = img.size
    base = max(22, int(round(img_w / 46)))
    title_size = int(base * 1.3)
    font = _load_cjk_font(base)
    title_font = _load_cjk_font(title_size)
    line_h = int(base * 1.5)
    line_gap = int(base * 0.4)
    ink = (17, 17, 17)

    # ---- 排版测量：注记行与图例块 ----
    lines = [
        f"田块面积：{annotations.get('area', '')}",
        f"制图单位：{annotations.get('unit', '')}",
        f"制图时间：{annotations.get('date', '')}",
    ]
    ann_w = int(max(measure.textlength(t, font=font) for t in lines))
    ann_h = line_h * len(lines)

    title = legend.get("title", "")
    if view == "class":
        rows = [("swatch", item) for item in legend.get("items", [])]
    else:
        rows = [("bar", legend)]

    swatch_w = int(base * 1.7)
    swatch_h = int(base * 0.95)
    bar_w = int(base * 7.5)
    bar_h = int(base * 0.62)
    item_ws = []
    for kind, payload in rows:
        if kind == "swatch":
            w = swatch_w + int(base * 0.45) + measure.textlength(payload.get("label", ""), font=font)
        else:
            mn_text = _fmt_num(payload.get("min", 0))
            mx_text = f"{_fmt_num(payload.get('max', 0))} {payload.get('unit', '')}".strip()
            w = (measure.textlength(mn_text, font=font) + int(base * 0.4) + bar_w
                 + int(base * 0.4) + measure.textlength(mx_text, font=font))
        item_ws.append(w)
    title_w = measure.textlength(title, font=title_font)
    leg_w = int(max([title_w] + item_ws))
    title_h = int(title_size * 1.35)
    leg_h = title_h + line_gap + line_h * len(rows)

    # ---- 画布扩展：说明文字全部放入白色边框区域，不与图面重叠 ----
    # 两种图均加左右白边；诊断图加较高顶边（下边框容纳注记与图例），
    # 处方图加较矮顶边（说明文字主要位于下边框）
    side = int(base * 1.5)
    top = int(base * 1.2) if view == "class" else int(base * 0.8)
    bottom = max(ann_h, leg_h) + int(base * 1.2)
    canvas = Image.new("RGB", (img_w + side * 2, img_h + top + bottom), (255, 255, 255))
    canvas.paste(img, (side, top))
    draw = ImageDraw.Draw(canvas)

    # ---- 底部边框区域：左下注记行（与图面左缘对齐，底对齐） ----
    text_left = side + int(base * 0.35)
    text_bottom = canvas.height - int(base * 0.55)
    ann_top = text_bottom - ann_h
    for i, text in enumerate(lines):
        draw.text(
            (text_left, ann_top + i * line_h + (line_h - base) * 0.25),
            text, font=font, fill=ink,
        )

    # ---- 底部边框区域：右下图例（与图面右缘对齐，底对齐） ----
    lx = canvas.width - side - int(base * 0.35) - leg_w
    leg_top = text_bottom - leg_h
    draw.text((lx, leg_top), title, font=title_font, fill=ink)

    ry = leg_top + title_h + line_gap
    for kind, payload in rows:
        ty = ry + (line_h - base) * 0.25
        if kind == "swatch":
            sy = ry + (line_h - swatch_h) // 2
            draw.rectangle(
                [lx, sy, lx + swatch_w, sy + swatch_h],
                fill=tuple(int(v) for v in _hex_to_rgb(payload.get("color", "#ffffff"))),
                outline=(130, 130, 130),
            )
            draw.text((lx + swatch_w + int(base * 0.45), ty), payload.get("label", ""), font=font, fill=ink)
        else:
            mn_text = _fmt_num(payload.get("min", 0))
            mx_text = f"{_fmt_num(payload.get('max', 0))} {payload.get('unit', '')}".strip()
            mn_w = draw.textlength(mn_text, font=font)
            by = ry + (line_h - bar_h) // 2
            bar_x = lx + mn_w + int(base * 0.4)
            canvas.paste(
                _gradient_bar(payload.get("stops", PRESCRIPTION_STOPS), bar_w, bar_h),
                (int(bar_x), int(by)),
            )
            draw.text((lx, ty), mn_text, font=font, fill=ink)
            draw.text((int(bar_x) + bar_w + int(base * 0.4), ty), mx_text, font=font, fill=ink)
        ry += line_h

    canvas.save(out_path)
    return out_path