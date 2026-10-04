"""输入识别与校验。

核心修复：原项目仅上传一张 result.tif（RGBA 预览图）就把它当作
B1-B4 波段参与营养诊断（其 B3 实为蓝光、B4 实为 Alpha 通道，结果无意义）。
重构后正确流程为上传 5 张同经纬度影像，按文件名自动识别通道：

    *文件名含 RedEdge -> B1（红边）   *文件名含 Green  -> B2（绿光）
    *文件名含 Red     -> B3（红光）   *文件名含 NIR    -> B4（近红外）
    *不含任何波段关键字的 result 图  -> 整体预览图，不参与诊断计算

识别顺序必须先判 RedEdge 再判 Red（"RedEdge" 中包含 "Red"）。
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import rasterio

TIFF_SUFFIXES = {".tif", ".tiff"}
IGNORED_SUFFIXES = {".ovr", ".enp", ".aux", ".aux.xml", ".tfw", ".json", ".prj", ".vat", ".xml"}

ROLE_LABELS = {
    "B1": "B1 · 红边 RedEdge",
    "B2": "B2 · 绿光 Green",
    "B3": "B3 · 红光 Red",
    "B4": "B4 · 近红外 NIR",
    "preview": "整体预览图（不参与诊断）",
}

BAND_ROLES = ("B1", "B2", "B3", "B4")

@dataclass
class InputFile:
    path: Path
    name: str
    size: int
    source: str  # "upload" 或 "local"


@dataclass
class InputBundle:
    """一次会话中按角色归位后的输入文件集合。"""

    files: dict = field(default_factory=dict)  # role -> InputFile
    conflicts: list = field(default_factory=list)  # [(role, InputFile)] 冲突被忽略的文件

    def add(self, role, input_file):
        if role in self.files:
            self.conflicts.append((role, input_file))
            return False
        self.files[role] = input_file
        return True

    def remove(self, role):
        self.conflicts = [(r, f) for (r, f) in self.conflicts if r != role]
        return self.files.pop(role, None)

    def clear(self):
        self.files.clear()
        self.conflicts.clear()

    def missing_channels(self):
        return [role for role in BAND_ROLES if role not in self.files]

    def base_name(self):
        """输出文件名基准：优先取预览图文件名主干，否则取任一波段图主干。"""
        for role in ("preview",) + BAND_ROLES:
            info = self.files.get(role)
            if info is None:
                continue
            stem = Path(info.name).stem
            for suffix in ("_rededge", "_green", "_nir", "_red"):
                low = stem.lower()
                if low.endswith(suffix):
                    stem = stem[: -len(suffix)]
                    break
            return stem
        return "result"


def is_tiff(path):
    return path.suffix.lower() in TIFF_SUFFIXES


def recognize_role(filename, recognition_config):
    """按文件名识别角色。返回 "B1"/"B2"/"B3"/"B4"/"preview"。

    任何 tif/tiff 都给一个角色；非 tif 返回 None（调用方决定忽略或报错）。
    """
    path = Path(filename)
    if path.suffix.lower() not in TIFF_SUFFIXES:
        return None

    stem = path.stem.lower()
    order = ("B1", "B2", "B4", "B3")  # RedEdge 必须先于 Red 判定
    for role in order:
        for keyword in recognition_config.get(role, []):
            if keyword.lower() in stem:
                return role
    return "preview"


def scan_folder(folder, recognition_config):
    """扫描文件夹并按文件名归位（不拷贝文件，就地引用）。"""
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"文件夹不存在: {folder}")

    bundle = InputBundle()
    skipped = []
    for path in sorted(folder.iterdir()):
        if not path.is_file():
            continue
        if path.suffix.lower() in IGNORED_SUFFIXES:
            continue
        role = recognize_role(path.name, recognition_config)
        if role is None:
            skipped.append(path.name)
            continue
        bundle.add(role, InputFile(path=path, name=path.name, size=path.stat().st_size, source="local"))
    return bundle, skipped


def _crs_key(crs):
    if crs is None:
        return None
    try:
        return crs.to_wkt()
    except Exception:
        return str(crs)


def _transform_close(t1, t2):
    for a, b in zip(t1[:6], t2[:6]):
        if abs(a - b) > 1e-9 * max(1.0, abs(a), abs(b)):
            return False
    return True


def _describe(path):
    """只读元数据，描述一张输入图（不读像素，大文件安全）。"""
    with rasterio.open(path) as src:
        nodata = src.nodata
        if nodata is not None and not np.isfinite(nodata):
            nodata = "NaN"  # JSON 不能序列化 NaN 值（如 float 波段图的 nodata=nan）
        return {
            "width": int(src.width),
            "height": int(src.height),
            "count": int(src.count),
            "dtype": src.dtypes[0],
            "crs": _crs_key(src.crs),
            "nodata": nodata,
            "transform": [float(v) for v in src.transform[:6]],
            "pixel_size_x": float(src.transform.a),
            "pixel_size_y": float(src.transform.e),
        }


def validate_bundle(bundle):
    """校验输入完整性、波段数与同经纬度一致性。

    返回 (errors, warnings, report)。errors 非空时不可运行诊断。
    """
    errors = []
    warnings = []
    report = {"files": {}, "band_grid": None, "preview_overlay_ready": False}

    for role in ("preview",) + BAND_ROLES:
        info = bundle.files.get(role)
        if info is None:
            continue
        entry = {
            "name": info.name,
            "size": info.size,
            "source": info.source,
            "role_label": ROLE_LABELS[role],
        }
        try:
            entry["meta"] = _describe(info.path)
        except Exception as exc:  # noqa: BLE001 - 上传损坏/非GeoTIFF文件时给出可读错误
            errors.append(f"无法读取 {info.name}: {exc}")
            entry["meta"] = None
        report["files"][role] = entry

    if bundle.missing_channels():
        for role in bundle.missing_channels():
            errors.append(f"缺少必要通道 {ROLE_LABELS[role]}（文件名需含对应关键字）")
        return errors, warnings, report

    metas = {}
    for role in BAND_ROLES:
        info = bundle.files[role]
        meta = report["files"][role].get("meta")
        if meta is None:
            return errors, warnings, report
        if meta["count"] != 1:
            errors.append(
                f"{info.name} 有 {meta['count']} 个波段，通道图必须为单波段影像"
                f"（多波段合成图如 result.tif 只能作为预览图上传）"
            )
        metas[role] = meta

    if errors:
        return errors, warnings, report

    ref = metas["B2"]
    for role in ("B1", "B3", "B4"):
        meta = metas[role]
        info = bundle.files[role]
        if (meta["width"], meta["height"]) != (ref["width"], ref["height"]):
            errors.append(
                f"{info.name} 尺寸 {meta['width']}×{meta['height']} 与 "
                f"{bundle.files['B2'].name} 尺寸 {ref['width']}×{ref['height']} 不一致，"
                "四个通道图必须是同网格影像"
            )
        elif meta["crs"] != ref["crs"] or not _transform_close(meta["transform"], ref["transform"]):
            errors.append(
                f"{info.name} 与 {bundle.files['B2'].name} 的坐标参考或仿射变换不一致，"
                "不是同经纬度对齐的影像，无法参与同一诊断"
            )

    report["band_grid"] = {"width": ref["width"], "height": ref["height"]}

    preview_meta = (report["files"].get("preview") or {}).get("meta")
    if preview_meta is not None:
        if preview_meta["count"] not in (1, 3, 4):
            warnings.append(
                f"预览图 {bundle.files['preview'].name} 有 {preview_meta['count']} 个波段，"
                "预览显示可能不准确（仅影响显示，不影响诊断）"
            )
        report["preview_overlay_ready"] = (
            (preview_meta["width"], preview_meta["height"]) == (ref["width"], ref["height"])
            and preview_meta["crs"] == ref["crs"]
            and _transform_close(preview_meta["transform"], ref["transform"])
        )
        if not report["preview_overlay_ready"]:
            warnings.append(
                "预览图与波段图网格/分辨率不同（正常现象，DJI 拼接导出即如此），"
                "预览图仅作整体浏览，诊断结果将按波段图网格输出"
            )

    for role, conflict_file in bundle.conflicts:
        warnings.append(
            f"忽略重复文件 {conflict_file.name}（{ROLE_LABELS[role]} 已由 "
            f"{bundle.files[role].name if role in bundle.files else '其他文件'} 占用）"
        )

    return errors, warnings, report


def iter_band_windows(paths, height, width, block_size):
    """按 block_size 窗口流式读取 4 个通道，yield (window, bands, valid_mask)。

    bands 只含诊断所需的 B2/B3/B4（当前 8 个模型均只用这三路）；
    B1 参与校验但不读入内存，避免不必要的开销。
    valid_mask 为各文件 dataset_mask 的交集（含 NaN 处理）。
    """
    from rasterio.windows import Window

    with rasterio.open(paths["B2"]) as b2, \
            rasterio.open(paths["B3"]) as b3, \
            rasterio.open(paths["B4"]) as b4:
        for row_off in range(0, height, block_size):
            row_h = min(block_size, height - row_off)
            for col_off in range(0, width, block_size):
                col_w = min(block_size, width - col_off)
                window = Window(col_off, row_off, col_w, row_h)
                bands = {
                    "B2": b2.read(1, window=window).astype("float32"),
                    "B3": b3.read(1, window=window).astype("float32"),
                    "B4": b4.read(1, window=window).astype("float32"),
                }
                valid = b2.dataset_mask(window=window) > 0
                valid &= b3.dataset_mask(window=window) > 0
                valid &= b4.dataset_mask(window=window) > 0
                stacked = np.stack([bands["B2"], bands["B3"], bands["B4"]], axis=0)
                valid &= np.all(np.isfinite(stacked), axis=0)
                yield window, bands, valid
