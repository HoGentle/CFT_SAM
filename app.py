"""施肥处方图生成系统（重构版）本地 Web 服务。

启动后浏览器访问 http://127.0.0.1:8765 （自动打开）。

输入约定（修复原项目仅上传一张 result.tif 即诊断的错误）：
    上传/选择 5 张同经纬度影像，按文件名自动识别：
      *含 RedEdge -> B1 红边    *含 Green -> B2 绿光
      *含 Red     -> B3 红光    *含 NIR   -> B4 近红外
      *不含任何波段关键字的 result 图 -> 整体预览图（不参与诊断）
    通道图 B1-B4 不在界面显示，仅用于计算；界面只展示整体预览图与成果图。
"""

import json
import math
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid
import webbrowser
from pathlib import Path

from flask import Flask, abort, jsonify, request, send_file, send_from_directory

# 先导入 core（其 __init__ 会在 rasterio 导入前修正 Windows 下的 proj.db 冲突）
from core import dj2pr as dj2pr_core
from core import preview as preview_core
from core import stats as stats_core
from core.diagnose import STAGE_NAME_SUFFIX, make_unique_output_path, run_diagnosis
from core.merge import merge_group_rasters
from core.inputs import (
    BAND_ROLES,
    ROLE_LABELS,
    InputBundle,
    InputFile,
    recognize_role,
    scan_folder,
    validate_bundle,
)
from core.jobs import JobManager
from core.prescription import resolve_mapping, write_prescription_raster, write_resampled_prescription
from core.roi import polygons_lonlat_to_pixel

import rasterio

BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config.json"
WORKSPACE = BASE_DIR / "workspace"
UPLOAD_TMP_DIR = WORKSPACE / "uploads"
CURRENT_DIR = WORKSPACE / "current"
PREVIEW_DIR = WORKSPACE / "previews"
OUTPUTS_DIR = WORKSPACE / "outputs"
DJ2PR_DIR = WORKSPACE / "dj2pr"

with CONFIG_PATH.open("r", encoding="utf-8") as _file:
    CONFIG = json.load(_file)

app = Flask(__name__, static_folder=str(BASE_DIR / "static"), static_url_path="/static")
app.config["MAX_CONTENT_LENGTH"] = None
app.json.ensure_ascii = False

_job_manager = JobManager()
_state_lock = threading.Lock()


class SessionState:
    def __init__(self):
        self.bundle = InputBundle()
        self.validation = None
        self.band_paths = None
        self.input_preview = None
        self.last_result = None

    def snapshot_paths(self):
        with _state_lock:
            return {
                role: Path(info.path)
                for role, info in self.bundle.files.items()
            }

    def reset(self):
        with _state_lock:
            self.bundle.clear()
            self.validation = None
            self.band_paths = None
            self.input_preview = None
            self.last_result = None


SESSION = SessionState()


def ensure_dirs():
    for path in (UPLOAD_TMP_DIR, CURRENT_DIR, PREVIEW_DIR, OUTPUTS_DIR, DJ2PR_DIR):
        path.mkdir(parents=True, exist_ok=True)


def sanitize_name(name):
    name = os.path.basename(str(name or "")).strip()
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name)
    return name or "unnamed.tif"


def _is_lonlat(lon, lat):
    """宽松的经纬度合法性检查（只拒绝明显非法值）。"""
    try:
        lon, lat = float(lon), float(lat)
    except (TypeError, ValueError):
        return False
    return -180.0 <= lon <= 180.0 and -90.0 <= lat <= 90.0


def file_payload(info):
    return {"name": info.name, "size": info.size, "source": info.source}


def session_payload():
    with _state_lock:
        files = {}
        for role in ("preview",) + BAND_ROLES:
            info = SESSION.bundle.files.get(role)
            if info is not None:
                entry = file_payload(info)
                entry["role"] = role
                entry["role_label"] = ROLE_LABELS[role]
                files[role] = entry
        conflicts = [
            {**file_payload(info), "role": role, "role_label": ROLE_LABELS[role]}
            for role, info in SESSION.bundle.conflicts
        ]
        validation = SESSION.validation
        input_preview = SESSION.input_preview
    return {
        "files": files,
        "conflicts": conflicts,
        "missing": [ROLE_LABELS[r] for r in (set(BAND_ROLES) - set(files))],
        "validation": validation,
        "has_input_preview": input_preview is not None,
        "georef": input_preview.get("georef") if input_preview else None,
        "input_area": input_preview.get("area") if input_preview else None,
        "busy": _job_manager.is_running(),
    }


def _attach_input_area(info, band_paths):
    """给整体预览信息附带影像有效范围面积（整体预览图下方展示）。"""
    try:
        info["area"] = stats_core.raster_valid_area(band_paths["B2"])
    except Exception as exc:  # noqa: BLE001 - 面积统计失败不阻塞预览
        app.logger.warning("影像面积统计失败: %s", exc)
    return info


def start_preview_job():
    """校验通过后生成输入整体预览（后台任务）。"""
    paths = SESSION.snapshot_paths()
    preview_tif = paths.get("preview")
    band_paths = {role: paths[role] for role in ("B2", "B3", "B4") if role in paths}
    if not band_paths:
        return

    def runner(job):
        job.progress("preview", 0.05, "正在生成整体预览图...")
        info = preview_core.generate_input_preview(
            preview_tif,
            band_paths,
            PREVIEW_DIR / "input_preview.jpg",
            max_long_side=int(CONFIG["preview"]["max_pixels_long_side"]),
            jpeg_quality=int(CONFIG["preview"]["jpeg_quality"]),
        )
        info = _attach_input_area(info, band_paths)
        with _state_lock:
            SESSION.input_preview = info
        job.log(f"预览图已生成（{info['mode']}）")
        job.finish(result={"input_preview": info})

    job, error = _job_manager.start(
        "预览生成",
        [("preview", "生成输入预览", 1.0)],
        runner,
    )
    if error:
        app.logger.warning("预览任务未能启动: %s", error)


# --------------------------------------------------------------------------
# 基础页面与配置
# --------------------------------------------------------------------------

@app.get("/")
def index():
    return send_from_directory(str(BASE_DIR / "static"), "index.html")


@app.get("/api/config")
def api_config():
    stages = []
    for stage_id in sorted(CONFIG["stages"], key=int):
        stage = CONFIG["stages"][stage_id]
        stages.append({
            "id": stage_id,
            "name": stage["name"],
            "diagnosis_name": stage["diagnosis_name"],
            "index_name": stage["index_name"],
            "index_formula": stage["index_formula"],
            "diagnosis_formula": stage["diagnosis_formula"],
            "level_count": len(stage["thresholds"]),
            "thresholds": stage["thresholds"],
        })
    return jsonify({
        "stages": stages,
        "prescription": CONFIG["prescription"],
        "resampling": CONFIG["resampling"],
        "round_digits": CONFIG["defaults"].get("round_digits", 2),
        "unit": "kg/mu",
        "recognition_note": CONFIG["recognition"]["note"],
        "role_labels": ROLE_LABELS,
    })


@app.get("/api/session")
def api_session():
    return jsonify(session_payload())


# --------------------------------------------------------------------------
# 文件上传（分块）与文件夹扫描
# --------------------------------------------------------------------------

_UPLOADS = {}  # upload_id -> {"name","size","path"}


@app.post("/api/upload/start")
def api_upload_start():
    data = request.get_json(force=True, silent=True) or {}
    name = sanitize_name(data.get("name"))
    try:
        size = int(data.get("size") or 0)
    except (TypeError, ValueError):
        size = 0
    if size <= 0:
        return jsonify({"error": f"文件大小无效: {name}"}), 400

    upload_id = uuid.uuid4().hex
    part_path = UPLOAD_TMP_DIR / f"{upload_id}.part"
    part_path.write_bytes(b"")
    _UPLOADS[upload_id] = {"name": name, "size": size, "path": part_path}
    return jsonify({"upload_id": upload_id, "chunk_size_hint": 16 * 1024 * 1024})


@app.post("/api/upload/chunk/<upload_id>")
def api_upload_chunk(upload_id):
    meta = _UPLOADS.get(upload_id)
    if meta is None:
        return jsonify({"error": "上传会话不存在或已过期"}), 404
    try:
        offset = int(request.args.get("offset", 0))
    except ValueError:
        offset = 0

    part_path = Path(meta["path"])
    with part_path.open("r+b") as handle:
        handle.seek(0, os.SEEK_END)
        current = handle.tell()
        if current != offset:
            return jsonify({"error": f"分块偏移不匹配（服务端 {current} != 客户端 {offset}）"}), 409
        while True:
            chunk = request.stream.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
    return jsonify({"received": part_path.stat().st_size})


@app.post("/api/upload/finish")
def api_upload_finish():
    data = request.get_json(force=True, silent=True) or {}
    upload_id = data.get("upload_id")
    meta = _UPLOADS.pop(upload_id, None)
    if meta is None:
        return jsonify({"error": "上传会话不存在或已完成"}), 404

    part_path = Path(meta["path"])
    actual = part_path.stat().st_size
    if actual != meta["size"]:
        part_path.unlink(missing_ok=True)
        return jsonify({"error": f"文件不完整（{actual}/{meta['size']} 字节），请重新上传"}), 400

    name = meta["name"]
    role = recognize_role(name, CONFIG["recognition"])
    if role is None:
        part_path.unlink(missing_ok=True)
        return jsonify({"error": f"{name} 不是 tif/tiff 影像，已忽略"}), 400

    target = CURRENT_DIR / name
    if target.exists():
        target.unlink()
    shutil.move(str(part_path), str(target))

    info = InputFile(path=target, name=name, size=actual, source="upload")
    with _state_lock:
        replaced = SESSION.bundle.files.get(role)
        if replaced is not None and replaced.source == "upload":
            Path(replaced.path).unlink(missing_ok=True)
        SESSION.bundle.files.pop(role, None)
        SESSION.bundle.conflicts = [(r, f) for (r, f) in SESSION.bundle.conflicts if r != role]
        SESSION.bundle.files[role] = info
        SESSION.validation = None

    return jsonify({
        "role": role,
        "role_label": ROLE_LABELS[role],
        "name": name,
        "size": actual,
        "session": session_payload(),
    })


@app.post("/api/files/remove")
def api_files_remove():
    data = request.get_json(force=True, silent=True) or {}
    role = data.get("role")
    with _state_lock:
        info = SESSION.bundle.remove(role)
        SESSION.validation = None
        if info is not None and info.source == "upload":
            Path(info.path).unlink(missing_ok=True)
    return jsonify(session_payload())


@app.post("/api/local/folder")
def api_local_folder():
    data = request.get_json(force=True, silent=True) or {}
    folder = str(data.get("path") or "").strip().strip('"')
    try:
        bundle, skipped = scan_folder(folder, CONFIG["recognition"])
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 400

    with _state_lock:
        SESSION.bundle = bundle
        SESSION.validation = None
    payload = session_payload()
    payload["skipped"] = skipped
    payload["folder"] = str(Path(folder).resolve())
    return jsonify(payload)


# --------------------------------------------------------------------------
# 校验
# --------------------------------------------------------------------------

@app.post("/api/validate")
def api_validate():
    with _state_lock:
        bundle = SESSION.bundle
    if not bundle.files:
        return jsonify({"error": "尚未添加任何影像文件"}), 400

    # 剔除磁盘上已不存在的登记文件（上传后被清理/移动/重置的情形），
    # 避免校验时逐个抛出 rasterio 的 "No such file or directory" 原始错误。
    missing_on_disk = []
    with _state_lock:
        for role in list(bundle.files.keys()):
            info = bundle.files[role]
            if not Path(info.path).is_file():
                missing_on_disk.append(f"{ROLE_LABELS[role]}（{info.name}）")
                bundle.remove(role)
    if missing_on_disk:
        if not bundle.files:
            return jsonify({
                "error": "登记的影像文件已不在磁盘上（workspace 目录被清空或文件被移动），"
                         "请重新上传: " + "、".join(missing_on_disk)
            }), 400

    errors, warnings, report = validate_bundle(bundle)
    if missing_on_disk:
        warnings.insert(0, "已从会话移除磁盘上不存在的文件: " + "、".join(missing_on_disk))
    validation = {"errors": errors, "warnings": warnings, "report": report}
    with _state_lock:
        SESSION.validation = validation
        SESSION.band_paths = (
            {role: Path(bundle.files[role].path) for role in BAND_ROLES}
            if not errors else None
        )

    response = {**validation, "session": session_payload()}
    if not errors:
        start_preview_job()
    return jsonify(response)


# --------------------------------------------------------------------------
# 感兴趣区域面积（编辑完成后即时展示）
# --------------------------------------------------------------------------

@app.post("/api/roi/area")
def api_roi_area():
    data = request.get_json(force=True, silent=True) or {}
    regions = data.get("regions") or []
    hulls = []
    for idx, region in enumerate(regions):
        hull = (region or {}).get("hull") or []
        try:
            points = [[float(lon), float(lat)] for lon, lat in hull]
        except (TypeError, ValueError):
            return jsonify({"error": f"第 {idx + 1} 个区域顶点坐标无效"}), 400
        if len(points) < 3:
            return jsonify({"error": f"第 {idx + 1} 个区域顶点不足 3 个"}), 400
        hulls.append(points)

    try:
        area = stats_core.lonlat_polygons_area(hulls)
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"面积计算失败: {exc}"}), 400
    return jsonify({"region_count": len(hulls), **area})


def _class_legend_items(result):
    """诊断图图例数据。

    分组结果中，各分组的等级数量与每级颜色、长势标签完全一致时，图例
    融合为单一列表（不带组名前缀，如「1级 最差」）；存在差异时按组
    前缀逐组列出（如「组1·1级 最差」），便于区分。
    """
    if result.get("grouped") and result.get("group_results"):
        groups = result["group_results"]
        base = groups[0].get("level_stats", [])
        mergeable = bool(base) and all(len(g.get("level_stats", [])) == len(base) for g in groups)
        if mergeable:
            for g in groups:
                for a, b in zip(base, g["level_stats"]):
                    if a["color"] != b["color"] or (a.get("growth_label") or "") != (b.get("growth_label") or ""):
                        mergeable = False
                        break
                if not mergeable:
                    break
        if mergeable:
            return [
                {"color": s["color"], "label": f"{s['level']}级 {s.get('growth_label') or ''}".strip()}
                for s in base
            ]
        items = []
        for g in groups:
            for s in g.get("level_stats", []):
                items.append({
                    "color": s["color"],
                    "label": f"{g['name']}·{s['level']}级 {s.get('growth_label') or ''}".strip(),
                })
        return items
    return [
        {"color": s["color"], "label": f"{s['level']}级 {s.get('growth_label') or ''}".strip()}
        for s in result.get("level_stats", [])
    ]


@app.post("/api/preview/annotate")
def api_preview_annotate():
    """把注记与图例模块绘制到当前预览图上（嵌入样式），返回预览/下载地址。"""
    data = request.get_json(force=True, silent=True) or {}
    view = data.get("view")
    if view not in ("class", "prescription"):
        return jsonify({"error": "view 必须是 class 或 prescription"}), 400

    with _state_lock:
        result = SESSION.last_result
    if not result:
        return jsonify({"error": "尚无生成结果，请先运行诊断与处方生成"}), 400

    src = PREVIEW_DIR / ("class_preview.png" if view == "class" else "prescription_preview.png")
    if not src.is_file():
        return jsonify({"error": "预览图尚未生成，请重新运行生成"}), 400

    # 预览图过小（源影像网格较小）时，从成果栅格重新渲染放大的嵌入底图
    if preview_core.image_max_side(src) < 1200:
        if view == "prescription":
            final_raster = result.get("final_raster")
            if final_raster and Path(final_raster).is_file():
                src = PREVIEW_DIR / "prescription_preview_embed_base.png"
                try:
                    preview_core.generate_prescription_preview(
                        final_raster, src, max_long_side=2048, allow_upscale=True
                    )
                except Exception as exc:  # noqa: BLE001
                    return jsonify({"error": f"嵌入底图渲染失败: {exc}"}), 500
        else:
            class_raster = result.get("class_raster")
            if class_raster and Path(class_raster).is_file():
                if result.get("grouped") and result.get("group_results"):
                    level_count = max(g.get("level_count", 0) for g in result["group_results"])
                    palette = preview_core.build_class_legend(level_count)
                    for g in result["group_results"]:
                        for s in g.get("level_stats", []):
                            if 0 < s["level"] <= level_count:
                                palette[s["level"] - 1] = s["color"]
                else:
                    palette = [s["color"] for s in result.get("level_stats", [])]
                    level_count = max(len(palette), 2)
                if level_count >= 2:
                    src = PREVIEW_DIR / "class_preview_embed_base.png"
                    try:
                        preview_core.generate_class_preview(
                            class_raster, level_count, src,
                            max_long_side=2048, palette=palette, allow_upscale=True,
                        )
                    except Exception as exc:  # noqa: BLE001
                        return jsonify({"error": f"嵌入底图渲染失败: {exc}"}), 500

    # 注记文本：前端传当前显示值；缺省时服务端按结果补默认
    area = str(data.get("area") or "").strip()
    if not area:
        stats = result.get("prescription_stats") or {}
        roi_area = result.get("roi_area") or {}
        if stats.get("area_mu") is not None:
            area = f"{round(stats['area_mu'] * 100) / 100}亩"
        elif roi_area.get("area_mu"):
            area = f"{round(roi_area['area_mu'] * 100) / 100}亩"
    unit = str(data.get("unit") or "").strip() or "扬州大学"
    date_text = str(data.get("date") or "").strip()
    if not date_text:
        m = re.match(r"^(\d{4})(\d{2})(\d{2})", str(result.get("run_id") or ""))
        date_text = f"{int(m[1])}年{int(m[2])}月{int(m[3])}日" if m else time.strftime("%Y年%m月%d日")

    if view == "class":
        items = _class_legend_items(result)
        if not items:
            return jsonify({"error": "诊断图分级统计尚未生成"}), 400
        legend = {"title": "长势等级", "items": items}
    else:
        rng = result.get("prescription_preview_range") or {}
        if rng.get("min") is None:
            return jsonify({"error": "处方图预览范围尚未生成"}), 400
        legend = {
            "title": "推荐施肥量",
            "min": rng["min"],
            "max": rng["max"],
            "unit": "kg/亩",
            "stops": preview_core.PRESCRIPTION_STOPS,
        }

    out = PREVIEW_DIR / f"{view}_preview_annotated.png"
    try:
        preview_core.annotate_preview_png(
            src, out, view, {"area": area, "unit": unit, "date": date_text}, legend
        )
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"注记渲染失败: {exc}"}), 500

    return jsonify({
        "view": view,
        "url": f"/previews/{out.name}",
        "download_url": f"/api/download?path={out}",
        "file": str(out),
    })


# --------------------------------------------------------------------------
# 运行诊断 + 处方图
# --------------------------------------------------------------------------

@app.post("/api/run")
def api_run():
    if _job_manager.is_running():
        return jsonify({"error": "已有任务正在运行，请等待完成。"}), 409

    with _state_lock:
        validation = SESSION.validation
        band_paths = dict(SESSION.band_paths) if SESSION.band_paths else None
    if band_paths is None:
        return jsonify({"error": "请先完成数据校验（当前输入不可用或未校验）"}), 400

    data = request.get_json(force=True, silent=True) or {}
    stage_id = str(data.get("stage") or "")
    if stage_id not in CONFIG["stages"]:
        return jsonify({"error": f"无效的生育期选项: {stage_id}"}), 400
    stage_config = dict(CONFIG["stages"][stage_id])

    # ---- 自定义分级：占比（累计分位断点）与长势标签可由前端覆盖 ----
    level_thresholds = data.get("level_thresholds") or []
    if level_thresholds:
        if not 2 <= len(level_thresholds) <= 20:
            return jsonify({"error": "分级数量必须在 2–20 级之间"}), 400
        cleaned_thresholds = []
        previous_q = 0.0
        for idx, item in enumerate(level_thresholds):
            try:
                level = int(item.get("level", idx + 1))
                quantile = float(item.get("quantile"))
            except (TypeError, ValueError):
                return jsonify({"error": f"第 {idx + 1} 级的累计占比无效"}), 400
            if not 0.0 < quantile <= 1.0:
                return jsonify({"error": f"第 {idx + 1} 级的累计占比必须是 0–100 之间的数值"}), 400
            if quantile <= previous_q:
                return jsonify({"error": "各级累计占比必须从小到大严格递增"}), 400
            label = str(item.get("growth_label") or "").strip()
            cleaned_thresholds.append({
                "level": level,
                "quantile": quantile,
                "growth_label": label,
                "color": str(item.get("color") or "").strip(),
            })
            previous_q = quantile
        if cleaned_thresholds[-1]["quantile"] != 1.0:
            return jsonify({"error": "最后一级的累计占比必须为 100%"}), 400
        stage_config["thresholds"] = cleaned_thresholds

    run_ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = OUTPUTS_DIR / run_ts
    out_dir.mkdir(parents=True, exist_ok=True)
    base_name = SESSION.bundle.base_name()

    # ---- 生成范围：全图 或 感兴趣区域（凸包经纬度 -> 诊断网格像素多边形） ----
    # 影像 CRS 同时供 ROI 换算与地面栅格重采样使用
    with rasterio.open(band_paths["B2"]) as band_src:
        band_transform = band_src.transform
        band_crs = band_src.crs
    if band_crs is None:
        return jsonify({
            "error": "影像缺少坐标系（CRS）信息，无法按地理地面距离执行栅格重采样，"
                     "请使用带地理参考的影像。"
        }), 400
    make_mode = data.get("make_mode", "full")
    roi_polygons = None
    roi_meta = None
    if make_mode == "roi":
        roi_regions = data.get("roi_regions") or []
        cleaned_regions = []
        for idx, region in enumerate(roi_regions):
            hull = (region or {}).get("hull") or []
            points = (region or {}).get("points") or []
            if len(hull) < 3:
                return jsonify({"error": f"第 {idx + 1} 个感兴趣区域凸包顶点不足 3 个"}), 400
            try:
                cleaned = [[[float(lon), float(lat)] for lon, lat in hull]]
            except (TypeError, ValueError):
                return jsonify({"error": f"第 {idx + 1} 个感兴趣区域顶点坐标无效"}), 400
            cleaned_regions.append({
                "hull": cleaned[0],
                "points": [
                    [float(lon), float(lat)]
                    for lon, lat in (points or [])
                    if _is_lonlat(lon, lat)
                ],
            })
        if not cleaned_regions:
            return jsonify({"error": "感兴趣区域为空，请先在预览图上标点"}), 400

        try:
            roi_polygons = polygons_lonlat_to_pixel(
                [region["hull"] for region in cleaned_regions], band_transform
            )
        except Exception as exc:  # noqa: BLE001
            return jsonify({"error": f"感兴趣区域坐标换算失败: {exc}"}), 400
        try:
            roi_area = stats_core.lonlat_polygons_area([r["hull"] for r in cleaned_regions])
        except Exception as exc:  # noqa: BLE001
            return jsonify({"error": f"感兴趣区域面积计算失败: {exc}"}), 400
        roi_meta = {
            "mode": "roi",
            "region_count": len(cleaned_regions),
            "regions": cleaned_regions,
            "area": roi_area,
            "note": "hull 为最小凸包顶点经纬度；points 为全部标点经纬度（标注顺序）",
        }
        job_log_roi = f"感兴趣区域制作：{len(cleaned_regions)} 个凸包区域（{roi_area['area_mu']:.2f} 亩）"
    else:
        roi_area = None
        job_log_roi = "全图制作"

    run_params = {
        "mode": data.get("mapping_mode", "formula"),
        "formula": data.get("formula") or {},
        "manual": data.get("manual_mapping") or {},
    }
    round_digits = int(CONFIG["defaults"].get("round_digits", 2))
    levels = [int(t["level"]) for t in stage_config["thresholds"]]
    try:
        mapping, mapping_mode = resolve_mapping(
            levels, run_params, CONFIG["prescription"], round_digits
        )
    except (TypeError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 400

    # ---- ROI 分组（可选）：每组独立的诊断分级阈值与施肥映射 ----
    # 未被任何分组勾选的区域归入「未分组」隐式分组，沿用上方全局设置。
    group_defs = []
    if make_mode == "roi":
        for gi, grp in enumerate(data.get("roi_groups") or [], 1):
            grp = grp or {}
            name = str(grp.get("name") or f"组{gi}").strip() or f"组{gi}"
            try:
                idxs = sorted({int(i) for i in (grp.get("region_indexes") or [])})
            except (TypeError, ValueError):
                return jsonify({"error": f"分组「{name}」的区域编号无效"}), 400
            if not idxs:
                return jsonify({"error": f"分组「{name}」未勾选任何区域"}), 400
            bad = [i for i in idxs if not 0 <= i < len(cleaned_regions)]
            if bad:
                return jsonify({"error": f"分组「{name}」的区域编号超出范围: {', '.join(str(i + 1) for i in bad)}"}), 400

            try:
                level_count = int(grp.get("level_count"))
            except (TypeError, ValueError):
                return jsonify({"error": f"分组「{name}」的分级数量无效"}), 400
            if not 2 <= level_count <= 20:
                return jsonify({"error": f"分组「{name}」的分级数量必须在 2–20 级之间"}), 400

            cleaned_g = []
            previous_q = 0.0
            for ti, item in enumerate(grp.get("thresholds") or []):
                item = item or {}
                try:
                    level = int(item.get("level", ti + 1))
                    quantile = float(item.get("quantile"))
                except (TypeError, ValueError):
                    return jsonify({"error": f"分组「{name}」第 {ti + 1} 级的累计占比无效"}), 400
                if not 0.0 < quantile <= 1.0:
                    return jsonify({"error": f"分组「{name}」第 {ti + 1} 级的累计占比必须是 0–100 之间的数值"}), 400
                if quantile <= previous_q:
                    return jsonify({"error": f"分组「{name}」各级累计占比必须从小到大严格递增"}), 400
                cleaned_g.append({
                    "level": level,
                    "quantile": quantile,
                    "growth_label": str(item.get("growth_label") or "").strip(),
                    "color": str(item.get("color") or "").strip(),
                })
                previous_q = quantile
            if len(cleaned_g) != level_count or cleaned_g[-1]["quantile"] != 1.0:
                return jsonify({"error": f"分组「{name}」分级数量与累计占比不一致（最后一级须为 100%）"}), 400

            g_mode = grp.get("mapping_mode") or "formula"
            if g_mode not in ("formula", "manual"):
                return jsonify({"error": f"分组「{name}」的施肥映射方式无效"}), 400
            try:
                g_mapping, g_mapping_mode = resolve_mapping(
                    [t["level"] for t in cleaned_g],
                    {"mode": g_mode, "formula": grp.get("formula") or {}, "manual": grp.get("manual_mapping") or {}},
                    CONFIG["prescription"], round_digits,
                )
            except (TypeError, ValueError) as exc:
                return jsonify({"error": f"分组「{name}」: {exc}"}), 400

            group_defs.append({
                "name": name,
                "region_indexes": idxs,
                "level_count": level_count,
                "thresholds": cleaned_g,
                "mapping": g_mapping,
                "mapping_mode": g_mapping_mode,
            })

        if group_defs:
            assigned = [i for g in group_defs for i in g["region_indexes"]]
            dup = sorted({i for i in assigned if assigned.count(i) > 1})
            if dup:
                return jsonify({"error": "区域被重复分组: " + "、".join(f"区域{i + 1}" for i in dup)}), 400
            unassigned = [i for i in range(len(cleaned_regions)) if i not in set(assigned)]
            if unassigned:
                group_defs.append({
                    "name": "未分组",
                    "region_indexes": unassigned,
                    "level_count": len(stage_config["thresholds"]),
                    "thresholds": stage_config["thresholds"],
                    "mapping": mapping,
                    "mapping_mode": mapping_mode,
                })
            job_log_groups = f"分组制作：{'、'.join(g['name'] for g in group_defs)}"
        else:
            job_log_groups = None
    else:
        job_log_groups = None

    # ---- 重采样栅格设置：栅格宽度由前端传入（最低 1m），缺省用服务端配置；方法固定取配置 ----
    resample_cfg = CONFIG["resampling"]
    resample_method = str(resample_cfg.get("method", "average"))
    raw_target = (data.get("resampling") or {}).get("target_ground_resolution_m")
    if raw_target is None:
        resample_target_m = float(resample_cfg["target_ground_resolution_m"])
    else:
        try:
            resample_target_m = float(raw_target)
        except (TypeError, ValueError):
            return jsonify({"error": "栅格宽度必须是数值"}), 400
        if not math.isfinite(resample_target_m) or resample_target_m < 1.0:
            return jsonify({"error": "栅格宽度最低为 1m"}), 400
    resample_phase_label = f"{resample_target_m:g}m 地面栅格重采样"

    phase_weights = [
        ("preview", "输入预览", 5),
        ("pass1", "统计诊断值样本", 20),
        ("pass2", "写出诊断图", 30),
        ("prescription", "写出处方图", 30),
        ("resample", resample_phase_label, 5),
        ("result_preview", "生成结果预览", 10),
    ]

    def diag_progress(fraction, message):
        if fraction <= 0.5:
            job.progress("pass1", fraction / 0.5, message)
        else:
            job.progress("pass2", (fraction - 0.5) / 0.5, message)

    def runner(job):
        started = time.time()
        job.log(f"生育期: {stage_config['name']}（{stage_config['index_name']} 模型）")
        job.log(job_log_roi)
        if group_defs:
            job.log("施肥映射: 按分组分别设置（各组映射见下方日志）")
        else:
            job.log(f"施肥映射（{mapping_mode}）: {mapping}")
        job.log(
            f"重采样: 聚合至 {resample_target_m:g}m × {resample_target_m:g}m 地面栅格"
            f"（{resample_method}，基于地理地面距离）"
        )

        if SESSION.input_preview is None:
            job.progress("preview", 0.1, "正在生成整体预览图...")
            preview_tif = SESSION.snapshot_paths().get("preview")
            info = preview_core.generate_input_preview(
                preview_tif,
                {role: band_paths[role] for role in ("B2", "B3", "B4")},
                PREVIEW_DIR / "input_preview.jpg",
                max_long_side=int(CONFIG["preview"]["max_pixels_long_side"]),
                jpeg_quality=int(CONFIG["preview"]["jpeg_quality"]),
            )
            info = _attach_input_area(info, band_paths)
            with _state_lock:
                SESSION.input_preview = info
        job.progress("preview", 1.0, "预览图就绪")

        band_paths_diag = {role: band_paths[role] for role in ("B1", "B2", "B3", "B4")}

        if group_defs:
            # ---- 分组流水线：每组独立诊断与处方，最后合并为单一输出 ----
            G = len(group_defs)
            group_results = []
            group_meta_list = []
            group_class_paths = []
            group_value_paths = []
            group_presc_paths = []
            value_min = None
            value_max = None
            for gi, gdef in enumerate(group_defs):
                gname = gdef["name"]
                g_polygons = [roi_polygons[i] for i in gdef["region_indexes"]]
                g_regions = [cleaned_regions[i] for i in gdef["region_indexes"]]
                try:
                    g_area = stats_core.lonlat_polygons_area([r["hull"] for r in g_regions])
                except Exception:  # noqa: BLE001
                    g_area = {"area_m2": 0.0, "area_mu": 0.0}
                g_dir = out_dir / f"group_{gi + 1}"
                job.log(
                    f"分组「{gname}」（区域 {'、'.join(str(i + 1) for i in gdef['region_indexes'])}）："
                    f"施肥映射（{gdef['mapping_mode']}）: {gdef['mapping']}"
                )

                def group_diag_progress(fraction, message, _gi=gi, _name=gname):
                    tag = f"[{_name}] {message}"
                    if fraction <= 0.5:
                        job.progress("pass1", (_gi + fraction / 0.5) / G, tag)
                    else:
                        job.progress("pass2", (_gi + (fraction - 0.5) / 0.5) / G, tag)

                g_stage = dict(stage_config)
                g_stage["thresholds"] = gdef["thresholds"]
                diag = run_diagnosis(
                    band_paths=band_paths_diag,
                    stage_config=g_stage,
                    output_dir=g_dir,
                    defaults=CONFIG["defaults"],
                    progress=diag_progress,
                    base_name=base_name,
                    roi_polygons=g_polygons,
                    roi_meta={
                        "mode": "roi",
                        "group": gname,
                        "region_numbers": [i + 1 for i in gdef["region_indexes"]],
                        "region_count": len(g_regions),
                        "regions": g_regions,
                        "area": g_area,
                    },
                )

                def group_pres_progress(fraction, message, _gi=gi, _name=gname):
                    job.progress("prescription", (_gi + fraction) / G, f"[{_name}] {message}")

                presc = write_prescription_raster(
                    class_raster_path=diag["class_raster"],
                    output_path=g_dir / f"{base_name}_prescription.tif",
                    mapping=gdef["mapping"],
                    defaults=CONFIG["defaults"],
                    progress=group_pres_progress,
                )
                if presc["unmatched_class_counts"]:
                    job.log(f"警告: 分组「{gname}」存在未映射的分级值 {presc['unmatched_class_counts']}")

                vr = diag.get("value_range") or {}
                if vr.get("min") is not None:
                    value_min = vr["min"] if value_min is None else min(value_min, vr["min"])
                if vr.get("max") is not None:
                    value_max = vr["max"] if value_max is None else max(value_max, vr["max"])

                g_palette = preview_core.build_class_legend(gdef["level_count"])
                g_user = [t.get("color") for t in gdef["thresholds"]]
                if all(isinstance(c, str) and len(c) == 7 and c.startswith("#") for c in g_user):
                    g_palette = g_user
                total_g = diag["valid_pixel_count"] or 1
                g_level_stats = []
                for threshold, resolved in zip(gdef["thresholds"], diag["resolved_thresholds"]):
                    level = int(threshold["level"])
                    count_g = diag["class_counts"].get(str(level), 0)
                    g_level_stats.append({
                        "level": level,
                        "color": g_palette[level - 1] if 0 < level <= len(g_palette) else g_palette[-1],
                        "growth_label": threshold.get("growth_label", ""),
                        "min": resolved["min"],
                        "max": resolved["max"],
                        "fertilizer": gdef["mapping"].get(str(level)),
                        "count": count_g,
                        "percent": round(count_g / total_g * 100.0, 2),
                    })

                group_results.append({
                    "name": gname,
                    "region_numbers": [i + 1 for i in gdef["region_indexes"]],
                    "area": g_area,
                    "mapping_mode": gdef["mapping_mode"],
                    "mapping": gdef["mapping"],
                    "level_count": gdef["level_count"],
                    "level_stats": g_level_stats,
                    "valid_pixel_count": diag["valid_pixel_count"],
                    "value_range": diag.get("value_range"),
                    "output_dir": str(g_dir),
                })
                group_meta_list.append({
                    "name": gname,
                    "region_numbers": [i + 1 for i in gdef["region_indexes"]],
                    "area": g_area,
                    "mapping_mode": gdef["mapping_mode"],
                    "mapping": gdef["mapping"],
                    "level_count": gdef["level_count"],
                    "thresholds": gdef["thresholds"],
                    "resolved_thresholds": diag["resolved_thresholds"],
                    "class_counts": diag["class_counts"],
                    "valid_pixel_count": diag["valid_pixel_count"],
                    "value_range": diag.get("value_range"),
                    "output_dir": str(g_dir),
                })
                group_class_paths.append(diag["class_raster"])
                group_value_paths.append(diag.get("value_raster"))
                group_presc_paths.append(presc["prescription_raster"])

            job.progress("prescription", 1.0, "分组处方图完成，开始合并...")
            job.log(f"合并 {G} 个分组的诊断图与处方图...")
            stage_suffix = STAGE_NAME_SUFFIX.get(stage_config["stage_key"], stage_config["stage_key"])
            merged_class = merge_group_rasters(
                group_class_paths, out_dir / f"{base_name}_diagnose_{stage_suffix}.tif"
            )
            merged_value = None
            if all(group_value_paths):
                merged_value = merge_group_rasters(
                    group_value_paths, out_dir / f"{base_name}_diagnose_{stage_suffix}_value.tif"
                )
            merged_presc = merge_group_rasters(
                group_presc_paths, out_dir / f"{base_name}_prescription.tif"
            )
            diagnose_result = {
                "class_raster": merged_class["output_path"],
                "value_raster": merged_value["output_path"] if merged_value else None,
                "valid_pixel_count": sum(g["valid_pixel_count"] for g in group_results),
                "value_range": {"min": value_min, "max": value_max},
            }
            final_raster = merged_presc["output_path"]
            job.log(f"处方图完成（{G} 组合并）: {Path(final_raster).name}")

            merged_counts = {}
            for gmeta in group_meta_list:
                for level, count in gmeta["class_counts"].items():
                    merged_counts[level] = merged_counts.get(level, 0) + count
            merged_meta = {
                "make_mode": "roi",
                "grouped": True,
                "stage_key": stage_config["stage_key"],
                "stage_name": stage_config["name"],
                "diagnosis_name": stage_config["diagnosis_name"],
                "index_name": stage_config["index_name"],
                "index_formula": stage_config["index_formula"],
                "diagnosis_formula": stage_config["diagnosis_formula"],
                "groups": group_meta_list,
                "class_counts": merged_counts,
                "valid_pixel_count": diagnose_result["valid_pixel_count"],
                "value_range": diagnose_result["value_range"],
                "output_class_raster": str(merged_class["output_path"]),
                "output_value_raster": str(merged_value["output_path"]) if merged_value else None,
                "output_prescription_raster": str(merged_presc["output_path"]),
                "note": "各分组独立分级阈值与施肥映射（同一诊断模型）；分组原始输出见 group_* 子目录",
            }
            merged_meta_path = make_unique_output_path(
                out_dir / f"{base_name}_diagnose_{stage_suffix}.json"
            )
            merged_meta_path.write_text(
                json.dumps(merged_meta, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        else:
            job.log("第一遍扫描：统计诊断值样本...")
            diagnose_result = run_diagnosis(
                band_paths=band_paths_diag,
                stage_config=stage_config,
                output_dir=out_dir,
                defaults=CONFIG["defaults"],
                progress=diag_progress,
                base_name=base_name,
                roi_polygons=roi_polygons,
                roi_meta=roi_meta,
            )
            job.log(
                f"诊断图完成: {Path(diagnose_result['class_raster']).name}，"
                f"有效像素 {diagnose_result['valid_pixel_count']}"
            )

            job.progress("prescription", 0.0, "开始写出处方图...")

            def pres_progress(fraction, message):
                job.progress("prescription", fraction, message)

            prescription_result = write_prescription_raster(
                class_raster_path=diagnose_result["class_raster"],
                output_path=out_dir / f"{base_name}_prescription.tif",
                mapping=mapping,
                defaults=CONFIG["defaults"],
                progress=pres_progress,
            )
            final_raster = prescription_result["prescription_raster"]
            job.log(f"处方图完成: {Path(final_raster).name}")

            if prescription_result["unmatched_class_counts"]:
                job.log(
                    f"警告: 存在未映射的分级值 {prescription_result['unmatched_class_counts']}，"
                    "相应像元已按无效值(NaN)写出"
                )
            group_results = None

        job.progress("resample", 0.0, f"开始 {resample_target_m:g}m 地面栅格重采样...")

        def resample_progress(fraction, message):
            job.progress("resample", fraction, message)

        resampled = write_resampled_prescription(
            input_path=final_raster,
            output_path=out_dir / f"{base_name}_prescription.tif",
            src_crs=band_crs,
            target_ground_resolution_m=resample_target_m,
            method=resample_method,
            progress=resample_progress,
        )
        final_raster = resampled["prescription_raster"]
        prescription_stats = resampled["statistics"]
        rs_meta = resampled["metadata"]["resampling"]
        job.log(
            f"{resample_target_m:g}m 地面栅格重采样完成: {Path(final_raster).name}"
            f"（源地面分辨率 {rs_meta['source_ground_resolution_m']['x']:.3f} m，"
            f"输出 {rs_meta['output_grid_size']['width']}×{rs_meta['output_grid_size']['height']}）"
        )

        # 调色板：优先用户自定义色（前端每个等级传 color），否则按级数取默认配色；
        # 分组模式取各分组颜色按级号合并（级号相同者优先取先出现的分组）
        if group_defs:
            max_levels = max(g["level_count"] for g in group_defs)
            palette = preview_core.build_class_legend(max_levels)
            for gdef in group_defs:
                for t in gdef["thresholds"]:
                    lv = int(t["level"])
                    c = t.get("color")
                    if isinstance(c, str) and len(c) == 7 and c.startswith("#") and 0 < lv <= max_levels:
                        palette[lv - 1] = c
            level_count_preview = max_levels
        else:
            palette = preview_core.build_class_legend(len(levels))
            user_palette = [t.get("color") for t in stage_config["thresholds"]]
            if all(isinstance(c, str) and len(c) == 7 and c.startswith("#") for c in user_palette):
                palette = user_palette
            level_count_preview = len(levels)

        job.progress("result_preview", 0.2, "生成诊断图预览...")
        preview_core.generate_class_preview(
            diagnose_result["class_raster"],
            level_count=level_count_preview,
            out_png=PREVIEW_DIR / "class_preview.png",
            max_long_side=int(CONFIG["preview"]["max_pixels_long_side"]),
            palette=palette,
        )
        job.progress("result_preview", 0.7, "生成处方图预览...")
        pres_preview = preview_core.generate_prescription_preview(
            final_raster,
            out_png=PREVIEW_DIR / "prescription_preview.png",
            max_long_side=int(CONFIG["preview"]["max_pixels_long_side"]),
        )

        output_files = []
        for path in sorted(out_dir.iterdir()):
            if path.is_file():
                output_files.append({
                    "name": path.name,
                    "size": path.stat().st_size,
                    "path": str(path),
                    "download_url": f"/api/download?path={path}",
                })

        level_stats = []
        if not group_defs:
            class_counts = diagnose_result["class_counts"]
            total = diagnose_result["valid_pixel_count"] or 1
            for threshold, resolved in zip(stage_config["thresholds"], diagnose_result["resolved_thresholds"]):
                level = int(threshold["level"])
                count = class_counts.get(str(level), 0)
                level_stats.append({
                    "level": level,
                    "color": palette[level - 1],
                    "growth_label": threshold.get("growth_label", ""),
                    "min": resolved["min"],
                    "max": resolved["max"],
                    "fertilizer": mapping.get(str(level)),
                    "count": count,
                    "percent": round(count / total * 100.0, 2),
                })

        result = {
            "run_id": run_ts,
            "output_dir": str(out_dir),
            "final_raster": str(final_raster),
            "class_raster": str(diagnose_result["class_raster"]),
            "stage": {
                "id": stage_id,
                "name": stage_config["name"],
                "index_name": stage_config["index_name"],
                "index_formula": stage_config["index_formula"],
                "diagnosis_name": stage_config["diagnosis_name"],
                "diagnosis_formula": stage_config["diagnosis_formula"],
            },
            "make_mode": "roi" if roi_polygons is not None else "full",
            "roi_region_count": len(roi_polygons) if roi_polygons is not None else 0,
            "grouped": bool(group_defs),
            "group_results": group_results,
            "mapping_mode": mapping_mode,
            "mapping": mapping,
            "unit": "kg/mu",
            "level_stats": level_stats,
            "valid_pixel_count": diagnose_result["valid_pixel_count"],
            "prescription_stats": prescription_stats,
            "diagnosis_value_range": diagnose_result.get("value_range"),
            "roi_area": roi_area,
            "prescription_preview_range": {
                "min": pres_preview["min"], "max": pres_preview["max"],
            },
            "gradient_css": preview_core.build_prescription_gradient_css(),
            "output_files": output_files,
            "elapsed_seconds": round(time.time() - started, 1),
        }
        with _state_lock:
            SESSION.last_result = result
        job.finish(result=result)

    job, error = _job_manager.start("诊断与处方生成", phase_weights, runner)
    if error:
        return jsonify({"error": error}), 409
    return jsonify({"ok": True, "message": "任务已启动"})


# --------------------------------------------------------------------------
# 结果查询 / 下载 / 模板替换
# --------------------------------------------------------------------------

@app.get("/api/results")
def api_results():
    with _state_lock:
        result = SESSION.last_result
    return jsonify({"result": result})


@app.get("/api/job")
def api_job():
    status = _job_manager.current_status()
    return jsonify({"job": status})


@app.get("/api/download")
def api_download():
    raw = request.args.get("path", "")
    path = Path(raw).resolve()
    workspace_root = WORKSPACE.resolve()
    if workspace_root not in path.parents and path != workspace_root:
        abort(403)
    if not path.is_file():
        abort(404)
    return send_file(path, as_attachment=True, download_name=path.name)


@app.get("/previews/<path:name>")
def api_preview(name):
    return send_from_directory(str(PREVIEW_DIR), name)


@app.post("/api/dj2pr")
def api_dj2pr():
    if _job_manager.is_running():
        return jsonify({"error": "已有任务正在运行，请等待完成。"}), 409

    data = request.get_json(force=True, silent=True) or {}
    template_path = Path(str(data.get("template_path") or "").strip().strip('"'))
    method = data.get("method") or "nearest"
    if method not in ("nearest", "average"):
        return jsonify({"error": "重采样方法仅支持 nearest / average"}), 400
    if not template_path.is_file():
        return jsonify({"error": f"模板文件不存在: {template_path}"}), 400

    with _state_lock:
        result = SESSION.last_result
    source_path = data.get("source_path")
    if not source_path:
        if not result:
            return jsonify({"error": "尚无本次生成的处方图，请先运行诊断生成或手动填写路径"}), 400
        candidates = [
            Path(f["path"]) for f in result.get("output_files", [])
            if f["name"].endswith(".tif") and "prescription" in f["name"] and "diagnose" not in f["name"]
        ]
        if not candidates:
            return jsonify({"error": "未在本次结果中找到处方图 tif"}), 400
        source_path = candidates[0]
    source_path = Path(source_path)
    if not source_path.is_file():
        return jsonify({"error": f"处方图不存在: {source_path}"}), 400

    out_dir = DJ2PR_DIR / time.strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    def runner(job):
        job.progress("convert", 0.1, "正在按模板网格重采样...")
        report = dj2pr_core.convert_by_template(
            dj_path=template_path,
            my_path=source_path,
            new_fertilizer_path=out_dir / "new_fertilizer.tif",
            method=method,
        )
        job.progress("convert", 1.0, "模板替换完成")
        job.log(f"模板替换输出: {report['new_fertilizer_path']}")
        job.finish(result={"dj2pr": report})

    job, error = _job_manager.start("模板替换", [("convert", "模板替换", 1.0)], runner)
    if error:
        return jsonify({"error": error}), 409
    return jsonify({"ok": True, "message": "模板替换任务已启动"})


@app.post("/api/reveal")
def api_reveal():
    data = request.get_json(force=True, silent=True) or {}
    raw = str(data.get("path") or "")
    path = Path(raw).resolve()
    workspace_root = WORKSPACE.resolve()
    if workspace_root not in path.parents and path != workspace_root:
        abort(403)
    target = path if path.is_dir() else path.parent
    try:
        if os.name == "nt":
            os.startfile(str(target))  # noqa: S606 - 本地工具，打开输出目录
        elif os.uname().sysname == "Darwin":
            subprocess.Popen(["open", str(target)])
        else:
            subprocess.Popen(["xdg-open", str(target)])
    except OSError as exc:
        return jsonify({"error": f"无法打开文件夹: {exc}"}), 500
    return jsonify({"ok": True})


@app.post("/api/reset")
def api_reset():
    SESSION.reset()
    for directory in (UPLOAD_TMP_DIR, CURRENT_DIR, PREVIEW_DIR):
        if directory.exists():
            for item in directory.iterdir():
                try:
                    if item.is_file():
                        item.unlink()
                    else:
                        shutil.rmtree(item)
                except OSError:
                    pass
    _UPLOADS.clear()
    return jsonify({"ok": True, "session": session_payload()})


def _port_in_use(host, port):
    """检测端口是否已被占用（含上次未退干净的服务进程）。"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((host if host != "0.0.0.0" else "127.0.0.1", port)) == 0


def _kill_existing_service(host, port):
    """启动前清理残留的服务进程：端口被占用时，找到监听该端口的进程并结束它。"""
    if not _port_in_use(host, port):
        return
    try:
        result = subprocess.run(
            ["netstat", "-ano", "-p", "TCP"],
            capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return
    local = f"{host}:{port}" if host != "0.0.0.0" else f"0.0.0.0:{port}"
    pids = set()
    for line in result.stdout.splitlines():
        parts = line.split()
        # 形如: TCP  127.0.0.1:8765  0.0.0.0:0  LISTENING  12345
        if len(parts) >= 5 and parts[0] == "TCP" and parts[3] == "LISTENING" and parts[1] == local:
            pids.add(parts[4])
    current = os.getpid()
    for pid in pids:
        if not pid.isdigit() or int(pid) == current:
            continue
        try:
            subprocess.run(
                ["taskkill", "/PID", pid, "/T", "/F"],
                capture_output=True, timeout=10,
            )
            print(f"已结束残留服务进程 (PID {pid})，释放端口 {port}。")
        except OSError:
            pass


def main():
    ensure_dirs()
    host = CONFIG["server"]["host"]
    port = int(CONFIG["server"]["port"])
    url = f"http://{host}:{port}"

    # 清理上次 Ctrl+C 后未退干净、仍占用端口的旧进程
    _kill_existing_service(host, port)

    # Ctrl+C 时强制退出整个进程：避免非守护线程（GDAL/任务线程）挂住导致端口不释放
    signal.signal(signal.SIGINT, lambda *_: os._exit(0))

    threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    print(f"施肥处方图生成系统已启动: {url}")
    print("按 Ctrl+C 停止服务。")
    app.run(host=host, port=port, threaded=True, debug=False)


if __name__ == "__main__":
    main()
