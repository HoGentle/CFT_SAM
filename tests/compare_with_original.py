"""回归对比测试：验证 Redo/core 的诊断与处方输出与原脚本一致。

用法: python tests/compare_with_original.py
前提: 原 code/diagnose.py、code/prescription.py 及其 json 配置可访问。
样本上限关闭（max_quantile_samples=0）时，两边都使用全量有效像素计算分位数，
输出应完全一致（蓄水池均匀采样仅在启用上限时介入）。
"""

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import rasterio

REDO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REDO_ROOT))

from core.diagnose import run_diagnosis  # noqa: E402
from core.prescription import write_prescription_raster  # noqa: E402

ORIGINAL_CODE_DIR = REDO_ROOT.parent  # code/
DATA_DIR = Path(r"E:\pythoncode\wurenji\nx_data")
ORIGINAL_PY = Path(r"D:\miniconda3\envs\farm_edge\python.exe")


def load_tif(path):
    with rasterio.open(path) as src:
        return src.read(1)


def compare_tif(a_path, b_path, label, tol=0.0):
    a = load_tif(a_path)
    b = load_tif(b_path)
    assert a.shape == b.shape, f"{label}: 形状不一致 {a.shape} vs {b.shape}"
    if a.dtype.kind == "f" or b.dtype.kind == "f":
        fa = np.isfinite(a) if a.dtype.kind == "f" else np.ones(a.shape, dtype=bool)
        fb = np.isfinite(b) if b.dtype.kind == "f" else np.ones(b.shape, dtype=bool)
        assert np.array_equal(fa, fb), f"{label}: 有效区域不一致"
        valid = fa & fb
    else:
        valid = np.ones(a.shape, dtype=bool)
    diff = np.abs(a.astype(np.float64)[valid] - b.astype(np.float64)[valid])
    max_diff = float(diff.max()) if diff.size else 0.0
    assert max_diff <= tol, f"{label}: 最大差异 {max_diff} > {tol}（差异像素 {int(np.count_nonzero(diff > tol))}）"
    print(f"  [一致] {label}")


def compare_world_file(a_path, b_path, label):
    a = Path(a_path).with_suffix(".tfw").read_text()
    b = Path(b_path).with_suffix(".tfw").read_text()
    assert a == b, f"{label}: tfw 不一致\n{a}\n---\n{b}"
    print(f"  [一致] {label} .tfw")


def prepare_original_env(tmp: Path, input_path: Path, diagnose_out: Path, prescription_out: Path):
    """把原脚本拷贝到临时目录并写好指向测试数据的配置。"""
    work = tmp / "original"
    work.mkdir(parents=True, exist_ok=True)
    for name in ("diagnose.py", "prescription.py"):
        shutil.copy(ORIGINAL_CODE_DIR / name, work / name)

    diagnose_config = json.loads((ORIGINAL_CODE_DIR / "diagnose.json").read_text(encoding="utf-8"))
    diagnose_config["defaults"]["input_path"] = str(input_path)
    diagnose_config["defaults"]["output_path"] = str(diagnose_out)
    diagnose_config["defaults"]["max_quantile_samples"] = 0  # 关闭样本上限，两边全量计算
    (work / "diagnose.json").write_text(
        json.dumps(diagnose_config, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    prescription_config = json.loads((ORIGINAL_CODE_DIR / "prescription.json").read_text(encoding="utf-8"))
    prescription_config["defaults"]["input_paths"] = [str(diagnose_out)]
    prescription_config["defaults"]["output_paths"] = [str(prescription_out)]
    (work / "prescription.json").write_text(
        json.dumps(prescription_config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return work


def run_original_diagnose(work: Path, stage_id: str):
    proc = subprocess.run(
        [str(ORIGINAL_PY), str(work / "diagnose.py")],
        input=f"\n\n{stage_id}\n", capture_output=True, text=True,
        encoding="utf-8", errors="replace", cwd=str(work), timeout=900,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"原 diagnose.py 失败:\n{proc.stdout[-1500:]}\n{proc.stderr[-2500:]}")


def run_original_prescription(work: Path):
    proc = subprocess.run(
        [str(ORIGINAL_PY), str(work / "prescription.py")],
        input="0\n", capture_output=True, text=True,
        encoding="utf-8", errors="replace", cwd=str(work), timeout=900,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"原 prescription.py 失败:\n{proc.stdout[-1500:]}\n{proc.stderr[-2500:]}")


def main():
    assert DATA_DIR.exists(), f"测试数据不存在: {DATA_DIR}"
    band_paths = {
        "B1": DATA_DIR / "result_RedEdge.tif",
        "B2": DATA_DIR / "result_Green.tif",
        "B3": DATA_DIR / "result_Red.tif",
        "B4": DATA_DIR / "result_NIR.tif",
    }
    with (REDO_ROOT / "config.json").open("r", encoding="utf-8") as f:
        config = json.load(f)

    tmp = Path(tempfile.mkdtemp(prefix="redo_compare_"))
    try:
        out_diagnose_orig = tmp / "diagnose_orig"
        out_diagnose_new = tmp / "diagnose_new"
        out_presc_orig = tmp / "presc_orig"
        out_presc_new = tmp / "presc_new"
        for p in (out_diagnose_orig, out_diagnose_new, out_presc_orig, out_presc_new):
            p.mkdir(parents=True, exist_ok=True)

        work = prepare_original_env(tmp, band_paths["B2"], out_diagnose_orig, out_presc_orig)

        for stage_id in ("2", "7"):
            stage_config = config["stages"][stage_id]
            levels = [int(t["level"]) for t in stage_config["thresholds"]]
            mapping = {str(level): round(10.0 * (1 + (3 - level) * 0.025), 2) for level in levels}
            stage_suffix = {
                "seedling": "youmiao", "tillering": "fennie", "jointing": "bajie",
                "booting": "yunsui", "heading": "chousui", "flowering": "yanghua",
                "pre_maturity": "chengshuqian", "maturity": "chengshu",
            }[stage_config["stage_key"]]

            # ---------- 诊断 ----------
            print(f"\n=== 生育期 {stage_id}（{stage_config['name']}）诊断对比 ===")
            run_original_diagnose(work, stage_id)
            orig_meta_path = out_diagnose_orig / f"result_diagnose_{stage_suffix}.json"
            orig_meta = json.loads(orig_meta_path.read_text(encoding="utf-8"))
            orig_class = Path(orig_meta["output_class_raster"])
            orig_value = Path(orig_meta["output_value_raster"])

            new_result = run_diagnosis(
                band_paths=band_paths,
                stage_config=stage_config,
                output_dir=out_diagnose_new,
                defaults={"processing_block_size": 1024, "max_quantile_samples": None,
                          "nodata_value": 255, "save_value_raster": True},
            )

            compare_tif(orig_class, new_result["class_raster"], f"诊断分级图 stage={stage_id}")
            compare_world_file(orig_class, new_result["class_raster"], f"诊断分级图 stage={stage_id}")

            for ot, nt in zip(orig_meta["level_thresholds"], new_result["resolved_thresholds"]):
                assert ot["level"] == nt["level"]
                for key in ("min", "max"):
                    if ot[key] is None:
                        assert nt[key] is None
                    else:
                        assert abs(ot[key] - nt[key]) < 1e-4 * max(1.0, abs(ot[key])), \
                            f"阈值不一致 {ot[key]} vs {nt[key]}"
            assert orig_meta["class_counts"] == new_result["class_counts"], \
                f"分级统计不一致 {orig_meta['class_counts']} vs {new_result['class_counts']}"
            print(f"  [一致] 分级阈值与统计 stage={stage_id}")

            # 诊断值图：有效区域一致（无效标记原为 255.0、新为 NaN，属有意改动）
            orig_arr = load_tif(orig_value)
            new_arr = load_tif(new_result["value_raster"])
            orig_valid = np.isfinite(orig_arr) & (orig_arr != 255.0)
            new_valid = np.isfinite(new_arr)
            assert np.array_equal(orig_valid, new_valid), "诊断值图有效区域不一致"
            assert float(np.abs(orig_arr[orig_valid] - new_arr[new_valid]).max()) == 0, "诊断值图数值有差异"
            print(f"  [一致] 诊断值图（有效区域） stage={stage_id}")

            # ---------- 处方图 ----------
            print(f"\n=== 生育期 {stage_id} 处方图对比 ===")
            run_original_prescription(work)
            orig_pres = out_presc_orig / f"{orig_class.stem}_prescription.tif"
            orig_pres_meta = json.loads(
                (out_presc_orig / f"{orig_class.stem}_prescription.json").read_text(encoding="utf-8")
            )

            pres_result = write_prescription_raster(
                class_raster_path=new_result["class_raster"],
                output_path=out_presc_new / f"{new_result['class_raster'].stem}_prescription.tif",
                mapping=mapping,
                defaults=config["defaults"],
            )
            compare_tif(orig_pres, pres_result["prescription_raster"], f"处方图 stage={stage_id}")
            compare_world_file(orig_pres, pres_result["prescription_raster"], f"处方图 stage={stage_id}")
            assert orig_pres_meta["class_counts"] == pres_result["class_counts"], "处方图分级统计不一致"
            assert orig_pres_meta["unmatched_class_counts"] == pres_result["unmatched_class_counts"], \
                "未映射分级统计不一致"
            print(f"  [一致] 处方图元数据 stage={stage_id}")

        print("\n全部对比通过：重构版与原脚本输出一致。")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
