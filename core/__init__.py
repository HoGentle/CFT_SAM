"""core 包初始化。

Windows 下 pip 安装的 rasterio 打包了与自身 PROJ 版本匹配的 proj.db；
若机器上另有 conda/系统的旧版 proj.db 被优先找到，EPSG 创建与重投影会报
DATABASE.LAYOUT.VERSION 不匹配错误。这里在 rasterio 被导入前（PROJ 上下文
创建时读取环境变量）显式指向 rasterio 自带的 proj_data。仅在用户未自行
设置 PROJ_DATA/PROJ_LIB 时生效；非 Windows 或 conda 版 rasterio 无副作用。
"""

import importlib.util
import os
from pathlib import Path


def _ensure_bundled_proj_data():
    if os.name != "nt":
        return
    try:
        spec = importlib.util.find_spec("rasterio")
    except (ImportError, ValueError):
        return
    if spec is None or not spec.submodule_search_locations:
        return
    proj_data = Path(next(iter(spec.submodule_search_locations))) / "proj_data"
    if (proj_data / "proj.db").is_file():
        os.environ.setdefault("PROJ_DATA", str(proj_data))
        os.environ.setdefault("PROJ_LIB", str(proj_data))


_ensure_bundled_proj_data()
