"""本地保存对话框；只将用户确认的目标路径返回给服务进程。"""

import json
import sys
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox


def choose_path(default_name="感兴趣区域"):
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    try:
        while True:
            selected = filedialog.asksaveasfilename(
                parent=root, title="保存感兴趣区域：选择文件夹并填写文件名",
                initialfile=f"{default_name}.shp", defaultextension=".shp",
                filetypes=[("感兴趣区域", "*.shp")], confirmoverwrite=False,
            )
            if not selected:
                return None
            path = Path(selected)
            if path.suffix.lower() != ".shp":
                path = Path(str(path) + ".shp")
            existing = [path.with_suffix(f".{suffix}")
                        for suffix in ("shp", "shx", "dbf", "prj", "cpg")
                        if path.with_suffix(f".{suffix}").exists()]
            if existing and not messagebox.askyesno(
                "确认覆盖区域文件", "下列文件已存在，是否覆盖？\n\n" +
                "\n".join(p.name for p in existing), parent=root,
            ):
                continue
            return str(path)
    finally:
        root.destroy()


if __name__ == "__main__":
    print(json.dumps(choose_path(sys.argv[1] if len(sys.argv) > 1 else "感兴趣区域"), ensure_ascii=True))
