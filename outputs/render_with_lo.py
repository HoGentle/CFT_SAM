import os
import runpy

os.environ["PATH"] = (
    r"C:\Program Files\LibreOffice\program;"
    r"C:\Users\hjt\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\poppler\Library\bin;"
    + os.environ.get("PATH", "")
)
runpy.run_path(
    r"C:\Users\hjt\.codex\plugins\cache\openai-primary-runtime\documents\26.909.22227\skills\documents\render_docx.py",
    run_name="__main__",
)
