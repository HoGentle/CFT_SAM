"""后台任务线程与进度管理（单任务槽，本地单人使用）。"""

import threading
import traceback


class Job:
    def __init__(self, name, phase_weights):
        self.name = name
        self.phase_weights = phase_weights  # [(key, label, weight)]
        self.lock = threading.Lock()
        self.status = {
            "name": name,
            "running": True,
            "phase": None,
            "phase_label": "",
            "progress": 0.0,
            "message": "准备中...",
            "error": None,
            "result": None,
            "log": [],
        }

    def log(self, message):
        with self.lock:
            self.status["log"].append(message)
            if len(self.status["log"]) > 600:
                del self.status["log"][: len(self.status["log"]) - 600]

    def _phase_weight(self, phase_key):
        for key, _label, weight in self.phase_weights:
            if key == phase_key:
                return weight
        return 0.0

    def progress(self, phase_key, fraction, message=None):
        """报告当前阶段进度；fraction ∈ [0, 1]。"""
        fraction = max(0.0, min(1.0, float(fraction)))
        with self.lock:
            accumulated = 0.0
            current_weight = 1.0
            for key, label, weight in self.phase_weights:
                if key == phase_key:
                    current_weight = weight
                    break
                accumulated += weight
            self.status["phase"] = phase_key
            self.status["phase_label"] = dict((k, l) for k, l, _ in self.phase_weights).get(phase_key, phase_key)
            # phase_weights 以百分制权重给出（总和 100），accumulated + fraction*weight 即为百分比
            self.status["progress"] = round(accumulated + fraction * current_weight, 1)
            if message:
                self.status["message"] = message

    def finish(self, result=None, error=None):
        with self.lock:
            self.status["running"] = False
            if error is not None:
                self.status["error"] = str(error)
                self.status["message"] = f"失败: {error}"
            else:
                self.status["progress"] = 100.0
                self.status["message"] = "完成"
                self.status["result"] = result


class JobManager:
    """同一时间只允许一个后台任务（预览生成、诊断、模板替换共用）。"""

    def __init__(self):
        self._lock = threading.Lock()
        self._job = None

    def current_status(self):
        with self._lock:
            if self._job is None:
                return None
            with self._job.lock:
                return dict(self._job.status)

    def is_running(self):
        with self._lock:
            return self._job is not None and self._job.status["running"]

    def start(self, name, phase_weights, runner):
        """runner(job) 在后台线程执行。返回 (job, error)。"""
        with self._lock:
            if self._job is not None and self._job.status["running"]:
                return None, "已有任务正在运行，请等待其完成。"
            job = Job(name, phase_weights)
            self._job = job

        def _thread_main():
            try:
                runner(job)
            except Exception as exc:  # noqa: BLE001 - 后台任务统一兜底
                job.log(f"[错误] {exc}")
                job.log(traceback.format_exc(limit=6))
                job.finish(error=exc)

        thread = threading.Thread(target=_thread_main, daemon=True, name=f"job-{name}")
        thread.start()
        return job, None
