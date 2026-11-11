"""Local run configuration and progress files."""

import csv
import json
import os
import time
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
from importlib.metadata import version

import torch


def write_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


class TrainingRun:
    def __init__(self, args, device, start_step=0, elapsed_seconds=0.0):
        self.args = args
        self.started = time.perf_counter()
        self.last_time = self.started
        self.last_step = start_step
        self.elapsed_offset = elapsed_seconds
        self.intervals = deque(maxlen=10)
        checkpoint = Path(args.save_path)
        self.progress_path = checkpoint.with_suffix(".progress.json")
        self.metrics_path = checkpoint.with_suffix(".metrics.csv")
        config_path = checkpoint.with_suffix(".config.json")
        previous_config = {}
        if args.resume and config_path.exists():
            previous_config = json.loads(config_path.read_text(encoding="utf-8"))
        config = dict(vars(args))
        config.update(
            started_at_utc=previous_config.get("started_at_utc", datetime.now(timezone.utc).isoformat()),
            torch_version=torch.__version__,
            cuda_version=torch.version.cuda,
            gymnasium_version=version("gymnasium"),
            ale_version=version("ale-py"),
            numpy_version=version("numpy"),
            device=str(device),
            device_name=torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
            numerical_runtime={
                "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                "cudnn_deterministic": torch.backends.cudnn.deterministic,
                "cudnn_benchmark": torch.backends.cudnn.benchmark,
                "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            },
        )
        if args.resume:
            config["resumed_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(config_path, config)
        if args.resume and self.metrics_path.exists():
            with self.metrics_path.open(encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                fields = reader.fieldnames
                old_rows = list(reader)
            rows = [row for row in old_rows if int(row["step"]) <= start_step]
            if len(rows) != len(old_rows):
                with self.metrics_path.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=fields)
                    writer.writeheader()
                    writer.writerows(rows)
        else:
            self.metrics_path.write_text("", encoding="utf-8")

    def elapsed_seconds(self):
        return self.elapsed_offset + time.perf_counter() - self.started

    def record(self, step, loss, epsilon, buffer_size, episodes, last_reward,
               status="running", error=None):
        now = time.perf_counter()
        timestamp = datetime.now(timezone.utc)
        if self.last_step >= self.args.min_replay_size and step > self.last_step:
            self.intervals.append((step - self.last_step, now - self.last_time))
        rate = (sum(count for count, _ in self.intervals) /
                sum(duration for _, duration in self.intervals)) if self.intervals else None
        remaining = (self.args.steps - step) / rate if rate and status == "running" else None
        data = {
            "status": status,
            "updated_at_utc": timestamp.isoformat(),
            "pid": os.getpid(),
            "step": step,
            "total_steps": self.args.steps,
            "elapsed_seconds": round(self.elapsed_offset + now - self.started, 2),
            "training_steps_per_second": round(rate, 2) if rate else None,
            "remaining_seconds": round(remaining, 1) if remaining is not None else None,
            "estimated_finish_utc": (timestamp + timedelta(seconds=remaining)).isoformat()
                if remaining is not None else None,
            "loss": loss,
            "epsilon": epsilon,
            "buffer_size": buffer_size,
            "episodes": episodes,
            "last_episode_reward": last_reward,
            "error": error,
        }
        write_json(self.progress_path, data)
        with self.metrics_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(data))
            if handle.tell() == 0:
                writer.writeheader()
            writer.writerow(data)
        self.last_time = now
        self.last_step = step
