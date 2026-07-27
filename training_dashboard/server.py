"""Zero-dependency HTTP server for live RF-DETR training metrics."""

from __future__ import annotations

import argparse
import csv
import json
import mimetypes
import os
import re
import subprocess
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote, urlparse


STATIC_DIR = (Path(__file__).resolve().parent / "static").resolve()
METRIC_ALIASES = {
    "precision": (
        "val/precision",
        "val/precision(B)",
        "metrics/precision(B)",
        "precision",
    ),
    "recall": ("val/recall", "val/recall(B)", "metrics/recall(B)", "recall"),
    "f1": ("val/F1", "val/f1", "metrics/F1", "F1", "f1"),
    "map50": ("val/mAP_50", "val/mAP50", "metrics/mAP50(B)", "mAP50"),
    "map5095": (
        "val/mAP_50_95",
        "val/mAP50-95",
        "metrics/mAP50-95(B)",
        "mAP50-95",
    ),
    "loss": ("train/loss_epoch", "train/loss", "loss"),
    "loss_bbox": ("train/loss_bbox_epoch", "train/loss_bbox", "loss_bbox"),
    "loss_ce": ("train/loss_ce_epoch", "train/loss_ce", "loss_ce"),
    "loss_giou": ("train/loss_giou_epoch", "train/loss_giou", "loss_giou"),
    "lr": ("lr-AdamW", "lr", "train/lr"),
}
EPOCH_KEYS = ("epoch", "Epoch")
SUBMISSION_PATTERN = re.compile(r"^submission_epoch(?P<epoch>\d+)\.csv$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--training-root",
        type=Path,
        default=Path(os.environ.get("TRAINING_ROOT", "/root/autodl-tmp/FuckAnthropic/rfdetr_runs")),
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6006)
    return parser.parse_args()


def run_text(command: list[str], timeout: float = 2.0) -> str:
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return completed.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return ""


def safe_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number else None


def first_metric(row: dict[str, Any], aliases: tuple[str, ...]) -> float | None:
    for key in aliases:
        value = safe_float(row.get(key))
        if value is not None:
            return value
    return None


def latest_file(root: Path, pattern: str) -> Path | None:
    if not root.exists():
        return None
    files = [path for path in root.rglob(pattern) if path.is_file()]
    return max(files, key=lambda path: path.stat().st_mtime) if files else None


def load_config(root: Path) -> tuple[dict[str, Any], Path | None]:
    config_path = latest_file(root, "training_config.json")
    if config_path is None:
        return {}, None
    try:
        return json.loads(config_path.read_text(encoding="utf-8")), config_path
    except (OSError, json.JSONDecodeError):
        return {}, config_path


def deep_find(payload: Any, key: str) -> Any:
    if isinstance(payload, dict):
        if key in payload:
            return payload[key]
        for value in payload.values():
            found = deep_find(value, key)
            if found is not None:
                return found
    elif isinstance(payload, list):
        for value in payload:
            found = deep_find(value, key)
            if found is not None:
                return found
    return None


def parse_metrics(path: Path | None) -> list[dict[str, float]]:
    if path is None:
        return []
    epochs: dict[int, dict[str, float]] = {}
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for raw in csv.DictReader(handle):
                epoch_value = next(
                    (safe_float(raw.get(key)) for key in EPOCH_KEYS if raw.get(key) not in (None, "")),
                    None,
                )
                if epoch_value is None:
                    continue
                epoch = int(epoch_value) + 1
                record = epochs.setdefault(epoch, {"epoch": float(epoch)})
                for target, aliases in METRIC_ALIASES.items():
                    value = first_metric(raw, aliases)
                    if value is not None:
                        record[target] = value
                precision = record.get("precision")
                recall = record.get("recall")
                if "f1" not in record and precision is not None and recall is not None:
                    denominator = precision + recall
                    record["f1"] = 2 * precision * recall / denominator if denominator else 0.0
    except OSError:
        return []
    return [epochs[key] for key in sorted(epochs)]


def gpu_status() -> dict[str, Any]:
    output = run_text(
        [
            "nvidia-smi",
            "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
            "--format=csv,noheader,nounits",
        ]
    )
    if not output:
        return {}
    fields = [field.strip() for field in output.splitlines()[0].split(",")]
    if len(fields) != 6:
        return {}
    return {
        "name": fields[0],
        "utilization": safe_float(fields[1]),
        "memory_used_mib": safe_float(fields[2]),
        "memory_total_mib": safe_float(fields[3]),
        "temperature": safe_float(fields[4]),
        "power_watts": safe_float(fields[5]),
    }


def training_process() -> dict[str, Any]:
    output = run_text(["ps", "-eo", "pid=,etimes=,args="])
    for line in output.splitlines():
        lowered = line.lower()
        if "rfdetr/train.py" not in lowered and "rfdetr.train" not in lowered:
            continue
        if "training_dashboard" in lowered:
            continue
        match = re.match(r"\s*(\d+)\s+(\d+)\s+(.+)", line)
        if match:
            return {
                "pid": int(match.group(1)),
                "elapsed_seconds": int(match.group(2)),
                "command": match.group(3),
            }
    return {}


def checkpoints(root: Path) -> list[dict[str, Any]]:
    if not root.exists():
        return []
    result = []
    for path in root.rglob("checkpoint*.pth"):
        match = re.search(r"checkpoint[_-](\d+)\.pth$", path.name)
        result.append(
            {
                "name": path.name,
                "epoch": int(match.group(1)) if match else None,
                "size_mb": round(path.stat().st_size / 1024 / 1024, 1),
                "modified": path.stat().st_mtime,
            }
        )
    return sorted(result, key=lambda item: item["modified"], reverse=True)


def submissions(root: Path) -> list[dict[str, Any]]:
    project_root = root.parent.resolve()
    result = []
    for path in project_root.glob("submission_epoch*.csv"):
        match = SUBMISSION_PATTERN.fullmatch(path.name)
        if not match or not path.is_file():
            continue
        result.append(
            {
                "name": path.name,
                "epoch": int(match.group("epoch")),
                "size_kb": round(path.stat().st_size / 1024, 1),
                "modified": path.stat().st_mtime,
                "download_url": (
                    "/api/download/submission.csv?name=" + quote(path.name)
                ),
            }
        )
    return sorted(
        result,
        key=lambda item: (item["epoch"], item["modified"]),
        reverse=True,
    )


def submission_watcher(root: Path) -> dict[str, Any]:
    state_path = latest_file(root, "submission_watcher_state.json")
    if state_path is None:
        return {}
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return {
        "status": state.get("status"),
        "started_from_epoch": state.get("started_from_epoch"),
        "current_epoch": state.get("current_epoch"),
        "last_completed_epoch": state.get("last_completed_epoch"),
        "last_error": state.get("last_error"),
        "updated_at": state.get("updated_at"),
    }


def build_status(root: Path) -> dict[str, Any]:
    metrics_path = latest_file(root, "metrics.csv")
    history = parse_metrics(metrics_path)
    config, config_path = load_config(root)
    process = training_process()
    gpu = gpu_status()
    checkpoint_rows = checkpoints(root)

    command = str(process.get("command", ""))
    epochs_match = re.search(r"--epochs(?:=|\s+)(\d+)", command)
    total_epochs = int(
        safe_float(deep_find(config, "epochs"))
        or (int(epochs_match.group(1)) if epochs_match else 100)
    )
    current = history[-1] if history else {}
    current_epoch = int(current.get("epoch", 0))
    f1_rows = [row for row in history if row.get("f1") is not None]
    best = max(f1_rows, key=lambda row: row["f1"]) if f1_rows else {}
    metrics_age = (
        time.time() - metrics_path.stat().st_mtime if metrics_path is not None else None
    )
    running = bool(process) or (metrics_age is not None and metrics_age < 120)
    complete = total_epochs > 0 and current_epoch >= total_epochs

    notes = deep_find(config, "notes") or {}
    resolution = int(safe_float(deep_find(config, "resolution")) or 1360)
    effective_batch = deep_find(config, "auto_batch_target_effective") or 16
    target_match = re.search(r"--target-f1(?:=|\s+)([0-9.]+)", command)
    target_f1 = (
        safe_float(deep_find(config, "target_f1"))
        or (safe_float(target_match.group(1)) if target_match else None)
        or 0.95
    )
    model_name = (
        deep_find(notes, "model")
        or deep_find(config, "model_name")
        or root.name
        or "RF-DETR-2XL"
    )

    return {
        "timestamp": time.time(),
        "status": "completed" if complete else "training" if running else "waiting",
        "model": model_name,
        "root": str(root),
        "epoch": current_epoch,
        "total_epochs": total_epochs,
        "current": current,
        "best": best,
        "history": history[-200:],
        "gpu": gpu,
        "process": process,
        "checkpoints": checkpoint_rows[:12],
        "submissions": submissions(root),
        "submission_watcher": submission_watcher(root),
        "metrics_file": str(metrics_path) if metrics_path else None,
        "metrics_download_url": "/api/download/metrics.csv" if metrics_path else None,
        "config_file": str(config_path) if config_path else None,
        "metrics_age_seconds": round(metrics_age, 1) if metrics_age is not None else None,
        "parameters": {
            "resolution": resolution,
            "precision": "BF16 / FP16 auto",
            "effective_batch": effective_batch,
            "scheduler": "cosine",
            "ema": bool(deep_find(config, "use_ema") if config else True),
            "holdout": "25% balanced",
            "tiling": "720px + full image",
            "target_f1": target_f1,
        },
    }


class DashboardHandler(BaseHTTPRequestHandler):
    training_root: Path
    cache: tuple[float, dict[str, Any]] = (0.0, {})

    def log_message(self, format_string: str, *args: Any) -> None:
        print(f"[dashboard] {self.address_string()} {format_string % args}")

    def send_bytes(
        self,
        content: bytes,
        content_type: str,
        status: int = 200,
        *,
        include_body: bool = True,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        for name, value in (extra_headers or {}).items():
            self.send_header(name, value)
        self.end_headers()
        if include_body:
            self.wfile.write(content)

    def send_metrics_csv(self, *, include_body: bool = True) -> None:
        metrics_path = latest_file(self.training_root, "metrics.csv")
        if metrics_path is None:
            self.send_error(HTTPStatus.NOT_FOUND, "Training metrics CSV is not available yet")
            return
        try:
            content = metrics_path.read_bytes()
        except OSError:
            self.send_error(HTTPStatus.SERVICE_UNAVAILABLE, "Training metrics CSV is being updated")
            return
        self.send_bytes(
            content,
            "text/csv; charset=utf-8",
            include_body=include_body,
            extra_headers={
                "Content-Disposition": 'attachment; filename="rfdetr_metrics_current.csv"',
            },
        )

    def send_submission_csv(
        self, name: str | None, *, include_body: bool = True
    ) -> None:
        if name is None or SUBMISSION_PATTERN.fullmatch(name) is None:
            self.send_error(HTTPStatus.NOT_FOUND, "Submission CSV not found")
            return
        available = {item["name"]: item for item in submissions(self.training_root)}
        if name not in available:
            self.send_error(HTTPStatus.NOT_FOUND, "Submission CSV not found")
            return
        target = (self.training_root.parent.resolve() / name).resolve()
        if target.parent != self.training_root.parent.resolve():
            self.send_error(HTTPStatus.NOT_FOUND, "Submission CSV not found")
            return
        try:
            content = target.read_bytes()
        except OSError:
            self.send_error(HTTPStatus.SERVICE_UNAVAILABLE, "Submission CSV is unavailable")
            return
        self.send_bytes(
            content,
            "text/csv; charset=utf-8",
            include_body=include_body,
            extra_headers={
                "Content-Disposition": f'attachment; filename="{name}"',
            },
        )

    def static_target(self, route: str) -> Path | None:
        relative = "index.html" if route == "/" else route.lstrip("/")
        target = (STATIC_DIR / relative).resolve()
        try:
            target.relative_to(STATIC_DIR)
        except ValueError:
            return None
        return target if target.is_file() else None

    def do_HEAD(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        route = parsed.path
        if route == "/api/health":
            self.send_bytes(
                b'{"ok":true}',
                "application/json; charset=utf-8",
                include_body=False,
            )
            return
        if route == "/api/download/metrics.csv":
            self.send_metrics_csv(include_body=False)
            return
        if route == "/api/download/submission.csv":
            name = parse_qs(parsed.query).get("name", [None])[0]
            self.send_submission_csv(name, include_body=False)
            return
        target = self.static_target(route)
        if target is None:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        content_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        self.send_bytes(
            target.read_bytes(),
            f"{content_type}; charset=utf-8",
            include_body=False,
        )

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        route = parsed.path
        if route == "/api/health":
            self.send_bytes(b'{"ok":true}', "application/json; charset=utf-8")
            return
        if route == "/api/status":
            now = time.time()
            cached_at, payload = self.cache
            if now - cached_at > 2:
                payload = build_status(self.training_root)
                type(self).cache = (now, payload)
            self.send_bytes(
                json.dumps(payload, separators=(",", ":")).encode("utf-8"),
                "application/json; charset=utf-8",
            )
            return
        if route == "/api/download/metrics.csv":
            self.send_metrics_csv()
            return
        if route == "/api/download/submission.csv":
            name = parse_qs(parsed.query).get("name", [None])[0]
            self.send_submission_csv(name)
            return

        target = self.static_target(route)
        if target is None:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        content_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        self.send_bytes(target.read_bytes(), f"{content_type}; charset=utf-8")


def main() -> None:
    args = parse_args()
    DashboardHandler.training_root = args.training_root.resolve()
    server = ThreadingHTTPServer((args.host, args.port), DashboardHandler)
    print(
        f"RF-DETR dashboard: http://{args.host}:{args.port} "
        f"(training root: {DashboardHandler.training_root})",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
