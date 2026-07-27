"""Generate a validated competition CSV whenever RF-DETR finishes an epoch."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class CheckpointChanged(RuntimeError):
    """Raised when last.ckpt changes while its immutable snapshot is copied."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("rfdetr_runs/rfdetr_2xl_1360_refine"),
    )
    parser.add_argument(
        "--images",
        type=Path,
        default=Path("rfdetr_dataset_v2/test/images"),
    )
    parser.add_argument(
        "--thresholds",
        type=Path,
        default=Path("rfdetr/thresholds_epoch006.json"),
    )
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    parser.add_argument("--stable-seconds", type=float, default=8.0)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--accept-pml", action="store_true")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_state(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(state, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def checkpoint_epoch(path: Path) -> int:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required to inspect last.ckpt") from exc

    payload = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    if "epoch" not in payload:
        raise ValueError(f"Checkpoint has no epoch metadata: {path}")
    return int(payload["epoch"]) + 1


def checkpoint_is_stable(path: Path, stable_seconds: float) -> bool:
    try:
        stat = path.stat()
    except OSError:
        return False
    return stat.st_size > 0 and time.time() - stat.st_mtime >= stable_seconds


def snapshot_checkpoint(source: Path, destination: Path) -> None:
    before = source.stat()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.unlink(missing_ok=True)
    shutil.copy2(source, destination)
    after = source.stat()
    if (
        before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
    ):
        destination.unlink(missing_ok=True)
        raise CheckpointChanged("last.ckpt changed while its snapshot was copied")


def archive_existing(target: Path, archive_dir: Path, epoch: int) -> Path | None:
    if not target.exists():
        return None
    archive_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archived = archive_dir / (
        f"{target.stem}_before_refine_epoch{epoch:03d}_{stamp}{target.suffix}"
    )
    counter = 1
    while archived.exists():
        archived = archive_dir / (
            f"{target.stem}_before_refine_epoch{epoch:03d}_{stamp}_{counter}"
            f"{target.suffix}"
        )
        counter += 1
    os.replace(target, archived)
    return archived


def generate_submission(
    *,
    project_root: Path,
    run_dir: Path,
    images: Path,
    thresholds: Path,
    source: Path,
    epoch: int,
) -> tuple[Path, Path | None]:
    snapshot = run_dir / f".submission_epoch{epoch:03d}.ckpt"
    partial = project_root / f".submission_epoch{epoch:03d}.csv.partial"
    target = project_root / f"submission_epoch{epoch:03d}.csv"
    partial.unlink(missing_ok=True)

    try:
        snapshot_checkpoint(source, snapshot)
        actual_epoch = checkpoint_epoch(snapshot)
        if actual_epoch != epoch:
            raise CheckpointChanged(
                f"Snapshot epoch changed from {epoch} to {actual_epoch}"
            )

        command = [
            sys.executable,
            "-u",
            str(project_root / "rfdetr" / "predict_submission.py"),
            "--accept-pml",
            "--checkpoint",
            str(snapshot),
            "--images",
            str(images),
            "--thresholds",
            str(thresholds),
            "--submission",
            str(partial),
            "--no-detections",
        ]
        subprocess.run(command, cwd=project_root, check=True)
        if not partial.is_file() or partial.stat().st_size == 0:
            raise RuntimeError(f"Inference did not create {partial}")

        archived = archive_existing(
            target,
            project_root / "submission_archive",
            epoch,
        )
        os.replace(partial, target)
        return target, archived
    finally:
        snapshot.unlink(missing_ok=True)
        partial.unlink(missing_ok=True)


def main() -> None:
    args = parse_args()
    if not args.accept_pml:
        raise SystemExit("Rerun with --accept-pml after accepting PML-1.0.")
    if args.poll_seconds <= 0 or args.stable_seconds < 0:
        raise SystemExit("Polling intervals must be positive.")

    project_root = args.project_root.resolve()
    run_dir = (
        args.run_dir.resolve()
        if args.run_dir.is_absolute()
        else (project_root / args.run_dir).resolve()
    )
    images = (
        args.images.resolve()
        if args.images.is_absolute()
        else (project_root / args.images).resolve()
    )
    thresholds = (
        args.thresholds.resolve()
        if args.thresholds.is_absolute()
        else (project_root / args.thresholds).resolve()
    )
    source = run_dir / "last.ckpt"
    state_path = run_dir / "submission_watcher_state.json"

    for required in (project_root / "rfdetr" / "predict_submission.py", images, thresholds):
        if not required.exists():
            raise FileNotFoundError(required)

    state = load_state(state_path)
    state.setdefault("processed", {})
    state.update(
        {
            "status": "watching",
            "project_root": str(project_root),
            "run_dir": str(run_dir),
            "updated_at": utc_now(),
        }
    )
    save_state(state_path, state)
    print(f"Watching {source}", flush=True)

    while True:
        try:
            if not checkpoint_is_stable(source, args.stable_seconds):
                if args.once:
                    raise RuntimeError(f"Stable checkpoint not available: {source}")
                time.sleep(args.poll_seconds)
                continue

            epoch = checkpoint_epoch(source)
            if state.get("started_from_epoch") is None:
                state["started_from_epoch"] = epoch
                state["updated_at"] = utc_now()
                save_state(state_path, state)

            target = project_root / f"submission_epoch{epoch:03d}.csv"
            processed = state.setdefault("processed", {})
            if str(epoch) in processed and target.is_file():
                if args.once:
                    break
                time.sleep(args.poll_seconds)
                continue

            state.update(
                {
                    "status": "generating",
                    "current_epoch": epoch,
                    "last_error": None,
                    "updated_at": utc_now(),
                }
            )
            save_state(state_path, state)
            print(f"Generating submission for epoch {epoch}", flush=True)

            generated, archived = generate_submission(
                project_root=project_root,
                run_dir=run_dir,
                images=images,
                thresholds=thresholds,
                source=source,
                epoch=epoch,
            )
            processed[str(epoch)] = {
                "file": generated.name,
                "size": generated.stat().st_size,
                "generated_at": utc_now(),
                "archived_previous": str(archived) if archived else None,
            }
            state.update(
                {
                    "status": "watching",
                    "last_completed_epoch": epoch,
                    "current_epoch": None,
                    "last_error": None,
                    "updated_at": utc_now(),
                }
            )
            save_state(state_path, state)
            print(f"Ready: {generated}", flush=True)
            if args.once:
                break
        except CheckpointChanged as exc:
            print(f"Retrying: {exc}", flush=True)
            if args.once:
                raise
        except Exception as exc:
            state.update(
                {
                    "status": "error",
                    "last_error": f"{type(exc).__name__}: {exc}",
                    "updated_at": utc_now(),
                }
            )
            save_state(state_path, state)
            print(f"Submission generation failed: {exc}", flush=True)
            if args.once:
                raise

        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
