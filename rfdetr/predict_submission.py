"""Generate the competition count submission with a trained RF-DETR-2XL model."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Iterable

try:
    from .inference_utils import CLASS_NAMES, image_files, infer_image
except ImportError:  # Direct execution: python rfdetr/predict_submission.py
    from inference_utils import CLASS_NAMES, image_files, infer_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("rfdetr_runs/rfdetr_2xl_1360/checkpoint_best_total.pth"),
    )
    parser.add_argument("--images", type=Path, default=Path("bdd100k_selected/images/test"))
    parser.add_argument("--thresholds", type=Path, default=Path("rfdetr/thresholds.json"))
    parser.add_argument("--resolution", type=int, default=1360)
    parser.add_argument("--tile-size", type=int, default=720)
    parser.add_argument("--tile-overlap", type=int, default=160)
    parser.add_argument("--nms-iou", type=float, default=0.65)
    parser.add_argument("--no-tiles", action="store_true")
    parser.add_argument("--submission", type=Path, default=Path("submission_rfdetr.csv"))
    parser.add_argument("--detections", type=Path, default=Path("rfdetr_detections.csv"))
    parser.add_argument(
        "--no-detections",
        action="store_true",
        help="Skip the optional per-box diagnostics CSV.",
    )
    parser.add_argument("--accept-pml", action="store_true")
    return parser.parse_args()


def load_thresholds(path: Path) -> list[float]:
    values = json.loads(path.read_text(encoding="utf-8"))
    missing = [name for name in CLASS_NAMES if name not in values]
    if missing:
        raise ValueError(f"Missing thresholds for: {', '.join(missing)}")
    thresholds = [float(values[name]) for name in CLASS_NAMES]
    if any(not 0.0 <= value <= 1.0 for value in thresholds):
        raise ValueError("All thresholds must be between 0 and 1")
    return thresholds


def format_submission_result(image_counts: Counter[int]) -> str:
    """Return the official nine-class count vector without nullable fields."""
    invalid_class_ids = sorted(
        class_id
        for class_id, count in image_counts.items()
        if count and not 0 <= class_id < len(CLASS_NAMES)
    )
    if invalid_class_ids:
        raise ValueError(f"Invalid submission class IDs: {invalid_class_ids}")
    return ";".join(
        str(image_counts.get(class_index, 0))
        for class_index in range(len(CLASS_NAMES))
    )


def write_submission(
    path: Path,
    image_paths: Iterable[Path],
    counts: dict[str, Counter[int]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["pic_name", "results"],
            extrasaction="raise",
            lineterminator="\n",
        )
        writer.writeheader()
        for image_path in image_paths:
            writer.writerow(
                {
                    "pic_name": image_path.name,
                    "results": format_submission_result(counts[image_path.name]),
                }
            )


def validate_submission(path: Path, expected_image_names: list[str]) -> None:
    """Fail fast if the generated CSV does not match the competition schema."""
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["pic_name", "results"]:
            raise ValueError(
                "Submission header must be exactly: pic_name,results"
            )
        rows = list(reader)

    actual_names = [row["pic_name"] for row in rows]
    if actual_names != expected_image_names:
        raise ValueError(
            "Submission image rows do not exactly match the sorted test image list"
        )

    expected_fields = len(CLASS_NAMES)
    for line_number, row in enumerate(rows, start=2):
        image_name = row["pic_name"]
        if image_name is None or not image_name.strip():
            raise ValueError(
                f"Null or empty pic_name value at CSV line {line_number}"
            )
        result = row["results"]
        if result is None or not result.strip():
            raise ValueError(
                f"Null or empty results value at CSV line {line_number}"
            )
        values = result.split(";")
        if len(values) != expected_fields or any(
            not value.isdigit() for value in values
        ):
            raise ValueError(
                f"Invalid results value at CSV line {line_number}: {result!r}"
            )


def main() -> None:
    args = parse_args()
    if not args.accept_pml:
        raise SystemExit("Rerun with --accept-pml after accepting PML-1.0.")

    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    thresholds = load_thresholds(args.thresholds.resolve())
    minimum_threshold = min(thresholds)

    try:
        from rfdetr_plus import RFDETR2XLarge
    except ImportError as exc:
        raise SystemExit(
            'RF-DETR+ is not installed. Install with: pip install "rfdetr[train,plus]"'
        ) from exc

    model = RFDETR2XLarge(
        pretrain_weights=str(checkpoint),
        resolution=args.resolution,
        num_classes=len(CLASS_NAMES),
        accept_platform_model_license=True,
    )
    image_paths = image_files(args.images.resolve())
    if not image_paths:
        raise RuntimeError(f"No test images found in {args.images.resolve()}")

    counts: dict[str, Counter[int]] = {}
    detection_rows: list[dict[str, object]] = []
    for index, image_path in enumerate(image_paths, start=1):
        detections = infer_image(
            model,
            image_path,
            resolution=args.resolution,
            minimum_confidence=minimum_threshold,
            tile_size=args.tile_size,
            tile_overlap=args.tile_overlap,
            nms_iou=args.nms_iou,
            use_tiles=not args.no_tiles,
        )
        image_counts: Counter[int] = Counter()
        for detection in detections:
            class_index = detection.class_id
            if detection.score < thresholds[class_index]:
                continue

            image_counts[class_index] += 1
            if not args.no_detections:
                x1, y1, x2, y2 = detection.box
                detection_rows.append(
                    {
                        "image_id": image_path.stem,
                        "class_id": class_index,
                        "class_name": CLASS_NAMES[class_index],
                        "confidence": round(detection.score, 6),
                        "x_min": round(x1, 2),
                        "y_min": round(y1, 2),
                        "x_max": round(x2, 2),
                        "y_max": round(y2, 2),
                    }
                )
        counts[image_path.name] = image_counts
        if index == 1 or index % 25 == 0 or index == len(image_paths):
            print(f"[{index:4d}/{len(image_paths)}] {image_path.name}")

    if not args.no_detections:
        args.detections.parent.mkdir(parents=True, exist_ok=True)
        with args.detections.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "image_id",
                    "class_id",
                    "class_name",
                    "confidence",
                    "x_min",
                    "y_min",
                    "x_max",
                    "y_max",
                ],
            )
            writer.writeheader()
            writer.writerows(detection_rows)

    write_submission(args.submission, image_paths, counts)
    validate_submission(args.submission, [path.name for path in image_paths])

    message = (
        f"Validated submission: {args.submission.resolve()} "
        f"({len(image_paths)} images, exact pic_name/results schema)"
    )
    if not args.no_detections:
        message += (
            f"; detections: {args.detections.resolve()} "
            f"({len(detection_rows)} boxes)"
        )
    print(message)


if __name__ == "__main__":
    main()
