"""Shared RF-DETR full-image and tiled inference utilities."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image


# Competition order after removing the original BDD class 5, "train".
# The following original classes shift down by one, so motorcycle is class 5.
CLASS_NAMES = [
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "motorcycle",
    "bicycle",
    "traffic light",
    "traffic sign",
]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
TILED_CLASS_IDS = {0, 1, 5, 6, 7, 8}


@dataclass(frozen=True)
class Detection:
    box: tuple[float, float, float, float]
    score: float
    class_id: int


def image_files(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def _axis_starts(length: int, tile_size: int, overlap: int) -> list[int]:
    if length <= tile_size:
        return [0]
    stride = tile_size - overlap
    starts = list(range(0, max(length - tile_size, 0) + 1, stride))
    last = length - tile_size
    if starts[-1] != last:
        starts.append(last)
    return starts


def _prediction_detections(
    prediction: Any, offset_x: int = 0, offset_y: int = 0
) -> list[Detection]:
    detections: list[Detection] = []
    for box, score, class_id in zip(
        prediction.xyxy,
        prediction.confidence,
        prediction.class_id,
        strict=True,
    ):
        class_index = int(class_id)
        if not 0 <= class_index < len(CLASS_NAMES):
            continue
        x1, y1, x2, y2 = (float(value) for value in box)
        detections.append(
            Detection(
                box=(
                    x1 + offset_x,
                    y1 + offset_y,
                    x2 + offset_x,
                    y2 + offset_y,
                ),
                score=float(score),
                class_id=class_index,
            )
        )
    return detections


def _iou(reference: Detection, others: list[Detection]) -> np.ndarray:
    if not others:
        return np.empty(0, dtype=np.float32)
    boxes = np.asarray([detection.box for detection in others], dtype=np.float32)
    x1 = np.maximum(reference.box[0], boxes[:, 0])
    y1 = np.maximum(reference.box[1], boxes[:, 1])
    x2 = np.minimum(reference.box[2], boxes[:, 2])
    y2 = np.minimum(reference.box[3], boxes[:, 3])
    intersection = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    reference_area = max(0.0, reference.box[2] - reference.box[0]) * max(
        0.0, reference.box[3] - reference.box[1]
    )
    other_areas = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(
        0.0, boxes[:, 3] - boxes[:, 1]
    )
    return intersection / np.maximum(reference_area + other_areas - intersection, 1e-9)


def class_aware_nms(
    detections: Iterable[Detection], iou_threshold: float
) -> list[Detection]:
    if not 0.0 < iou_threshold < 1.0:
        raise ValueError("NMS IoU must be between 0 and 1")

    kept: list[Detection] = []
    by_class: dict[int, list[Detection]] = {}
    for detection in detections:
        by_class.setdefault(detection.class_id, []).append(detection)

    for class_detections in by_class.values():
        pending = sorted(class_detections, key=lambda item: item.score, reverse=True)
        while pending:
            best = pending.pop(0)
            kept.append(best)
            overlaps = _iou(best, pending)
            pending = [
                detection
                for detection, overlap in zip(pending, overlaps, strict=True)
                if overlap <= iou_threshold
            ]
    return sorted(kept, key=lambda item: item.score, reverse=True)


def infer_image(
    model: Any,
    image_path: Path,
    *,
    resolution: int,
    minimum_confidence: float,
    tile_size: int = 720,
    tile_overlap: int = 160,
    nms_iou: float = 0.65,
    use_tiles: bool = True,
) -> list[Detection]:
    with Image.open(image_path) as opened:
        image = opened.convert("RGB")
        width, height = image.size
        full_prediction = model.predict(
            image,
            threshold=minimum_confidence,
            shape=(resolution, resolution),
            include_source_image=False,
        )
        detections = _prediction_detections(full_prediction)

        if use_tiles and (width > tile_size or height > tile_size):
            if tile_overlap < 0 or tile_overlap >= tile_size:
                raise ValueError("tile overlap must satisfy 0 <= overlap < tile size")
            tiles: list[Image.Image] = []
            positions: list[tuple[int, int, int, int]] = []
            for top in _axis_starts(height, tile_size, tile_overlap):
                for left in _axis_starts(width, tile_size, tile_overlap):
                    right = min(left + tile_size, width)
                    bottom = min(top + tile_size, height)
                    tiles.append(image.crop((left, top, right, bottom)))
                    positions.append((left, top, right, bottom))

            tile_predictions = model.predict(
                tiles,
                threshold=minimum_confidence,
                shape=(resolution, resolution),
                include_source_image=False,
            )
            if not isinstance(tile_predictions, list):
                tile_predictions = [tile_predictions]

            half_overlap = tile_overlap / 2.0
            for prediction, (left, top, right, bottom) in zip(
                tile_predictions, positions, strict=True
            ):
                tile_width = right - left
                tile_height = bottom - top
                core_left = 0.0 if left == 0 else half_overlap
                core_top = 0.0 if top == 0 else half_overlap
                core_right = tile_width if right == width else tile_width - half_overlap
                core_bottom = (
                    tile_height if bottom == height else tile_height - half_overlap
                )
                for detection in _prediction_detections(prediction):
                    if detection.class_id not in TILED_CLASS_IDS:
                        continue
                    center_x = (detection.box[0] + detection.box[2]) / 2.0
                    center_y = (detection.box[1] + detection.box[3]) / 2.0
                    if not (
                        core_left <= center_x <= core_right
                        and core_top <= center_y <= core_bottom
                    ):
                        continue
                    detections.append(
                        Detection(
                            box=(
                                detection.box[0] + left,
                                detection.box[1] + top,
                                detection.box[2] + left,
                                detection.box[3] + top,
                            ),
                            score=detection.score,
                            class_id=detection.class_id,
                        )
                    )

    return class_aware_nms(detections, nms_iou)


def write_prediction_cache(
    path: Path, records: Iterable[tuple[str, list[Detection]]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for image_name, detections in records:
            payload = {
                "image": image_name,
                "detections": [asdict(detection) for detection in detections],
            }
            handle.write(json.dumps(payload, separators=(",", ":")) + "\n")


def read_prediction_cache(path: Path) -> dict[str, list[Detection]]:
    records: dict[str, list[Detection]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            image_name = str(payload["image"])
            if image_name in records:
                raise ValueError(f"Duplicate image {image_name} at {path}:{line_number}")
            records[image_name] = [
                Detection(
                    box=tuple(float(value) for value in item["box"]),
                    score=float(item["score"]),
                    class_id=int(item["class_id"]),
                )
                for item in payload["detections"]
            ]
    return records
