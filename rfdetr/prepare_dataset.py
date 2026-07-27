"""Create the directory layout RF-DETR expects without duplicating image data.

The source dataset uses Ultralytics' images/{split}, labels/{split} layout.
RF-DETR's YOLO reader expects {split}/images, {split}/labels. This script makes
hard links when possible and falls back to copying when the filesystem does not
support hard links.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
from pathlib import Path


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("bdd100k_selected"),
        help="Existing images/{split}, labels/{split} dataset root.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("rfdetr_dataset"),
        help="RF-DETR-compatible dataset root to create.",
    )
    parser.add_argument(
        "--holdout-fraction",
        type=float,
        default=0.25,
        help="Fraction of source val reserved from training/threshold calibration.",
    )
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def link_or_copy(source: Path, destination: Path) -> str:
    if destination.exists():
        if destination.stat().st_size != source.stat().st_size:
            raise RuntimeError(f"Destination conflicts with source: {destination}")
        return "existing"

    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
        return "linked"
    except OSError:
        shutil.copy2(source, destination)
        return "copied"


def label_counts(label_path: Path) -> list[int]:
    counts = [0] * len(CLASS_NAMES)
    if not label_path.exists():
        return counts
    for line in label_path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if not fields:
            continue
        class_id = int(fields[0])
        if not 0 <= class_id < len(CLASS_NAMES):
            raise ValueError(f"Invalid class {class_id} in {label_path}")
        counts[class_id] += 1
    return counts


def balanced_holdout(
    images: list[Path], label_dir: Path, fraction: float, seed: int
) -> set[str]:
    """Select an exact-size holdout and balance per-class object counts by swaps."""
    if not 0.05 <= fraction <= 0.5:
        raise ValueError("--holdout-fraction must be between 0.05 and 0.5")

    vectors = {
        image.name: label_counts(label_dir / f"{image.stem}.txt") for image in images
    }
    target_size = round(len(images) * fraction)
    ordered = sorted(
        images,
        key=lambda image: hashlib.sha256(
            f"{seed}:{image.name}".encode("utf-8")
        ).digest(),
    )
    selected = {image.name for image in ordered[:target_size]}
    outside = {image.name for image in ordered[target_size:]}

    totals = [
        sum(vector[class_id] for vector in vectors.values())
        for class_id in range(len(CLASS_NAMES))
    ]
    targets = [total * fraction for total in totals]
    current = [
        sum(vectors[name][class_id] for name in selected)
        for class_id in range(len(CLASS_NAMES))
    ]

    def imbalance(values: list[int]) -> float:
        return sum(
            ((value - target) / max(target, 1.0)) ** 2
            for value, target in zip(values, targets, strict=True)
        )

    rng = random.Random(seed)
    score = imbalance(current)
    selected_list = list(selected)
    outside_list = list(outside)
    for _ in range(25_000):
        selected_name = rng.choice(selected_list)
        outside_name = rng.choice(outside_list)
        candidate = [
            current[class_id]
            - vectors[selected_name][class_id]
            + vectors[outside_name][class_id]
            for class_id in range(len(CLASS_NAMES))
        ]
        candidate_score = imbalance(candidate)
        if candidate_score >= score:
            continue

        selected.remove(selected_name)
        selected.add(outside_name)
        outside.remove(outside_name)
        outside.add(selected_name)
        selected_index = selected_list.index(selected_name)
        outside_index = outside_list.index(outside_name)
        selected_list[selected_index] = outside_name
        outside_list[outside_index] = selected_name
        current = candidate
        score = candidate_score

    return selected


def prepare_images(
    source: Path,
    output: Path,
    old: str,
    new: str,
    images: list[Path] | None = None,
    *,
    allow_existing_extra: bool = False,
) -> dict[str, int]:
    stats = {"images": 0, "labels": 0, "linked": 0, "copied": 0, "existing": 0}
    image_dir = source / "images" / old
    label_dir = source / "labels" / old
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Missing image directory: {image_dir.resolve()}")

    if images is None:
        images = sorted(
            path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES
        )
    expected_names = {path.name for path in images}
    destination_images = output / new / "images"
    if destination_images.exists() and not allow_existing_extra:
        unexpected = {
            path.name
            for path in destination_images.iterdir()
            if path.suffix.lower() in IMAGE_SUFFIXES and path.name not in expected_names
        }
        if unexpected:
            raise RuntimeError(
                f"{destination_images} contains {len(unexpected)} files from a different "
                "split. Recreate the output directory before changing split settings."
            )

    for image_path in images:
        outcome = link_or_copy(image_path, output / new / "images" / image_path.name)
        stats["images"] += 1
        stats[outcome] += 1

        label_path = label_dir / f"{image_path.stem}.txt"
        if label_path.exists():
            outcome = link_or_copy(label_path, output / new / "labels" / label_path.name)
            stats["labels"] += 1
            stats[outcome] += 1
        elif old != "test":
            raise FileNotFoundError(f"Missing label for {image_path.name}: {label_path}")

    return stats


def write_data_yaml(output: Path) -> None:
    names = "\n".join(f"  {index}: {name}" for index, name in enumerate(CLASS_NAMES))
    text = (
        "path: .\n"
        "train: train/images\n"
        "val: valid/images\n"
        "test: test/images\n"
        f"names:\n{names}\n"
    )
    (output / "data.yaml").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    if source == output:
        raise ValueError("--source and --output must be different directories")

    output.mkdir(parents=True, exist_ok=True)
    split_stats: dict[str, dict[str, int]] = {}
    for old_split, new_split in (("train", "train"), ("test", "test")):
        stats = prepare_images(source, output, old_split, new_split)
        split_stats[new_split] = stats
        print(
            f"{new_split:5s}: images={stats['images']:5d}, labels={stats['labels']:5d}, "
            f"linked={stats['linked']:5d}, copied={stats['copied']:5d}, "
            f"existing={stats['existing']:5d}"
        )

    val_images = sorted(
        path
        for path in (source / "images" / "val").iterdir()
        if path.suffix.lower() in IMAGE_SUFFIXES
    )
    holdout_names = balanced_holdout(
        val_images,
        source / "labels" / "val",
        args.holdout_fraction,
        args.seed,
    )
    split_images = {
        "valid": [path for path in val_images if path.name not in holdout_names],
        "holdout": [path for path in val_images if path.name in holdout_names],
    }
    for new_split, images in split_images.items():
        stats = prepare_images(source, output, "val", new_split, images)
        split_stats[new_split] = stats
        print(
            f"{new_split:7s}: images={stats['images']:5d}, labels={stats['labels']:5d}, "
            f"linked={stats['linked']:5d}, copied={stats['copied']:5d}, "
            f"existing={stats['existing']:5d}"
        )

    write_data_yaml(output)
    manifest = {
        "source": str(source),
        "seed": args.seed,
        "holdout_fraction": args.holdout_fraction,
        "splits": {name: values["images"] for name, values in split_stats.items()},
        "holdout_images": sorted(holdout_names),
    }
    (output / "split_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"RF-DETR dataset ready: {output}")


if __name__ == "__main__":
    main()
