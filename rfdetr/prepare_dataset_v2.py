"""Build the higher-accuracy 8000/500/500 split without holdout leakage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .prepare_dataset import (
        IMAGE_SUFFIXES,
        balanced_holdout,
        prepare_images,
        write_data_yaml,
    )
except ImportError:  # Direct execution: python rfdetr/prepare_dataset_v2.py
    from prepare_dataset import (
        IMAGE_SUFFIXES,
        balanced_holdout,
        prepare_images,
        write_data_yaml,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("bdd100k_selected"))
    parser.add_argument("--base-dataset", type=Path, default=Path("rfdetr_dataset"))
    parser.add_argument("--output", type=Path, default=Path("rfdetr_dataset_v2"))
    parser.add_argument("--valid-count", type=int, default=500)
    parser.add_argument("--seed", type=int, default=2027)
    return parser.parse_args()


def add_stats(*rows: dict[str, int]) -> dict[str, int]:
    return {
        key: sum(row[key] for row in rows)
        for key in ("images", "labels", "linked", "copied", "existing")
    }


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    base_dataset = args.base_dataset.resolve()
    output = args.output.resolve()
    if output in (source, base_dataset):
        raise ValueError("--output must differ from --source and --base-dataset")

    manifest_path = base_dataset / "split_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Missing {manifest_path}; prepare the original leak-free split first"
        )
    base_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    untouched_holdout = set(base_manifest.get("holdout_images", []))
    if len(untouched_holdout) != 500:
        raise ValueError(
            f"Expected the original untouched 500-image holdout, got "
            f"{len(untouched_holdout)}"
        )

    val_images = sorted(
        path
        for path in (source / "images" / "val").iterdir()
        if path.suffix.lower() in IMAGE_SUFFIXES
    )
    candidates = [path for path in val_images if path.name not in untouched_holdout]
    if not 100 <= args.valid_count < len(candidates):
        raise ValueError("--valid-count must leave labeled validation images for training")

    valid_names = balanced_holdout(
        candidates,
        source / "labels" / "val",
        args.valid_count / len(candidates),
        args.seed,
    )
    promoted_to_train = [path for path in candidates if path.name not in valid_names]
    valid_images = [path for path in candidates if path.name in valid_names]
    holdout_images = [path for path in val_images if path.name in untouched_holdout]

    split_names = {
        "promoted_train": {path.name for path in promoted_to_train},
        "valid": {path.name for path in valid_images},
        "holdout": {path.name for path in holdout_images},
    }
    if any(
        split_names[left].intersection(split_names[right])
        for left, right in (
            ("promoted_train", "valid"),
            ("promoted_train", "holdout"),
            ("valid", "holdout"),
        )
    ):
        raise RuntimeError("The v2 train/valid/holdout split overlaps")

    output.mkdir(parents=True, exist_ok=True)
    original_train = prepare_images(source, output, "train", "train")
    promoted = prepare_images(
        source,
        output,
        "val",
        "train",
        promoted_to_train,
        allow_existing_extra=True,
    )
    stats = {
        "train": add_stats(original_train, promoted),
        "valid": prepare_images(source, output, "val", "valid", valid_images),
        "holdout": prepare_images(
            source, output, "val", "holdout", holdout_images
        ),
        "test": prepare_images(source, output, "test", "test"),
    }
    write_data_yaml(output)

    expected = {"train": 8000, "valid": 500, "holdout": 500, "test": 1000}
    actual = {name: values["images"] for name, values in stats.items()}
    if actual != expected:
        raise RuntimeError(f"Unexpected v2 split sizes: {actual}; expected {expected}")

    manifest = {
        "source": str(source),
        "base_manifest": str(manifest_path),
        "seed": args.seed,
        "splits": actual,
        "holdout_policy": "identical to original untouched holdout",
        "valid_images": sorted(valid_names),
        "holdout_images": sorted(untouched_holdout),
        "promoted_to_train": sorted(split_names["promoted_train"]),
    }
    (output / "split_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    for name, values in stats.items():
        print(
            f"{name:7s}: images={values['images']:5d}, "
            f"labels={values['labels']:5d}, linked={values['linked']:5d}, "
            f"copied={values['copied']:5d}, existing={values['existing']:5d}"
        )
    print(f"RF-DETR v2 dataset ready: {output}")


if __name__ == "__main__":
    main()
