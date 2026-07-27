"""Cache low-threshold full-image + tiled RF-DETR predictions once."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from .inference_utils import image_files, infer_image, write_prediction_cache
except ImportError:  # Direct execution: python rfdetr/cache_predictions.py
    from inference_utils import image_files, infer_image, write_prediction_cache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("rfdetr_runs/rfdetr_2xl_1360/checkpoint_best_total.pth"),
    )
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=1360)
    parser.add_argument("--min-confidence", type=float, default=0.01)
    parser.add_argument("--tile-size", type=int, default=720)
    parser.add_argument("--tile-overlap", type=int, default=160)
    parser.add_argument("--nms-iou", type=float, default=0.65)
    parser.add_argument("--no-tiles", action="store_true")
    parser.add_argument("--accept-pml", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.accept_pml:
        raise SystemExit("Rerun with --accept-pml after accepting PML-1.0.")
    if not 0.0 <= args.min_confidence <= 1.0:
        raise ValueError("--min-confidence must be between 0 and 1")

    try:
        from rfdetr_plus import RFDETR2XLarge
    except ImportError as exc:
        raise SystemExit(
            'RF-DETR+ is not installed. Install with: '
            'pip install "rfdetr[train,augment,plus]"'
        ) from exc

    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    images = image_files(args.images.resolve())
    if not images:
        raise RuntimeError(f"No images found in {args.images.resolve()}")

    model = RFDETR2XLarge(
        pretrain_weights=str(checkpoint),
        resolution=args.resolution,
        accept_platform_model_license=True,
    )

    def records():
        for index, image_path in enumerate(images, start=1):
            detections = infer_image(
                model,
                image_path,
                resolution=args.resolution,
                minimum_confidence=args.min_confidence,
                tile_size=args.tile_size,
                tile_overlap=args.tile_overlap,
                nms_iou=args.nms_iou,
                use_tiles=not args.no_tiles,
            )
            if index == 1 or index % 25 == 0 or index == len(images):
                print(f"[{index:4d}/{len(images)}] {image_path.name}: {len(detections)}")
            yield image_path.name, detections

    write_prediction_cache(args.output.resolve(), records())
    print(f"Prediction cache written: {args.output.resolve()}")


if __name__ == "__main__":
    main()
