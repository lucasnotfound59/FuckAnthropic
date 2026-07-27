"""Accuracy-first RF-DETR-2XL training entry point for BDD100K."""

from __future__ import annotations

import argparse
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

# Driving scenes should remain almost upright. These augmentations target
# weather/camera variation without crops or large rotations that erase tiny
# traffic lights and signs.
BDD_AUGMENTATIONS = {
    "HorizontalFlip": {"p": 0.5},
    "RandomBrightnessContrast": {
        "brightness_limit": 0.20,
        "contrast_limit": 0.20,
        "p": 0.45,
    },
    "HueSaturationValue": {
        "hue_shift_limit": 6,
        "sat_shift_limit": 16,
        "val_shift_limit": 14,
        "p": 0.30,
    },
    "GaussianBlur": {"blur_limit": [3, 5], "p": 0.06},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("rfdetr_dataset"))
    parser.add_argument("--output", type=Path, default=Path("rfdetr_runs/rfdetr_2xl_1360"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument(
        "--resolution",
        type=int,
        default=1360,
        help="1360 is divisible by the 2XL patch/window factor (40) and favors tiny objects.",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--lr", type=float, default=8e-5)
    parser.add_argument("--lr-encoder", type=float, default=4e-5)
    parser.add_argument("--warmup-epochs", type=float, default=3.0)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--drop-path", type=float, default=0.10)
    parser.add_argument("--smooth-alpha", type=float, default=0.02)
    parser.add_argument("--skip-best-epochs", type=int, default=5)
    parser.add_argument("--target-f1", type=float, default=0.95)
    parser.add_argument(
        "--pretrain-weights",
        type=Path,
        default=None,
        help="Start a fresh fine-tuning stage from a model-only .pth checkpoint.",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=5,
        help="Save periodic checkpoints every N epochs; 5 avoids filling a 50GB AutoDL disk.",
    )
    parser.add_argument(
        "--accept-pml",
        action="store_true",
        help="Confirm acceptance of the RF-DETR+ Platform Model License 1.0.",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Training checkpoint to resume, normally output/checkpoint.pth.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.accept_pml:
        raise SystemExit(
            "RF-DETR-2XL uses the Platform Model License 1.0. "
            "Read it first, then rerun with --accept-pml to confirm acceptance."
        )
    if args.resume is not None and args.pretrain_weights is not None:
        raise ValueError("--resume and --pretrain-weights are mutually exclusive")
    if not 0.0 < args.target_f1 <= 1.0:
        raise ValueError("--target-f1 must be in (0, 1]")

    dataset = args.dataset.resolve()
    output = args.output.resolve()
    if not (dataset / "data.yaml").is_file():
        raise FileNotFoundError(
            f"Missing {dataset / 'data.yaml'}. Run rfdetr/prepare_dataset.py first."
        )

    try:
        from rfdetr_plus import RFDETR2XLarge
    except ImportError as exc:
        raise SystemExit(
            'RF-DETR+ is not installed. Install with: pip install "rfdetr[train,plus]"'
        ) from exc

    # RF-DETR 1.8 keeps the checkpoint's 90-class head unless the target class
    # count is explicit. Fix it at dataset construction time so the classifier
    # is correctly reinitialized for the nine BDD100K competition classes.
    model_options = {
        "accept_platform_model_license": True,
        "num_classes": len(CLASS_NAMES),
    }
    if args.pretrain_weights is not None:
        pretrain_weights = args.pretrain_weights.resolve()
        if not pretrain_weights.is_file():
            raise FileNotFoundError(pretrain_weights)
        model_options["pretrain_weights"] = str(pretrain_weights)
    model = RFDETR2XLarge(
        **model_options,
    )
    model.train(
        dataset_file="yolo",
        dataset_dir=str(dataset),
        class_names=CLASS_NAMES,
        output_dir=str(output),
        epochs=args.epochs,
        resolution=args.resolution,
        # Auto-batch probing found micro-batch 5 safe at 1360px with a
        # conservative 100-target frame. Use 4 x 4 to preserve the intended
        # effective batch of 16 while retaining headroom for the 91-object max.
        batch_size=4,
        grad_accum_steps=4,
        lr=args.lr,
        lr_encoder=args.lr_encoder,
        warmup_epochs=args.warmup_epochs,
        lr_scheduler="cosine",
        lr_min_factor=0.01,
        weight_decay=args.weight_decay,
        drop_path=args.drop_path,
        smooth_alpha=args.smooth_alpha,
        checkpoint_interval=args.checkpoint_interval,
        skip_best_epochs=args.skip_best_epochs,
        eval_interval=1,
        eval_max_dets=500,
        log_per_class_metrics=True,
        use_ema=True,
        early_stopping=True,
        early_stopping_patience=args.patience,
        early_stopping_min_delta=0.001,
        early_stopping_use_ema=True,
        multi_scale=False,
        expanded_scales=False,
        aug_config=BDD_AUGMENTATIONS,
        augmentation_backend="auto",
        save_dataset_grids=True,
        amp_dtype="auto",
        num_workers=args.workers,
        seed=args.seed,
        tensorboard=True,
        progress_bar="rich",
        resume=str(args.resume.resolve()) if args.resume else None,
        notes={
            "model": "RF-DETR-2XLarge",
            "dataset": "BDD100K selected, 9 classes",
            "objective": f"holdout count-F1 >= {args.target_f1:.2f}",
            "resolution": args.resolution,
            "target_f1": args.target_f1,
            "optimization": {
                "lr": args.lr,
                "lr_encoder": args.lr_encoder,
                "warmup_epochs": args.warmup_epochs,
                "smooth_alpha": args.smooth_alpha,
            },
            "anti_overfit": {
                "independent_holdout": True,
                "ema": True,
                "early_stopping_patience": args.patience,
                "drop_path": args.drop_path,
                "weight_decay": args.weight_decay,
            },
        },
    )

    print(f"Training finished. Checkpoints: {output}")
    print(f"mAP-selected checkpoint: {output / 'checkpoint_best_total.pth'}")
    print("Re-rank saved checkpoints with holdout count-F1 before final submission.")


if __name__ == "__main__":
    main()
