"""Tune per-class thresholds for the competition's count-based micro F1."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    from .inference_utils import CLASS_NAMES, Detection, read_prediction_cache
except ImportError:  # Direct execution: python rfdetr/tune_thresholds.py
    from inference_utils import CLASS_NAMES, Detection, read_prediction_cache


@dataclass(frozen=True)
class Counts:
    tp: int
    fp: int
    fn: int

    def __add__(self, other: "Counts") -> "Counts":
        return Counts(self.tp + other.tp, self.fp + other.fp, self.fn + other.fn)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration-cache", type=Path, required=True)
    parser.add_argument("--calibration-labels", type=Path, required=True)
    parser.add_argument("--holdout-cache", type=Path)
    parser.add_argument("--holdout-labels", type=Path)
    parser.add_argument("--output", type=Path, default=Path("rfdetr/thresholds.json"))
    parser.add_argument("--minimum", type=float, default=0.02)
    parser.add_argument("--maximum", type=float, default=0.80)
    parser.add_argument("--step", type=float, default=0.01)
    parser.add_argument(
        "--objective",
        choices=("micro", "macro"),
        default="micro",
        help="Use the competition's exact definition if it is known.",
    )
    return parser.parse_args()


def ground_truth_counts(labels_dir: Path) -> dict[str, list[int]]:
    result: dict[str, list[int]] = {}
    for path in sorted(labels_dir.glob("*.txt")):
        counts = [0] * len(CLASS_NAMES)
        for line in path.read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if not fields:
                continue
            class_id = int(fields[0])
            if not 0 <= class_id < len(CLASS_NAMES):
                raise ValueError(f"Invalid class {class_id} in {path}")
            counts[class_id] += 1
        result[f"{path.stem}.jpg"] = counts
    return result


def f1(counts: Counts) -> float:
    denominator = 2 * counts.tp + counts.fp + counts.fn
    return 2 * counts.tp / denominator if denominator else 1.0


def precision(counts: Counts) -> float:
    denominator = counts.tp + counts.fp
    return counts.tp / denominator if denominator else 1.0


def recall(counts: Counts) -> float:
    denominator = counts.tp + counts.fn
    return counts.tp / denominator if denominator else 1.0


def stats_by_class_and_threshold(
    predictions: dict[str, list[Detection]],
    ground_truth: dict[str, list[int]],
    thresholds: np.ndarray,
) -> list[list[Counts]]:
    missing = sorted(set(ground_truth) - set(predictions))
    extra = sorted(set(predictions) - set(ground_truth))
    if missing or extra:
        raise ValueError(
            f"Prediction/label image mismatch: missing={len(missing)}, extra={len(extra)}"
        )

    scores: dict[str, list[list[float]]] = {}
    for image_name, detections in predictions.items():
        by_class = [[] for _ in CLASS_NAMES]
        for detection in detections:
            by_class[detection.class_id].append(detection.score)
        scores[image_name] = by_class

    all_stats: list[list[Counts]] = []
    for class_id in range(len(CLASS_NAMES)):
        class_stats: list[Counts] = []
        for threshold in thresholds:
            tp = fp = fn = 0
            for image_name, truth in ground_truth.items():
                predicted_count = sum(
                    score >= threshold for score in scores[image_name][class_id]
                )
                truth_count = truth[class_id]
                tp += min(predicted_count, truth_count)
                fp += max(predicted_count - truth_count, 0)
                fn += max(truth_count - predicted_count, 0)
            class_stats.append(Counts(tp, fp, fn))
        all_stats.append(class_stats)
    return all_stats


def optimize_micro_f1(all_stats: list[list[Counts]]) -> list[int]:
    selected = [
        max(range(len(class_stats)), key=lambda index: f1(class_stats[index]))
        for class_stats in all_stats
    ]
    for _ in range(10):
        changed = False
        for class_id, class_stats in enumerate(all_stats):
            base = Counts(0, 0, 0)
            for other_class, other_stats in enumerate(all_stats):
                if other_class != class_id:
                    base += other_stats[selected[other_class]]
            best_index = max(
                range(len(class_stats)),
                key=lambda index: f1(base + class_stats[index]),
            )
            if best_index != selected[class_id]:
                selected[class_id] = best_index
                changed = True
        if not changed:
            break
    return selected


def evaluate_selected(
    all_stats: list[list[Counts]], selected: list[int]
) -> tuple[Counts, list[Counts]]:
    per_class = [
        class_stats[selected[class_id]]
        for class_id, class_stats in enumerate(all_stats)
    ]
    total = Counts(0, 0, 0)
    for counts in per_class:
        total += counts
    return total, per_class


def print_report(
    name: str,
    thresholds: np.ndarray,
    selected: list[int],
    all_stats: list[list[Counts]],
) -> dict[str, float]:
    total, per_class = evaluate_selected(all_stats, selected)
    print(f"\n{name}")
    print("class             conf  precision  recall      F1")
    print("-" * 56)
    for class_name, index, counts in zip(
        CLASS_NAMES, selected, per_class, strict=True
    ):
        print(
            f"{class_name:16s} {thresholds[index]:.2f}  "
            f"{precision(counts):9.4f}  {recall(counts):6.4f}  {f1(counts):6.4f}"
        )
    macro_f1 = sum(f1(counts) for counts in per_class) / len(per_class)
    print("-" * 56)
    print(
        f"micro: P={precision(total):.4f} R={recall(total):.4f} F1={f1(total):.4f}; "
        f"macro-F1={macro_f1:.4f}"
    )
    return {
        "precision": precision(total),
        "recall": recall(total),
        "micro_f1": f1(total),
        "macro_f1": macro_f1,
    }


def main() -> None:
    args = parse_args()
    if args.step <= 0 or args.minimum < 0 or args.maximum > 1:
        raise ValueError("Threshold range must be within [0, 1] with a positive step")
    if args.minimum >= args.maximum:
        raise ValueError("--minimum must be less than --maximum")
    if bool(args.holdout_cache) != bool(args.holdout_labels):
        raise ValueError("Pass both --holdout-cache and --holdout-labels, or neither")

    thresholds = np.arange(
        args.minimum, args.maximum + args.step / 2.0, args.step, dtype=np.float64
    )
    calibration_predictions = read_prediction_cache(args.calibration_cache.resolve())
    calibration_truth = ground_truth_counts(args.calibration_labels.resolve())
    calibration_stats = stats_by_class_and_threshold(
        calibration_predictions, calibration_truth, thresholds
    )
    if args.objective == "micro":
        selected = optimize_micro_f1(calibration_stats)
    else:
        selected = [
            max(range(len(class_stats)), key=lambda index: f1(class_stats[index]))
            for class_stats in calibration_stats
        ]
    calibration_report = print_report(
        "CALIBRATION (used to choose thresholds)",
        thresholds,
        selected,
        calibration_stats,
    )

    holdout_report = None
    if args.holdout_cache:
        holdout_predictions = read_prediction_cache(args.holdout_cache.resolve())
        holdout_truth = ground_truth_counts(args.holdout_labels.resolve())
        overlap = set(calibration_predictions).intersection(holdout_predictions)
        if overlap:
            raise ValueError(
                f"Calibration and holdout overlap by {len(overlap)} images; "
                "this would leak tuning decisions."
            )
        holdout_stats = stats_by_class_and_threshold(
            holdout_predictions, holdout_truth, thresholds
        )
        holdout_report = print_report(
            "HOLDOUT (never used to choose thresholds)",
            thresholds,
            selected,
            holdout_stats,
        )
        gap = calibration_report["micro_f1"] - holdout_report["micro_f1"]
        if gap > 0.03:
            print(
                f"\nWARNING: calibration-to-holdout F1 gap is {gap:.4f}. "
                "Treat this as post-processing overfit."
            )
        if holdout_report["micro_f1"] >= 0.95:
            print("\nTarget reached honestly: holdout count-F1 >= 0.95.")
        else:
            print(
                f"\nTarget not yet reached: holdout count-F1="
                f"{holdout_report['micro_f1']:.4f} < 0.95."
            )

    payload: dict[str, object] = {
        class_name: round(float(thresholds[selected[class_id]]), 6)
        for class_id, class_name in enumerate(CLASS_NAMES)
    }
    payload["_meta"] = {
        "metric": f"count-based {args.objective} F1",
        "calibration": calibration_report,
        "holdout": holdout_report,
        "warning": "Only the holdout score is an unbiased threshold estimate.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nThresholds written: {args.output.resolve()}")


if __name__ == "__main__":
    main()
