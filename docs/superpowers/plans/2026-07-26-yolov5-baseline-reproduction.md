# YOLOv5 Baseline Reproduction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reproduce the supplied YOLOv5s BDD100K baseline by auditing the full dataset, completing a one-epoch local smoke run, reloading its checkpoint for validation, and documenting the exact 30-epoch CUDA command.

**Architecture:** Keep the supplied detector, loss, hyperparameters, and metric implementation authoritative. Add a thin reproduction layer around it: canonical path resolution, a deterministic dataset audit, a generated smoke split, pinned local dependencies, and explicit run commands. Generated datasets, environments, caches, checkpoints, and plots remain outside version control.

**Tech Stack:** Python 3.11, PyTorch 2.5.0, torchvision 0.20.0, YOLOv5-style baseline code, PyYAML, Pillow, pytest, uv.

## Global Constraints

- The strict baseline uses random initialization, YOLOv5s, 30 epochs, 640 × 640 input, SGD, batch size 16 where memory permits, seed 0, and the supplied `hyper_parameter.yaml`.
- The competition's primary metric is `F1 = 2 × precision × recall / (precision + recall)`; every standalone validation report must include this aggregate F1.
- Keep the baseline's existing mAP-dominated checkpoint fitness unchanged during strict reproduction.
- Do not tune the architecture, augmentation, loss, optimizer, or hyperparameters during reproduction.
- Do not modify source images or labels.
- The smoke run may use 320 × 320 input and a smaller batch because its metrics are not a quality measurement.
- The test split has no labels; submission-format repair is outside this plan.
- The local host has no CUDA or MPS device exposed, so the smoke run uses CPU and the full run is handed off to CUDA.

## File Map

- `26SummerHackathon_GenAI/baseline/train.py`: resolve the dataset through the baseline's canonical dataset checker.
- `26SummerHackathon_GenAI/baseline/dataset/dataset.yaml`: describe the repository-root BDD100K dataset independently of the current working directory.
- `26SummerHackathon_GenAI/baseline/dataset/smoke.yaml`: describe the generated miniature smoke dataset.
- `26SummerHackathon_GenAI/baseline/repro/requirements-common.txt`: pin non-PyTorch reproduction dependencies.
- `26SummerHackathon_GenAI/baseline/repro/requirements-local.txt`: pin the local CPU PyTorch stack.
- `26SummerHackathon_GenAI/baseline/repro/audit_dataset.py`: validate counts, pairing, label syntax, box ranges, and optional JPEG readability.
- `26SummerHackathon_GenAI/baseline/repro/create_smoke_dataset.py`: deterministically link selected source examples into an ignored smoke tree.
- `26SummerHackathon_GenAI/baseline/repro/README.md`: record local and CUDA commands plus completion criteria.
- `26SummerHackathon_GenAI/baseline/tests/test_dataset_config.py`: verify path resolution from arbitrary working directories.
- `26SummerHackathon_GenAI/baseline/tests/test_audit_dataset.py`: verify valid and invalid audit cases.
- `26SummerHackathon_GenAI/baseline/tests/test_create_smoke_dataset.py`: verify deterministic, idempotent smoke generation.
- `26SummerHackathon_GenAI/baseline/.gitignore`: ignore the environment and generated reproduction artifacts.

---

### Task 1: Reproducible Environment and Canonical Dataset Configuration

**Files:**
- Create: `26SummerHackathon_GenAI/baseline/repro/requirements-common.txt`
- Create: `26SummerHackathon_GenAI/baseline/repro/requirements-local.txt`
- Modify: `26SummerHackathon_GenAI/baseline/dataset/dataset.yaml`
- Modify: `26SummerHackathon_GenAI/baseline/train.py:35-55`
- Modify: `26SummerHackathon_GenAI/baseline/train.py:159`
- Create: `26SummerHackathon_GenAI/baseline/tests/test_dataset_config.py`

**Interfaces:**
- Consumes: repository dataset at `bdd100k_selected`.
- Produces: `check_dataset(path, autodownload=False) -> dict` with absolute `train`, `val`, and `test` paths.

- [ ] **Step 1: Write the failing dataset-configuration test**

```python
# 26SummerHackathon_GenAI/baseline/tests/test_dataset_config.py
from pathlib import Path

from utils.general import check_dataset


BASELINE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = BASELINE_ROOT.parents[1]


def test_dataset_yaml_resolves_from_an_arbitrary_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    data = check_dataset(BASELINE_ROOT / "dataset" / "dataset.yaml", autodownload=False)

    assert Path(data["train"]) == REPOSITORY_ROOT / "bdd100k_selected" / "images" / "train"
    assert Path(data["val"]) == REPOSITORY_ROOT / "bdd100k_selected" / "images" / "val"
    assert Path(data["test"]) == REPOSITORY_ROOT / "bdd100k_selected" / "images" / "test"
    assert data["nc"] == 9
    assert list(data["names"].values()) == [
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
```

- [ ] **Step 2: Create and populate the Python 3.11 environment**

```bash
cd 26SummerHackathon_GenAI/baseline
uv venv --python /opt/anaconda3/bin/python3.11 .venv
uv pip install --python .venv/bin/python -r repro/requirements-local.txt
```

Create `repro/requirements-common.txt`:

```text
gitpython==3.1.43
matplotlib==3.9.2
numpy==1.26.4
opencv-python==4.10.0.84
pandas==2.2.3
pillow==10.4.0
psutil==6.1.0
pytest==8.3.4
PyYAML==6.0.2
requests==2.32.3
scipy==1.14.1
seaborn==0.13.2
setuptools==75.6.0
thop==0.1.1.post2209072238
tqdm==4.67.0
ultralytics==8.2.34
```

Create `repro/requirements-local.txt`:

```text
-r requirements-common.txt
torch==2.5.0
torchvision==0.20.0
```

Expected: `.venv/bin/python --version` prints Python 3.11.x and installation completes without changing a global Python environment.

- [ ] **Step 3: Run the test to verify the current path configuration fails**

Run:

```bash
.venv/bin/python -m pytest tests/test_dataset_config.py -v
```

Expected: FAIL because the existing YAML resolves `dataset/bdd10k_selected`, which does not exist.

- [ ] **Step 4: Make the dataset YAML repository-relative**

Replace `dataset/dataset.yaml` with:

```yaml
path: ../..
train: bdd100k_selected/images/train
val: bdd100k_selected/images/val
test: bdd100k_selected/images/test

names:
  0: person
  1: rider
  2: car
  3: truck
  4: bus
  5: motorcycle
  6: bicycle
  7: traffic light
  8: traffic sign
```

The baseline's `check_dataset` resolves relative `path` values from the baseline root, so `../..` is the repository root.

- [ ] **Step 5: Route training through canonical path resolution**

Add `check_dataset` to the imports from `utils.general` in `train.py`, then replace:

```python
data_dict = yaml_load(opt.data)
```

with:

```python
data_dict = check_dataset(opt.data, autodownload=False)
```

Do not change model construction or training hyperparameters.

- [ ] **Step 6: Run the focused and syntax tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_dataset_config.py -v
.venv/bin/python -m py_compile train.py val.py
```

Expected: the pytest case passes and both entry points compile.

- [ ] **Step 7: Commit**

Run from the repository root:

```bash
git add 26SummerHackathon_GenAI/baseline/repro/requirements-common.txt \
  26SummerHackathon_GenAI/baseline/repro/requirements-local.txt \
  26SummerHackathon_GenAI/baseline/dataset/dataset.yaml \
  26SummerHackathon_GenAI/baseline/train.py \
  26SummerHackathon_GenAI/baseline/tests/test_dataset_config.py
git commit -m "fix: make baseline dataset configuration reproducible"
```

---

### Task 2: Dataset Integrity Audit

**Files:**
- Create: `26SummerHackathon_GenAI/baseline/repro/audit_dataset.py`
- Create: `26SummerHackathon_GenAI/baseline/tests/test_audit_dataset.py`

**Interfaces:**
- Consumes: `audit_dataset(root: Path, expectations: dict[str, SplitExpectation], verify_images: bool)`.
- Produces: `AuditResult(counts: dict[str, tuple[int, int]], errors: tuple[str, ...])` and CLI exit code 0 only for a valid dataset.

- [ ] **Step 1: Write failing audit tests**

```python
# 26SummerHackathon_GenAI/baseline/tests/test_audit_dataset.py
from pathlib import Path

from PIL import Image

from repro.audit_dataset import AuditResult, SplitExpectation, audit_dataset


def write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 18), color="black").save(path)


def write_label(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def miniature_expectations() -> dict[str, SplitExpectation]:
    return {
        "train": SplitExpectation(images=1, labels=1),
        "val": SplitExpectation(images=1, labels=1),
        "test": SplitExpectation(images=1, labels=0),
    }


def test_valid_dataset_passes(tmp_path):
    for split in ("train", "val", "test"):
        write_image(tmp_path / "images" / split / f"{split}.jpg")
    write_label(tmp_path / "labels" / "train" / "train.txt", "2 0.5 0.5 0.4 0.4\n")
    write_label(tmp_path / "labels" / "val" / "val.txt", "0 0.5 0.5 0.2 0.3\n")

    result = audit_dataset(tmp_path, miniature_expectations(), verify_images=True)

    assert result == AuditResult(
        counts={"train": (1, 1), "val": (1, 1), "test": (1, 0)},
        errors=(),
    )


def test_invalid_class_and_box_report_file_and_line(tmp_path):
    for split in ("train", "val", "test"):
        write_image(tmp_path / "images" / split / f"{split}.jpg")
    write_label(tmp_path / "labels" / "train" / "train.txt", "9 0.5 0.5 -0.2 0.4\n")
    write_label(tmp_path / "labels" / "val" / "val.txt", "0 0.5 0.5 0.2 0.3\n")

    result = audit_dataset(tmp_path, miniature_expectations(), verify_images=False)

    assert any("train.txt:1" in error and "class 9" in error for error in result.errors)
    assert any("train.txt:1" in error and "width and height must be positive" in error for error in result.errors)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
.venv/bin/python -m pytest tests/test_audit_dataset.py -v
```

Expected: FAIL because `repro.audit_dataset` does not exist.

- [ ] **Step 3: Implement the audit**

```python
# 26SummerHackathon_GenAI/baseline/repro/audit_dataset.py
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

from PIL import Image


@dataclass(frozen=True)
class SplitExpectation:
    images: int
    labels: int


@dataclass(frozen=True)
class AuditResult:
    counts: dict[str, tuple[int, int]]
    errors: tuple[str, ...]


DEFAULT_EXPECTATIONS = {
    "train": SplitExpectation(images=7000, labels=7000),
    "val": SplitExpectation(images=2000, labels=2000),
    "test": SplitExpectation(images=1000, labels=0),
}


def _files(path: Path, suffix: str) -> list[Path]:
    return sorted(item for item in path.glob(f"*{suffix}") if item.is_file())


def _audit_label(path: Path) -> list[str]:
    errors: list[str] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw_line.strip():
            continue
        fields = raw_line.split()
        location = f"{path}:{line_number}"
        if len(fields) != 5:
            errors.append(f"{location}: expected 5 fields, found {len(fields)}")
            continue
        try:
            values = [float(field) for field in fields]
        except ValueError:
            errors.append(f"{location}: all fields must be numeric")
            continue
        class_value, x_center, y_center, width, height = values
        if not all(math.isfinite(value) for value in values):
            errors.append(f"{location}: all values must be finite")
            continue
        if not class_value.is_integer() or not 0 <= int(class_value) <= 8:
            errors.append(f"{location}: class {fields[0]} is outside integer range 0-8")
        if not all(0.0 <= value <= 1.0 for value in (x_center, y_center, width, height)):
            errors.append(f"{location}: normalized box values must be within 0-1")
        if width <= 0.0 or height <= 0.0:
            errors.append(f"{location}: width and height must be positive")
    return errors


def audit_dataset(
    root: Path,
    expectations: dict[str, SplitExpectation] = DEFAULT_EXPECTATIONS,
    verify_images: bool = False,
) -> AuditResult:
    root = root.resolve()
    counts: dict[str, tuple[int, int]] = {}
    errors: list[str] = []

    for split, expected in expectations.items():
        images = _files(root / "images" / split, ".jpg")
        labels = _files(root / "labels" / split, ".txt")
        counts[split] = (len(images), len(labels))
        if len(images) != expected.images:
            errors.append(f"{split}: expected {expected.images} images, found {len(images)}")
        if len(labels) != expected.labels:
            errors.append(f"{split}: expected {expected.labels} labels, found {len(labels)}")

        image_stems = {path.stem for path in images}
        label_stems = {path.stem for path in labels}
        if expected.labels:
            for stem in sorted(image_stems - label_stems):
                errors.append(f"{split}: image has no label file: {stem}.jpg")
            for stem in sorted(label_stems - image_stems):
                errors.append(f"{split}: label has no image file: {stem}.txt")

        for label in labels:
            errors.extend(_audit_label(label))

        if verify_images:
            for image in images:
                try:
                    with Image.open(image) as opened:
                        opened.verify()
                except Exception as exc:
                    errors.append(f"{image}: unreadable image: {exc}")

    return AuditResult(counts=counts, errors=tuple(errors))


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit the selected BDD100K dataset.")
    parser.add_argument("root", type=Path)
    parser.add_argument("--verify-images", action="store_true")
    args = parser.parse_args()

    result = audit_dataset(args.root, verify_images=args.verify_images)
    for split, (image_count, label_count) in result.counts.items():
        print(f"{split}: images={image_count} labels={label_count}")
    if result.errors:
        for error in result.errors:
            print(f"ERROR: {error}")
        return 1
    print("Dataset audit passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run focused tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_audit_dataset.py -v
```

Expected: 2 tests pass.

- [ ] **Step 5: Audit all source data and decode every JPEG**

Run from the baseline directory:

```bash
.venv/bin/python repro/audit_dataset.py ../../bdd100k_selected --verify-images
```

Expected:

```text
train: images=7000 labels=7000
val: images=2000 labels=2000
test: images=1000 labels=0
Dataset audit passed.
```

- [ ] **Step 6: Commit**

Run from the repository root:

```bash
git add 26SummerHackathon_GenAI/baseline/repro/audit_dataset.py \
  26SummerHackathon_GenAI/baseline/tests/test_audit_dataset.py
git commit -m "test: add BDD100K dataset integrity audit"
```

---

### Task 3: Deterministic Smoke Dataset

**Files:**
- Create: `26SummerHackathon_GenAI/baseline/repro/create_smoke_dataset.py`
- Create: `26SummerHackathon_GenAI/baseline/tests/test_create_smoke_dataset.py`
- Create: `26SummerHackathon_GenAI/baseline/dataset/smoke.yaml`
- Modify: `26SummerHackathon_GenAI/baseline/.gitignore`

**Interfaces:**
- Consumes: `build_smoke(source: Path, destination: Path, train_count: int, val_count: int)`.
- Produces: an ignored YOLO tree containing symlinks under `images/{train,val}` and `labels/{train,val}`.

- [ ] **Step 1: Write the failing smoke-builder test**

```python
# 26SummerHackathon_GenAI/baseline/tests/test_create_smoke_dataset.py
from pathlib import Path

from repro.create_smoke_dataset import build_smoke


def add_sample(root: Path, split: str, name: str, class_id: int) -> None:
    image = root / "images" / split / f"{name}.jpg"
    label = root / "labels" / split / f"{name}.txt"
    image.parent.mkdir(parents=True, exist_ok=True)
    label.parent.mkdir(parents=True, exist_ok=True)
    image.write_bytes(b"jpeg-placeholder")
    label.write_text(f"{class_id} 0.5 0.5 0.2 0.2\n", encoding="utf-8")


def test_build_smoke_is_deterministic_and_idempotent(tmp_path):
    source = tmp_path / "source"
    destination = tmp_path / "smoke"
    for split in ("train", "val"):
        for class_id in range(9):
            add_sample(source, split, f"{split}-{class_id}", class_id)

    first = build_smoke(source, destination, train_count=9, val_count=9)
    second = build_smoke(source, destination, train_count=9, val_count=9)

    assert first == second
    assert len(list((destination / "images" / "train").glob("*.jpg"))) == 9
    assert len(list((destination / "labels" / "val").glob("*.txt"))) == 9
    assert all(path.is_symlink() for path in destination.rglob("*") if path.is_file())
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
.venv/bin/python -m pytest tests/test_create_smoke_dataset.py -v
```

Expected: FAIL because `repro.create_smoke_dataset` does not exist.

- [ ] **Step 3: Implement deterministic class-covering selection and linking**

```python
# 26SummerHackathon_GenAI/baseline/repro/create_smoke_dataset.py
from __future__ import annotations

import argparse
from pathlib import Path


def _classes(label: Path) -> set[int]:
    result: set[int] = set()
    for line in label.read_text(encoding="utf-8").splitlines():
        if line.strip():
            result.add(int(float(line.split()[0])))
    return result


def _select(label_directory: Path, limit: int) -> tuple[Path, ...]:
    candidates = sorted(label_directory.glob("*.txt"))
    if len(candidates) < limit:
        raise ValueError(f"{label_directory} contains {len(candidates)} labels, need {limit}")

    selected: list[Path] = []
    remaining = candidates.copy()
    uncovered = set(range(9))
    while remaining and uncovered and len(selected) < limit:
        best = max(remaining, key=lambda path: (len(_classes(path) & uncovered), path.name))
        if not (_classes(best) & uncovered):
            break
        selected.append(best)
        uncovered -= _classes(best)
        remaining.remove(best)

    selected.extend(remaining[: limit - len(selected)])
    return tuple(sorted(selected))


def _link(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        if destination.resolve() == source.resolve():
            return
        raise FileExistsError(f"{destination} points to a different source")
    if destination.exists():
        raise FileExistsError(f"{destination} already exists and is not a symlink")
    destination.symlink_to(source.resolve())


def build_smoke(
    source: Path,
    destination: Path,
    train_count: int = 48,
    val_count: int = 16,
) -> dict[str, tuple[str, ...]]:
    selections: dict[str, tuple[str, ...]] = {}
    for split, limit in (("train", train_count), ("val", val_count)):
        labels = _select(source / "labels" / split, limit)
        selections[split] = tuple(path.stem for path in labels)
        for label in labels:
            image = source / "images" / split / f"{label.stem}.jpg"
            if not image.is_file():
                raise FileNotFoundError(image)
            _link(image, destination / "images" / split / image.name)
            _link(label, destination / "labels" / split / label.name)
    return selections


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a deterministic BDD100K smoke split.")
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--train-count", type=int, default=48)
    parser.add_argument("--val-count", type=int, default=16)
    args = parser.parse_args()

    selections = build_smoke(args.source, args.destination, args.train_count, args.val_count)
    for split, names in selections.items():
        print(f"{split}: {len(names)} linked examples")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Add the generated-dataset configuration**

Create `dataset/smoke.yaml`:

```yaml
path: .repro/smoke
train: images/train
val: images/val

names:
  0: person
  1: rider
  2: car
  3: truck
  4: bus
  5: motorcycle
  6: bicycle
  7: traffic light
  8: traffic sign
```

Append to `.gitignore`:

```gitignore
.venv/
.repro/
```

- [ ] **Step 5: Run focused tests and build the real smoke split**

Run:

```bash
.venv/bin/python -m pytest tests/test_create_smoke_dataset.py -v
.venv/bin/python repro/create_smoke_dataset.py ../../bdd100k_selected .repro/smoke
find .repro/smoke/images/train -type l | wc -l
find .repro/smoke/images/val -type l | wc -l
```

Expected: pytest passes, the builder prints 48 train and 16 validation examples, and the two counts print `48` and `16`.

- [ ] **Step 6: Run all reproduction unit tests**

Run:

```bash
.venv/bin/python -m pytest tests -v
```

Expected: all tests pass.

- [ ] **Step 7: Commit**

Run from the repository root:

```bash
git add 26SummerHackathon_GenAI/baseline/repro/create_smoke_dataset.py \
  26SummerHackathon_GenAI/baseline/tests/test_create_smoke_dataset.py \
  26SummerHackathon_GenAI/baseline/dataset/smoke.yaml \
  26SummerHackathon_GenAI/baseline/.gitignore
git commit -m "test: add deterministic baseline smoke dataset"
```

---

### Task 4: Execute and Verify the Local Smoke Run

**Files:**
- Generated: `26SummerHackathon_GenAI/baseline/.repro/runs/smoke/**`
- Generated: `26SummerHackathon_GenAI/baseline/.repro/validation/smoke/**`

**Interfaces:**
- Consumes: `dataset/smoke.yaml`, existing YOLOv5s model, supplied hyperparameters.
- Produces: reloadable `.repro/runs/smoke/weights/best.pt` and standalone validation metrics.

- [ ] **Step 1: Run one CPU epoch**

Run from `26SummerHackathon_GenAI/baseline`:

```bash
.venv/bin/python train.py \
  --data dataset/smoke.yaml \
  --cfg yolov5s.yaml \
  --weights '' \
  --hyp hyper_parameter.yaml \
  --epochs 1 \
  --batch-size 2 \
  --imgsz 320 \
  --workers 0 \
  --device cpu \
  --seed 0 \
  --project .repro/runs \
  --name smoke \
  --exist-ok
```

Expected: exit code 0, finite box/object/class losses, one validation pass, and both checkpoint files.

- [ ] **Step 2: Assert checkpoint artifacts**

Run:

```bash
test -s .repro/runs/smoke/weights/last.pt
test -s .repro/runs/smoke/weights/best.pt
```

Expected: both commands exit 0.

- [ ] **Step 3: Reload `best.pt` in a standalone validation process**

Run:

```bash
.venv/bin/python val.py \
  --data dataset/smoke.yaml \
  --weights .repro/runs/smoke/weights/best.pt \
  --batch-size 2 \
  --imgsz 320 \
  --workers 0 \
  --device cpu \
  --verbose \
  --project .repro/validation \
  --name smoke \
  --exist-ok
```

Expected: exit code 0 and output containing overall P, R, mAP50, and mAP50-95. Compute aggregate F1 from the reported P and R with `2 * P * R / (P + R)`, treating `P + R == 0` as F1 0. Low values are acceptable for a one-epoch smoke run.

- [ ] **Step 4: Record the local environment**

Run:

```bash
.venv/bin/python -m torch.utils.collect_env > .repro/runs/smoke/environment.txt
uv pip freeze --python .venv/bin/python > .repro/runs/smoke/pip-freeze.txt
```

Expected: both metadata files are non-empty.

- [ ] **Step 5: Confirm generated outputs remain ignored**

Run from the repository root:

```bash
git status --short
```

Expected: no `.venv` or `.repro` files appear.

---

### Task 5: Document the Full CUDA Reproduction

**Files:**
- Create: `26SummerHackathon_GenAI/baseline/repro/README.md`

**Interfaces:**
- Consumes: a Linux CUDA host, repository checkout, and the audited dataset.
- Produces: exact 30-epoch command, explicit success criteria, environment capture, and final validation command.

- [ ] **Step 1: Write the reproduction runbook**

````markdown
# Baseline Reproduction

Run commands from `26SummerHackathon_GenAI/baseline`.

## Local smoke run

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r repro/requirements-local.txt
.venv/bin/python repro/audit_dataset.py ../../bdd100k_selected --verify-images
.venv/bin/python repro/create_smoke_dataset.py ../../bdd100k_selected .repro/smoke
.venv/bin/python train.py --data dataset/smoke.yaml --cfg yolov5s.yaml \
  --weights '' --hyp hyper_parameter.yaml --epochs 1 --batch-size 2 \
  --imgsz 320 --workers 0 --device cpu --seed 0 \
  --project .repro/runs --name smoke --exist-ok
.venv/bin/python val.py --data dataset/smoke.yaml \
  --weights .repro/runs/smoke/weights/best.pt --batch-size 2 --imgsz 320 \
  --workers 0 --device cpu --verbose \
  --project .repro/validation --name smoke --exist-ok
```

## Full CUDA baseline

Create Python 3.11 and install the CUDA build of PyTorch 2.5.0 and torchvision
0.20.0 appropriate for the host, followed by `repro/requirements-common.txt`.
Confirm `python -c "import torch; assert torch.cuda.is_available()"` before
training.

```bash
python repro/audit_dataset.py ../../bdd100k_selected --verify-images
python train.py --data dataset/dataset.yaml --cfg yolov5s.yaml \
  --weights '' --hyp hyper_parameter.yaml --epochs 30 --batch-size 16 \
  --imgsz 640 --workers 8 --device 0 --seed 0 \
  --project results --name baseline-yolov5s-random-seed0 --exist-ok
python val.py --data dataset/dataset.yaml \
  --weights results/baseline-yolov5s-random-seed0/weights/best.pt \
  --batch-size 16 --imgsz 640 --workers 8 --device 0 --half --verbose \
  --project results/validation --name baseline-yolov5s-random-seed0 \
  --exist-ok
python -m torch.utils.collect_env \
  > results/baseline-yolov5s-random-seed0/environment.txt
python -m pip freeze \
  > results/baseline-yolov5s-random-seed0/pip-freeze.txt
```

The strict reproduction is complete when training exits successfully,
`best.pt` reloads in standalone validation, and the validation output reports
overall and per-class P, R, mAP50, and mAP50-95. The test submission generator
is intentionally excluded because the supplied `test.py` depends on labels
that do not exist in the competition test split.

The competition score is F1. For each standalone validation result, also report
`2 * precision * recall / (precision + recall)`, or 0 when both values are 0.
The baseline's mAP-dominated checkpoint selection remains unchanged so this run
stays a strict reproduction.
````

- [ ] **Step 2: Verify every documented CLI option exists**

Run:

```bash
.venv/bin/python train.py --help
.venv/bin/python val.py --help
```

Expected: every option used by the runbook appears in the corresponding help output.

- [ ] **Step 3: Run the complete test suite**

Run:

```bash
.venv/bin/python -m pytest tests -v
.venv/bin/python -m py_compile train.py val.py repro/audit_dataset.py repro/create_smoke_dataset.py
```

Expected: all tests pass and all Python files compile.

- [ ] **Step 4: Commit**

Run from the repository root:

```bash
git add 26SummerHackathon_GenAI/baseline/repro/README.md
git commit -m "docs: add baseline reproduction runbook"
```

---

## Completion Report

Report:

- full dataset audit counts and whether JPEG verification passed;
- smoke command and wall-clock duration;
- smoke training losses and overall validation P, R, mAP50, and mAP50-95;
- aggregate validation F1 computed from P and R;
- confirmation that `best.pt` reloaded successfully;
- local environment limitations;
- exact CUDA command still required for the full 30-epoch metric;
- any baseline correctness defect discovered without silently broadening scope.
