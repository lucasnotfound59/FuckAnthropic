# YOLO26s-P2 Scratch Training and Submission Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an independent YOLO26s-P2 package that trains only from random initialization, evaluates standard and count-based F1, tunes confidence thresholds, and writes validated competition submissions.

**Architecture:** Use Ultralytics 8.4.106 only as the training and inference engine. Keep the official YOLO26 P2 architecture in a local scale-`s` YAML, wrap it with strict source/resume guards, and keep dataset audit, metrics, checkpoint selection, prediction caching, and CSV serialization in small testable modules. Never import or modify the YOLOv5 baseline.

**Tech Stack:** Python 3.11, PyTorch 2.4–2.10, Ultralytics 8.4.106, PyYAML 6.0.3, NumPy 2.2.6, Pillow 12.1.1, pytest 8.4.2, uv.

## Global Constraints

- Create all implementation files under `26SummerHackathon_GenAI/yolo26s_p2`; do not modify or import `26SummerHackathon_GenAI/baseline`.
- New training accepts only `configs/model.yaml`; never accept a model alias, URL, or `.pt` weight.
- `.pt` is allowed only with explicit resume, and only below this package's `runs` directory with matching sidecar metadata.
- Do not download or load public pretrained weights or external training images.
- Preserve class IDs exactly: `0 person`, `1 rider`, `2 car`, `3 truck`, `4 bus`, `5 motorcycle`, `6 bicycle`, `7 traffic light`, `8 traffic sign`.
- Source data remains under repository-root `bdd100k_selected` and is read-only.
- Pin `ultralytics==8.4.106`; this is the PyPI release used by the plan.
- Default reference training uses `epochs=200`, `imgsz=640`, `optimizer=MuSGD`, `lr0=0.005`, `lrf=0.01`, `momentum=0.948`, `weight_decay=0.0005`, `warmup_epochs=3.0`, `box=9.83`, `cls=0.65`, `dfl=0.96`, `mosaic=1.0`, `mixup=0.05`, `scale=0.9`, `translate=0.1`, `fliplr=0.5`, `flipud=0.0`, `degrees=0.0`, `shear=0.0`, `perspective=0.0`, `close_mosaic=20`, `patience=50`, `cls_pw=0.0`, `seed=0`, and deterministic mode.
- Code style follows the user's convention of no spaces around assignment and arithmetic operators where Python syntax permits.
- Generated `.venv`, `.artifacts`, `runs`, prediction caches, and submissions stay out of Git.

## File Map

- `26SummerHackathon_GenAI/yolo26s_p2/pyproject.toml`: package metadata and pinned dependencies.
- `26SummerHackathon_GenAI/yolo26s_p2/configs/model.yaml`: local official YOLO26-P2 graph with `scale: s` and `nc: 9`.
- `26SummerHackathon_GenAI/yolo26s_p2/configs/data.yaml`: repository-relative full dataset.
- `26SummerHackathon_GenAI/yolo26s_p2/configs/scratch_ref.yaml`: exact reference training recipe.
- `26SummerHackathon_GenAI/yolo26s_p2/configs/smoke.yaml`: one-epoch CPU smoke overlay.
- `26SummerHackathon_GenAI/yolo26s_p2/yolo26_experiment/config.py`: checked YAML loading and overlay merging.
- `26SummerHackathon_GenAI/yolo26s_p2/yolo26_experiment/initialization.py`: model-source and resume guards.
- `26SummerHackathon_GenAI/yolo26s_p2/yolo26_experiment/run_metadata.py`: fingerprints and JSON metadata.
- `26SummerHackathon_GenAI/yolo26s_p2/yolo26_experiment/data_audit.py`: source dataset validation.
- `26SummerHackathon_GenAI/yolo26s_p2/yolo26_experiment/smoke_data.py`: deterministic linked smoke split.
- `26SummerHackathon_GenAI/yolo26s_p2/yolo26_experiment/metrics.py`: count F1 and threshold search.
- `26SummerHackathon_GenAI/yolo26s_p2/yolo26_experiment/predictions.py`: low-threshold prediction cache.
- `26SummerHackathon_GenAI/yolo26s_p2/yolo26_experiment/submission.py`: count serialization and schema validation.
- `26SummerHackathon_GenAI/yolo26s_p2/yolo26_experiment/checkpoints.py`: F1 callback and atomic checkpoint copies.
- `26SummerHackathon_GenAI/yolo26s_p2/train.py`: guarded training and resume CLI.
- `26SummerHackathon_GenAI/yolo26s_p2/evaluate.py`: standalone Ultralytics validation and JSON report.
- `26SummerHackathon_GenAI/yolo26s_p2/experiment.py`: single-factor overlay runner.
- `26SummerHackathon_GenAI/yolo26s_p2/tune_thresholds.py`: compare checkpoint candidates and freeze thresholds.
- `26SummerHackathon_GenAI/yolo26s_p2/submit.py`: test inference and both CSV schema variants.
- `26SummerHackathon_GenAI/yolo26s_p2/tests/**`: unit and integration coverage.
- `26SummerHackathon_GenAI/yolo26s_p2/README.md`: local, single-GPU, multi-GPU, resume, tune, and submit commands.
- `.gitignore`: ignore only generated artifacts for the new package.

---

### Task 1: Package, Dataset Configuration, and YOLO26s-P2 Graph

**Files:**
- Create: `26SummerHackathon_GenAI/yolo26s_p2/pyproject.toml`
- Create: `26SummerHackathon_GenAI/yolo26s_p2/configs/model.yaml`
- Create: `26SummerHackathon_GenAI/yolo26s_p2/configs/data.yaml`
- Create: `26SummerHackathon_GenAI/yolo26s_p2/configs/scratch_ref.yaml`
- Create: `26SummerHackathon_GenAI/yolo26s_p2/configs/smoke.yaml`
- Create: `26SummerHackathon_GenAI/yolo26s_p2/yolo26_experiment/__init__.py`
- Create: `26SummerHackathon_GenAI/yolo26s_p2/tests/test_model_architecture.py`
- Create: `26SummerHackathon_GenAI/yolo26s_p2/tests/test_static_config.py`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: repository-root `bdd100k_selected`.
- Produces: local YOLO model YAML with `nc=9`, `scale="s"`, `end2end=True`, and Detect inputs `[19,22,25,28]`.
- Produces: `configs/data.yaml` whose `path` resolves to the repository root.

- [ ] **Step 1: Write static configuration tests**

Create `tests/test_static_config.py`:

```python
from pathlib import Path

import yaml


ROOT=Path(__file__).resolve().parents[1]
REPO_ROOT=ROOT.parents[1]


def load_yaml(name: str)->dict:
    return yaml.safe_load((ROOT/"configs"/name).read_text())


def test_data_yaml_targets_repository_dataset():
    data=load_yaml("data.yaml")
    resolved=(ROOT/"configs"/data["path"]).resolve()
    assert resolved==REPO_ROOT
    assert data["train"]=="bdd100k_selected/images/train"
    assert data["val"]=="bdd100k_selected/images/val"
    assert data["test"]=="bdd100k_selected/images/test"
    assert list(data["names"])==[
        "person","rider","car","truck","bus","motorcycle","bicycle",
        "traffic light","traffic sign",
    ]


def test_scratch_reference_has_exact_recipe():
    cfg=load_yaml("scratch_ref.yaml")
    assert cfg["epochs"]==200
    assert cfg["imgsz"]==640
    assert cfg["optimizer"]=="MuSGD"
    assert cfg["pretrained"] is False
    assert cfg["resume"] is False
    assert cfg["seed"]==0
    assert cfg["deterministic"] is True
    assert cfg["cls_pw"]==0.0
```

Create `tests/test_model_architecture.py`:

```python
from pathlib import Path

import yaml


ROOT=Path(__file__).resolve().parents[1]


def test_model_yaml_is_yolo26s_p2_with_nine_classes():
    cfg=yaml.safe_load((ROOT/"configs/model.yaml").read_text())
    assert cfg["nc"]==9
    assert cfg["scale"]=="s"
    assert cfg["end2end"] is True
    assert cfg["reg_max"]==1
    assert cfg["head"][-1]==[[19,22,25,28],1,"Detect",["nc"]]
```

- [ ] **Step 2: Run tests to verify missing files fail**

Run:

```bash
cd 26SummerHackathon_GenAI/yolo26s_p2
python3 -m pytest tests/test_static_config.py tests/test_model_architecture.py -v
```

Expected: FAIL because the package and YAML files do not exist.

- [ ] **Step 3: Create package metadata**

Create `pyproject.toml`:

```toml
[build-system]
requires=["hatchling"]
build-backend="hatchling.build"

[project]
name="yolo26s-p2-bdd"
version="0.1.0"
description="Scratch-only YOLO26s-P2 experiments for the BDD100K selected dataset"
requires-python=">=3.11,<3.14"
dependencies=[
  "numpy==2.2.6",
  "pillow==12.1.1",
  "pyyaml==6.0.3",
  "torch>=2.4,<2.11",
  "torchvision>=0.19,<0.26",
  "ultralytics==8.4.106",
]

[dependency-groups]
dev=["pytest==8.4.2"]

[tool.hatch.build.targets.wheel]
packages=["yolo26_experiment"]

[tool.pytest.ini_options]
testpaths=["tests"]
addopts="-ra"
```

Create `yolo26_experiment/__init__.py`:

```python
"""Scratch-only YOLO26s-P2 experiment utilities."""

__version__="0.1.0"
```

- [ ] **Step 4: Add exact data and training YAML files**

Create `configs/data.yaml`:

```yaml
path: ../../..
train: bdd100k_selected/images/train
val: bdd100k_selected/images/val
test: bdd100k_selected/images/test
names:
  - person
  - rider
  - car
  - truck
  - bus
  - motorcycle
  - bicycle
  - traffic light
  - traffic sign
```

Create `configs/scratch_ref.yaml`:

```yaml
model: configs/model.yaml
data: configs/data.yaml
project: runs
name: scratch_ref
epochs: 200
patience: 50
batch: -1
imgsz: 640
device: cpu
workers: 8
cache: false
pretrained: false
resume: false
optimizer: MuSGD
lr0: 0.005
lrf: 0.01
cos_lr: true
momentum: 0.948
weight_decay: 0.0005
warmup_epochs: 3.0
box: 9.83
cls: 0.65
dfl: 0.96
cls_pw: 0.0
mosaic: 1.0
mixup: 0.05
scale: 0.9
translate: 0.1
fliplr: 0.5
flipud: 0.0
degrees: 0.0
shear: 0.0
perspective: 0.0
close_mosaic: 20
seed: 0
deterministic: true
amp: true
save: true
save_period: 10
plots: true
```

Create `configs/smoke.yaml` as an overlay:

```yaml
name: smoke
data: .artifacts/smoke/data.yaml
epochs: 1
patience: 1
batch: 2
imgsz: 320
device: cpu
workers: 0
cache: false
amp: false
save_period: -1
plots: false
```

- [ ] **Step 5: Add the exact local YOLO26s-P2 model graph**

Create `configs/model.yaml` from the official AGPL-3.0 `yolo26-p2.yaml`, changing only `nc` and adding explicit `scale`:

```yaml
# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
# YOLO26s-P2 object detection model with P2/4 - P5/32 outputs.
nc: 9
scale: s
end2end: true
reg_max: 1
scales:
  n: [0.50,0.25,1024]
  s: [0.50,0.50,1024]
  m: [0.50,1.00,512]
  l: [1.00,1.00,512]
  x: [1.00,1.50,512]
backbone:
  - [-1,1,Conv,[64,3,2]]
  - [-1,1,Conv,[128,3,2]]
  - [-1,2,C3k2,[256,false,0.25]]
  - [-1,1,Conv,[256,3,2]]
  - [-1,2,C3k2,[512,false,0.25]]
  - [-1,1,Conv,[512,3,2]]
  - [-1,2,C3k2,[512,true]]
  - [-1,1,Conv,[1024,3,2]]
  - [-1,2,C3k2,[1024,true]]
  - [-1,1,SPPF,[1024,5,3,true]]
  - [-1,2,C2PSA,[1024]]
head:
  - [-1,1,nn.Upsample,[null,2,nearest]]
  - [[-1,6],1,Concat,[1]]
  - [-1,2,C3k2,[512,true]]
  - [-1,1,nn.Upsample,[null,2,nearest]]
  - [[-1,4],1,Concat,[1]]
  - [-1,2,C3k2,[256,true]]
  - [-1,1,nn.Upsample,[null,2,nearest]]
  - [[-1,2],1,Concat,[1]]
  - [-1,2,C3k2,[128,true]]
  - [-1,1,Conv,[128,3,2]]
  - [[-1,16],1,Concat,[1]]
  - [-1,2,C3k2,[256,true]]
  - [-1,1,Conv,[256,3,2]]
  - [[-1,13],1,Concat,[1]]
  - [-1,2,C3k2,[512,true]]
  - [-1,1,Conv,[512,3,2]]
  - [[-1,10],1,Concat,[1]]
  - [-1,1,C3k2,[1024,true,0.5,true]]
  - [[19,22,25,28],1,Detect,[nc]]
```

- [ ] **Step 6: Ignore only generated artifacts**

Append to repository `.gitignore`:

```gitignore
26SummerHackathon_GenAI/yolo26s_p2/.venv/
26SummerHackathon_GenAI/yolo26s_p2/.artifacts/
26SummerHackathon_GenAI/yolo26s_p2/runs/
26SummerHackathon_GenAI/yolo26s_p2/submissions/
```

- [ ] **Step 7: Create the isolated environment and run tests**

Run:

```bash
cd 26SummerHackathon_GenAI/yolo26s_p2
uv sync --python 3.11 --group dev
uv run pytest tests/test_static_config.py tests/test_model_architecture.py -v
```

Expected: 3 passed.

- [ ] **Step 8: Commit**

```bash
git add .gitignore 26SummerHackathon_GenAI/yolo26s_p2
git commit -m "feat: scaffold YOLO26s-P2 scratch package"
```

---

### Task 2: Checked Configuration, Initialization Guard, and Metadata

**Files:**
- Create: `26SummerHackathon_GenAI/yolo26s_p2/yolo26_experiment/config.py`
- Create: `26SummerHackathon_GenAI/yolo26s_p2/yolo26_experiment/initialization.py`
- Create: `26SummerHackathon_GenAI/yolo26s_p2/yolo26_experiment/run_metadata.py`
- Create: `26SummerHackathon_GenAI/yolo26s_p2/tests/test_config.py`
- Create: `26SummerHackathon_GenAI/yolo26s_p2/tests/test_initialization.py`

**Interfaces:**
- Produces: `load_experiment(base: Path, overlay: Path|None, overrides: dict|None)->dict`.
- Produces: `new_model_source(project_root: Path, value: str|Path)->Path`.
- Produces: `resume_source(project_root: Path, value: str|Path, expected_fingerprint: str)->Path`.
- Produces: `write_metadata(path: Path, payload: dict)->None` and `sha256_file(path: Path)->str`.

- [ ] **Step 1: Write failing configuration and guard tests**

Create `tests/test_config.py`:

```python
from pathlib import Path

import pytest
import yaml

from yolo26_experiment.config import ConfigError,load_experiment


def write_yaml(path: Path,data: dict)->Path:
    path.write_text(yaml.safe_dump(data))
    return path


def test_overlay_and_cli_override_are_applied(tmp_path):
    base=write_yaml(tmp_path/"base.yaml",{"epochs":200,"imgsz":640,"name":"base"})
    overlay=write_yaml(tmp_path/"overlay.yaml",{"imgsz":320,"name":"smoke"})
    result=load_experiment(base,overlay,{"epochs":1})
    assert result=={"epochs":1,"imgsz":320,"name":"smoke"}


def test_unknown_key_is_rejected(tmp_path):
    base=write_yaml(tmp_path/"base.yaml",{"model":"configs/model.yaml","mystery":1})
    with pytest.raises(ConfigError,match="mystery"):
        load_experiment(base)
```

Create `tests/test_initialization.py`:

```python
from pathlib import Path

import pytest

from yolo26_experiment.initialization import SourceError,new_model_source,resume_source
from yolo26_experiment.run_metadata import write_metadata


def test_new_run_accepts_only_local_model_yaml(tmp_path):
    project=tmp_path/"project"
    model=project/"configs/model.yaml"
    model.parent.mkdir(parents=True)
    model.write_text("nc: 9\n")
    assert new_model_source(project,model)==model.resolve()
    for forbidden in ("yolo26s.pt","https://example.com/model.yaml","yolo26s.yaml"):
        with pytest.raises(SourceError):
            new_model_source(project,forbidden)


def test_resume_requires_local_run_and_matching_metadata(tmp_path):
    project=tmp_path/"project"
    checkpoint=project/"runs/exp/weights/last.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    write_metadata(
        checkpoint.parents[1]/"run_metadata.json",
        {"project":"yolo26s_p2","model_fingerprint":"abc"},
    )
    assert resume_source(project,checkpoint,expected_fingerprint="abc")==checkpoint.resolve()
    with pytest.raises(SourceError,match="fingerprint"):
        resume_source(project,checkpoint,expected_fingerprint="wrong")
```

- [ ] **Step 2: Run tests to verify imports fail**

Run:

```bash
uv run pytest tests/test_config.py tests/test_initialization.py -v
```

Expected: FAIL with missing `config`, `initialization`, and `run_metadata` modules.

- [ ] **Step 3: Implement checked configuration loading**

Create `yolo26_experiment/config.py`:

```python
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    pass


ALLOWED_KEYS={
    "model","data","project","name","epochs","patience","batch","imgsz",
    "device","workers","cache","pretrained","resume","optimizer","lr0","lrf",
    "cos_lr","momentum","weight_decay","warmup_epochs","box","cls","dfl",
    "cls_pw","mosaic","mixup","scale","translate","fliplr","flipud","degrees",
    "shear","perspective","close_mosaic","seed","deterministic","amp",
    "save","save_period","plots","exist_ok",
}


def _read_yaml(path: Path)->dict[str,Any]:
    value=yaml.safe_load(path.read_text()) or {}
    if not isinstance(value,dict):
        raise ConfigError(f"{path} must contain a mapping")
    unknown=set(value)-ALLOWED_KEYS
    if unknown:
        raise ConfigError(f"unknown configuration keys: {sorted(unknown)}")
    return value


def load_experiment(
    base: Path,
    overlay: Path|None=None,
    overrides: dict[str,Any]|None=None,
)->dict[str,Any]:
    result=_read_yaml(Path(base))
    if overlay is not None:
        result.update(_read_yaml(Path(overlay)))
    if overrides:
        unknown=set(overrides)-ALLOWED_KEYS
        if unknown:
            raise ConfigError(f"unknown override keys: {sorted(unknown)}")
        result.update({key:value for key,value in overrides.items() if value is not None})
    if result.get("pretrained",False):
        raise ConfigError("pretrained must remain false")
    return result
```

- [ ] **Step 4: Implement atomic metadata and fingerprints**

Create `yolo26_experiment/run_metadata.py`:

```python
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


def sha256_file(path: Path)->str:
    digest=hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_metadata(path: Path,payload: dict[str,Any])->None:
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    fd,temp_name=tempfile.mkstemp(dir=path.parent,prefix=f".{path.name}.")
    try:
        with os.fdopen(fd,"w",encoding="utf-8") as handle:
            json.dump(payload,handle,indent=2,sort_keys=True)
            handle.write("\n")
        os.replace(temp_name,path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)
```

- [ ] **Step 5: Implement strict new-run and resume guards**

Create `yolo26_experiment/initialization.py`:

```python
import json
from pathlib import Path


class SourceError(ValueError):
    pass


def _inside(path: Path,parent: Path)->bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def new_model_source(project_root: Path,value: str|Path)->Path:
    project_root=Path(project_root).resolve()
    raw=str(value)
    if "://" in raw or Path(raw).suffix==".pt":
        raise SourceError("new runs accept only the local model YAML")
    candidate=Path(value)
    if not candidate.is_absolute():
        candidate=project_root/candidate
    candidate=candidate.resolve()
    expected=(project_root/"configs/model.yaml").resolve()
    if candidate!=expected or not candidate.is_file():
        raise SourceError(f"new model source must be {expected}")
    return candidate


def resume_source(
    project_root: Path,
    value: str|Path,
    expected_fingerprint: str,
)->Path:
    project_root=Path(project_root).resolve()
    candidate=Path(value).resolve()
    runs=(project_root/"runs").resolve()
    if candidate.suffix!=".pt" or not candidate.is_file() or not _inside(candidate,runs):
        raise SourceError("resume checkpoint must be a local .pt below runs")
    metadata_path=candidate.parents[1]/"run_metadata.json"
    if not metadata_path.is_file():
        raise SourceError("resume checkpoint is missing run_metadata.json")
    metadata=json.loads(metadata_path.read_text())
    if metadata.get("project")!="yolo26s_p2":
        raise SourceError("resume checkpoint belongs to another project")
    if metadata.get("model_fingerprint")!=expected_fingerprint:
        raise SourceError("resume checkpoint model fingerprint mismatch")
    return candidate
```

- [ ] **Step 6: Run focused tests**

Run:

```bash
uv run pytest tests/test_config.py tests/test_initialization.py -v
```

Expected: 4 passed.

- [ ] **Step 7: Commit**

```bash
git add 26SummerHackathon_GenAI/yolo26s_p2
git commit -m "feat: guard scratch initialization and experiment config"
```

---

### Task 3: Dataset Audit and Deterministic Smoke Split

**Files:**
- Create: `26SummerHackathon_GenAI/yolo26s_p2/yolo26_experiment/data_audit.py`
- Create: `26SummerHackathon_GenAI/yolo26s_p2/yolo26_experiment/smoke_data.py`
- Create: `26SummerHackathon_GenAI/yolo26s_p2/tests/test_data_audit.py`
- Create: `26SummerHackathon_GenAI/yolo26s_p2/tests/test_smoke_data.py`

**Interfaces:**
- Produces: `audit_dataset(dataset_root: Path, expected: dict[str,int]|None=None)->AuditReport`.
- Produces: `create_smoke_dataset(source_root: Path, target_root: Path, train_size: int=18, val_size: int=9)->Path`.

- [ ] **Step 1: Write failing audit tests**

Create fixtures using two 16×16 RGB JPEGs and YOLO labels. Assert:

```python
def test_audit_accepts_valid_dataset(tiny_dataset):
    report=audit_dataset(tiny_dataset,{"train":1,"val":1,"test":1})
    assert report.image_counts=={"train":1,"val":1,"test":1}
    assert report.label_counts=={"train":1,"val":1}


def test_audit_reports_file_and_line_for_invalid_class(tiny_dataset):
    label=tiny_dataset/"labels/train/train.txt"
    label.write_text("9 0.5 0.5 0.2 0.2\n")
    with pytest.raises(DatasetError,match=r"train\.txt:1.*class"):
        audit_dataset(tiny_dataset)
```

Create a smoke test asserting the generated train split contains every class
`0..8`, repeated generation is idempotent, and `data.yaml` points at the
generated tree.

- [ ] **Step 2: Run tests to verify imports fail**

Run:

```bash
uv run pytest tests/test_data_audit.py tests/test_smoke_data.py -v
```

Expected: FAIL with missing modules.

- [ ] **Step 3: Implement the dataset audit**

Use these exact validation rules in `data_audit.py`:

```python
@dataclass(frozen=True)
class AuditReport:
    image_counts: dict[str,int]
    label_counts: dict[str,int]
    class_counts: tuple[int,...]


def parse_label(path: Path)->list[int]:
    classes=[]
    for line_number,line in enumerate(path.read_text().splitlines(),1):
        fields=line.split()
        if len(fields)!=5:
            raise DatasetError(f"{path}:{line_number}: expected 5 fields")
        try:
            cls_float,x,y,w,h=map(float,fields)
        except ValueError as exc:
            raise DatasetError(f"{path}:{line_number}: non-numeric field") from exc
        if not all(math.isfinite(value) for value in (cls_float,x,y,w,h)):
            raise DatasetError(f"{path}:{line_number}: non-finite value")
        cls=int(cls_float)
        if cls_float!=cls or not 0<=cls<9:
            raise DatasetError(f"{path}:{line_number}: class must be integer 0..8")
        if not 0<=x<=1 or not 0<=y<=1 or not 0<w<=1 or not 0<h<=1:
            raise DatasetError(f"{path}:{line_number}: invalid normalized box")
        if x-w/2<0 or x+w/2>1 or y-h/2<0 or y+h/2>1:
            raise DatasetError(f"{path}:{line_number}: box extends outside image")
        classes.append(cls)
    return classes
```

`audit_dataset` must decode each image with `Image.open(path).verify()`, require
matching image/label stems for train and val, ignore `.DS_Store`, require no
test `.txt` labels, and enforce expected full counts when supplied.

- [ ] **Step 4: Implement deterministic smoke linking**

`smoke_data.py` must:

1. parse all source train labels;
2. greedily choose lexicographically sorted images until all nine classes are
   represented;
3. add lexicographically sorted images until `train_size`;
4. choose the first `val_size` valid validation images;
5. create relative symlinks below `.artifacts/smoke/images` and
   `.artifacts/smoke/labels`;
6. write this exact YAML:

```yaml
path: .
train: images/train
val: images/val
test: images/val
names:
  - person
  - rider
  - car
  - truck
  - bus
  - motorcycle
  - bicycle
  - traffic light
  - traffic sign
```

- [ ] **Step 5: Run focused and full-data audit tests**

Run:

```bash
uv run pytest tests/test_data_audit.py tests/test_smoke_data.py -v
uv run python -m yolo26_experiment.data_audit \
  --dataset ../../bdd100k_selected \
  --train-count 7000 --val-count 2000 --test-count 1000
```

Expected: all tests pass and the audit prints the three image counts plus
per-class instance counts.

- [ ] **Step 6: Commit**

```bash
git add 26SummerHackathon_GenAI/yolo26s_p2
git commit -m "feat: audit BDD data and create smoke split"
```

---

### Task 4: Count-F1, Threshold Search, and Submission Serialization

**Files:**
- Create: `26SummerHackathon_GenAI/yolo26s_p2/yolo26_experiment/metrics.py`
- Create: `26SummerHackathon_GenAI/yolo26s_p2/yolo26_experiment/predictions.py`
- Create: `26SummerHackathon_GenAI/yolo26s_p2/yolo26_experiment/submission.py`
- Create: `26SummerHackathon_GenAI/yolo26s_p2/tests/test_metrics.py`
- Create: `26SummerHackathon_GenAI/yolo26s_p2/tests/test_submission.py`

**Interfaces:**
- Consumes: `Detection(image_name: str,class_id: int,confidence: float)`.
- Produces: `count_f1(true_counts: np.ndarray,pred_counts: np.ndarray)->CountMetrics`.
- Produces: `tune_thresholds(true_counts: np.ndarray,detections: list[list[Detection]])->ThresholdResult`.
- Produces: `write_submission(image_names,counts,path,column,blank_empty)->None`.
- Produces: `validate_submission(path,expected_names,column,blank_empty)->None`.

- [ ] **Step 1: Write failing metric tests**

Create tests for:

```python
def test_count_f1_uses_multiset_intersection():
    truth=np.array([[2,0],[0,1]])
    prediction=np.array([[1,1],[0,2]])
    metrics=count_f1(truth,prediction)
    assert metrics.tp==2
    assert metrics.fp==2
    assert metrics.fn==1
    assert metrics.micro_f1==pytest.approx(4/7)


def test_threshold_tuning_can_choose_per_class_values():
    truth=np.array([[1,0],[0,1]])
    detections=[
        [Detection("a.jpg",0,0.8),Detection("a.jpg",1,0.2)],
        [Detection("b.jpg",0,0.4),Detection("b.jpg",1,0.6)],
    ]
    result=tune_thresholds(truth,detections,candidates=(0.3,0.5,0.7))
    assert result.class_thresholds==(0.7,0.5)
    assert result.metrics.micro_f1==1.0
```

- [ ] **Step 2: Write failing submission tests**

Cover:

- exact `pic_name,results` header;
- exact `pic_name,correct_predictions` header;
- nine semicolon-separated counts;
- lexicographic row ordering;
- all-zero and blank empty modes;
- duplicate/missing/unexpected filenames;
- negative or wrong-length count vectors.

- [ ] **Step 3: Run tests to verify imports fail**

Run:

```bash
uv run pytest tests/test_metrics.py tests/test_submission.py -v
```

Expected: FAIL with missing modules.

- [ ] **Step 4: Implement metrics and deterministic threshold search**

Implement `metrics.py` with immutable dataclasses:

```python
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from yolo26_experiment.predictions import Detection


@dataclass(frozen=True)
class CountMetrics:
    tp: int
    fp: int
    fn: int
    micro_precision: float
    micro_recall: float
    micro_f1: float
    per_class_f1: tuple[float,...]


@dataclass(frozen=True)
class ThresholdResult:
    global_threshold: float
    class_thresholds: tuple[float,...]
    metrics: CountMetrics


def _prf(tp: int,fp: int,fn: int)->tuple[float,float,float]:
    precision=tp/(tp+fp) if tp+fp else 0.0
    recall=tp/(tp+fn) if tp+fn else 0.0
    f1=2*precision*recall/(precision+recall) if precision+recall else 0.0
    return precision,recall,f1


def count_f1(truth: np.ndarray,prediction: np.ndarray)->CountMetrics:
    if truth.shape!=prediction.shape or truth.ndim!=2:
        raise ValueError("truth and prediction must be equal 2D arrays")
    if np.any(truth<0) or np.any(prediction<0):
        raise ValueError("counts must be non-negative")
    overlap=np.minimum(truth,prediction)
    tp=int(overlap.sum())
    fp=int((prediction-overlap).sum())
    fn=int((truth-overlap).sum())
    precision,recall,f1=_prf(tp,fp,fn)
    per_class=[]
    for class_id in range(truth.shape[1]):
        class_tp=int(overlap[:,class_id].sum())
        class_fp=int((prediction[:,class_id]-overlap[:,class_id]).sum())
        class_fn=int((truth[:,class_id]-overlap[:,class_id]).sum())
        per_class.append(_prf(class_tp,class_fp,class_fn)[2])
    return CountMetrics(tp,fp,fn,precision,recall,f1,tuple(per_class))


def _prediction_counts(
    detections: Sequence[Sequence[Detection]],
    thresholds: Sequence[float],
)->np.ndarray:
    counts=np.zeros((len(detections),len(thresholds)),dtype=np.int64)
    for image_index,image_detections in enumerate(detections):
        for detection in image_detections:
            if detection.confidence>=thresholds[detection.class_id]:
                counts[image_index,detection.class_id]+=1
    return counts


def tune_thresholds(
    truth: np.ndarray,
    detections: Sequence[Sequence[Detection]],
    candidates: Sequence[float]=tuple(value/100 for value in range(1,91)),
)->ThresholdResult:
    if len(detections)!=len(truth):
        raise ValueError("detection rows must align with truth rows")
    class_count=truth.shape[1]

    def evaluate(values: Sequence[float])->CountMetrics:
        return count_f1(truth,_prediction_counts(detections,values))

    global_threshold=max(
        candidates,
        key=lambda value:(
            evaluate([value]*class_count).micro_f1,
            evaluate([value]*class_count).micro_precision,
            value,
        ),
    )
    thresholds=[global_threshold]*class_count
    changed=True
    while changed:
        changed=False
        for class_id in range(class_count):
            best=max(
                candidates,
                key=lambda value:(
                    evaluate(
                        thresholds[:class_id]+[value]+thresholds[class_id+1:]
                    ).micro_f1,
                    evaluate(
                        thresholds[:class_id]+[value]+thresholds[class_id+1:]
                    ).micro_precision,
                    value,
                ),
            )
            if best!=thresholds[class_id]:
                thresholds[class_id]=best
                changed=True
    metrics=evaluate(thresholds)
    return ThresholdResult(global_threshold,tuple(thresholds),metrics)
```

Use `tp=np.minimum(truth,prediction)`, `fp=prediction-tp`, and
`fn=truth-tp`. Define zero-denominator precision, recall, and F1 as `0.0`.
Search candidates `0.01..0.90` in increments of `0.01`. Break equal-F1 ties by
higher precision, then the higher threshold. Start per-class thresholds at the
best global value and perform class-coordinate passes until no threshold
changes.

- [ ] **Step 5: Implement prediction cache schema**

Create `predictions.py`:

```python
import json
from dataclasses import asdict,dataclass
from pathlib import Path
from typing import Iterable,Sequence

import numpy as np


@dataclass(frozen=True)
class Detection:
    image_name: str
    class_id: int
    confidence: float


def write_jsonl(path: Path,detections: Iterable[Detection])->None:
    rows=sorted(
        detections,
        key=lambda row:(row.image_name,row.class_id,-row.confidence),
    )
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(asdict(row),sort_keys=True)+"\n")


def read_jsonl(path: Path)->list[Detection]:
    rows=[]
    for line_number,line in enumerate(Path(path).read_text().splitlines(),1):
        payload=json.loads(line)
        if set(payload)!={"image_name","class_id","confidence"}:
            raise ValueError(f"{path}:{line_number}: invalid detection fields")
        row=Detection(**payload)
        if not row.image_name or not 0<=row.class_id<9:
            raise ValueError(f"{path}:{line_number}: invalid detection")
        if not 0<=row.confidence<=1:
            raise ValueError(f"{path}:{line_number}: invalid confidence")
        rows.append(row)
    return sorted(rows,key=lambda row:(row.image_name,row.class_id,-row.confidence))


def counts_from_detections(
    image_names: Sequence[str],
    detections: Iterable[Detection],
    thresholds: Sequence[float],
    class_count: int=9,
)->np.ndarray:
    if len(thresholds)!=class_count:
        raise ValueError(f"expected {class_count} thresholds")
    if len(set(image_names))!=len(image_names):
        raise ValueError("image names must be unique")
    indices={name:index for index,name in enumerate(image_names)}
    counts=np.zeros((len(image_names),class_count),dtype=np.int64)
    for detection in detections:
        if detection.image_name not in indices:
            raise ValueError(f"unexpected image {detection.image_name}")
        if not 0<=detection.class_id<class_count:
            raise ValueError(f"invalid class {detection.class_id}")
        if detection.confidence>=thresholds[detection.class_id]:
            counts[indices[detection.image_name],detection.class_id]+=1
    return counts
```

JSONL rows must have only `image_name`, `class_id`, and `confidence`, reject
unknown fields and invalid classes/confidences, and sort by
`(image_name,class_id,-confidence)`.

- [ ] **Step 6: Implement submission writer and validator**

Create `submission.py` using Python's `csv` module:

```python
import csv
from pathlib import Path
from typing import Sequence

import numpy as np


VALID_COLUMNS={"results","correct_predictions"}


def write_submission(
    image_names: Sequence[str],
    counts: np.ndarray,
    path: Path,
    column: str="results",
    blank_empty: bool=False,
)->None:
    if column not in VALID_COLUMNS:
        raise ValueError(f"invalid prediction column {column}")
    if counts.shape!=(len(image_names),9) or np.any(counts<0):
        raise ValueError("counts must be a non-negative [images,9] array")
    if len(set(image_names))!=len(image_names):
        raise ValueError("image names must be unique")
    rows=sorted(zip(image_names,counts.tolist(),strict=True))
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=["pic_name",column])
        writer.writeheader()
        for image_name,row in rows:
            value="" if blank_empty and sum(row)==0 else ";".join(map(str,row))
            writer.writerow({"pic_name":image_name,column:value})


def validate_submission(
    path: Path,
    expected_names: Sequence[str],
    column: str="results",
    blank_empty: bool=False,
)->None:
    expected=sorted(expected_names)
    with Path(path).open(newline="",encoding="utf-8") as handle:
        reader=csv.DictReader(handle)
        if reader.fieldnames!=["pic_name",column]:
            raise ValueError(f"invalid header {reader.fieldnames}")
        rows=list(reader)
    names=[row["pic_name"] for row in rows]
    if names!=expected or len(names)!=len(set(names)):
        raise ValueError("submission filenames do not match the test set")
    for row in rows:
        value=row[column]
        if not value:
            if blank_empty:
                continue
            raise ValueError(f"{row['pic_name']} has an empty result")
        fields=value.split(";")
        if len(fields)!=9:
            raise ValueError(f"{row['pic_name']} must contain nine counts")
        try:
            values=[int(field) for field in fields]
        except ValueError as exc:
            raise ValueError(f"{row['pic_name']} has a non-integer count") from exc
        if any(value<0 for value in values):
            raise ValueError(f"{row['pic_name']} has a negative count")
```

`write_submission` sorts expected image names and writes either:

```text
image.jpg,0;1;2;0;0;0;0;0;1
```

or a blank second field when `blank_empty=True` and the row sum is zero.
`validate_submission` round-trips with `csv.DictReader` and enforces every rule
from the design.

- [ ] **Step 7: Run focused tests**

Run:

```bash
uv run pytest tests/test_metrics.py tests/test_submission.py -v
```

Expected: all tests pass.

- [ ] **Step 8: Commit**

```bash
git add 26SummerHackathon_GenAI/yolo26s_p2
git commit -m "feat: tune count F1 and validate submissions"
```

---

### Task 5: Checkpoint Selection and Guarded Training CLI

**Files:**
- Create: `26SummerHackathon_GenAI/yolo26s_p2/yolo26_experiment/checkpoints.py`
- Create: `26SummerHackathon_GenAI/yolo26s_p2/train.py`
- Create: `26SummerHackathon_GenAI/yolo26s_p2/experiment.py`
- Create: `26SummerHackathon_GenAI/yolo26s_p2/tests/test_checkpoints.py`
- Create: `26SummerHackathon_GenAI/yolo26s_p2/tests/test_train_cli.py`

**Interfaces:**
- Produces: `DetectionF1Checkpoint(run_dir: Path)` callable for Ultralytics `on_model_save`.
- Produces: `train.run(config: Path,overlay: Path|None,resume: Path|None,overrides: dict)->Path`.
- Produces: run weights `last.pt`, `best_map.pt`, and `best_detection_f1.pt`.

- [ ] **Step 1: Write failing checkpoint callback test**

Use a fake trainer with `metrics`, `last`, and `epoch`. Call the callback with
F1 `0.5`, then `0.4`, then `0.6`. Assert only the first and third calls replace
`best_detection_f1.pt`, and `best_detection_f1.json` records epoch 3 and F1
`0.6`.

- [ ] **Step 2: Write failing CLI guard tests**

Monkeypatch `train.YOLO` with a fake class. Assert:

- a new run constructs it with the resolved local model YAML;
- no argument passed to `.train()` has `pretrained=True`;
- a `.pt` supplied without `--resume` raises `SourceError`;
- resume calls `YOLO(local_last_pt)` and `.train(resume=True)`;
- project/name resolve below this package's `runs` directory.

- [ ] **Step 3: Run tests to verify imports fail**

Run:

```bash
uv run pytest tests/test_checkpoints.py tests/test_train_cli.py -v
```

Expected: FAIL with missing module/scripts.

- [ ] **Step 4: Implement atomic F1 checkpoint selection**

Create `checkpoints.py`:

```python
import json
import os
import shutil
import tempfile
from pathlib import Path

from yolo26_experiment.run_metadata import write_metadata


def atomic_copy(source: Path,target: Path)->None:
    target=Path(target)
    target.parent.mkdir(parents=True,exist_ok=True)
    fd,temp_name=tempfile.mkstemp(dir=target.parent,prefix=f".{target.name}.")
    os.close(fd)
    try:
        shutil.copy2(source,temp_name)
        with open(temp_name,"rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temp_name,target)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


class DetectionF1Checkpoint:
    def __init__(self,run_dir: Path):
        self.run_dir=Path(run_dir)
        state_path=self.run_dir/"best_detection_f1.json"
        self.best=(
            float(json.loads(state_path.read_text())["f1"])
            if state_path.is_file()
            else -1.0
        )

    def __call__(self,trainer)->None:
        metrics=trainer.metrics or {}
        precision=float(metrics.get("metrics/precision(B)",0.0))
        recall=float(metrics.get("metrics/recall(B)",0.0))
        f1=2*precision*recall/(precision+recall) if precision+recall else 0.0
        if f1<=self.best:
            return
        self.best=f1
        atomic_copy(
            Path(trainer.last),
            self.run_dir/"weights/best_detection_f1.pt",
        )
        write_metadata(
            self.run_dir/"best_detection_f1.json",
            {"epoch":int(trainer.epoch)+1,"precision":precision,"recall":recall,"f1":f1},
        )
```

- [ ] **Step 5: Implement guarded training**

`train.run` must:

1. load `scratch_ref.yaml`, optional overlay, and typed CLI overrides;
2. resolve the run directory and reject an existing run unless resume;
3. fingerprint `configs/model.yaml`;
4. audit the full data or construct/audit smoke data;
5. for new training, call `YOLO(local_model_yaml,task="detect")`;
6. verify `model.model.yaml["scale"]=="s"`, `nc==9`, and end-to-end mode;
7. write `resolved_config.yaml` and `run_metadata.json`;
8. register `DetectionF1Checkpoint` on `on_model_save`;
9. call `model.train(**train_args)` with `model`, `project`, `name`, and
   project-only keys removed;
10. atomically copy native `weights/best.pt` to `weights/best_map.pt`;
11. reload both selected checkpoints with `YOLO(path)` and fail if either is
   missing.

CLI arguments:

```text
--config PATH
--overlay PATH
--resume PATH
--device VALUE
--batch INTEGER
--workers INTEGER
--imgsz INTEGER
--epochs INTEGER
--name VALUE
```

- [ ] **Step 6: Implement single-factor experiment runner**

`experiment.py` accepts a base config, a list of overlay YAMLs, and server
overrides. It invokes `train.run` once per overlay with unique names and writes
`runs/experiment_index.json`; it refuses duplicate names.

- [ ] **Step 7: Run focused tests and syntax checks**

Run:

```bash
uv run pytest tests/test_checkpoints.py tests/test_train_cli.py -v
uv run python -m py_compile train.py experiment.py yolo26_experiment/*.py
```

Expected: all tests and compilation pass.

- [ ] **Step 8: Commit**

```bash
git add 26SummerHackathon_GenAI/yolo26s_p2
git commit -m "feat: train YOLO26s-P2 with F1 checkpoints"
```

---

### Task 6: Evaluation, Prediction Cache, Threshold Freezing, and Submission CLIs

**Files:**
- Create: `26SummerHackathon_GenAI/yolo26s_p2/evaluate.py`
- Create: `26SummerHackathon_GenAI/yolo26s_p2/tune_thresholds.py`
- Create: `26SummerHackathon_GenAI/yolo26s_p2/submit.py`
- Create: `26SummerHackathon_GenAI/yolo26s_p2/tests/test_workflow_clis.py`

**Interfaces:**
- `evaluate.py --weights PATH --device VALUE` writes `evaluation.json`.
- `tune_thresholds.py --run-dir PATH --device VALUE` writes
  `thresholds.json`, cached validation predictions, and `best_count_f1.pt`.
- `submit.py --run-dir PATH --device VALUE` writes four validated CSVs:
  two headers × two empty modes.

- [ ] **Step 1: Write failing workflow tests with a fake YOLO**

The fake result exposes:

```python
result.path="/data/a.jpg"
result.boxes.cls=np.array([0,2])
result.boxes.conf=np.array([0.8,0.6])
```

Assert low-threshold inference becomes deterministic JSONL detections, tuning
writes a threshold vector of length 9 plus checkpoint SHA256, and submission
writes every expected filename.

- [ ] **Step 2: Run tests to verify scripts are missing**

Run:

```bash
uv run pytest tests/test_workflow_clis.py -v
```

Expected: FAIL with missing scripts.

- [ ] **Step 3: Implement standalone detection evaluation**

`evaluate.py` must call:

```python
metrics=model.val(
    data=str(data_yaml),
    split="val",
    imgsz=imgsz,
    batch=batch,
    device=device,
    plots=True,
    project=str(output.parent),
    name=output.name,
)
```

Write overall `precision`, `recall`, computed `f1`, `map50`, and `map50_95`.
For each class, use `metrics.box.class_result(index)` and compute per-class F1.
Include checkpoint SHA256, Ultralytics version, device, image size, and class
names.

- [ ] **Step 4: Implement reusable low-threshold prediction caching**

Add to `predictions.py`:

```python
def predict_to_jsonl(
    weights: Path,
    source: Path,
    output: Path,
    imgsz: int,
    device: str,
    batch: int,
    conf: float=0.001,
)->Path:
    model=YOLO(str(weights))
    results=model.predict(
        source=str(source),
        imgsz=imgsz,
        device=device,
        batch=batch,
        conf=conf,
        stream=True,
        verbose=False,
    )
    detections=[]
    for result in results:
        if result.boxes is None:
            continue
        classes=result.boxes.cls.detach().cpu().tolist()
        confidences=result.boxes.conf.detach().cpu().tolist()
        image_name=Path(result.path).name
        detections.extend(
            Detection(image_name,int(class_id),float(confidence))
            for class_id,confidence in zip(classes,confidences,strict=True)
        )
    write_jsonl(output,detections)
    return output
```

YOLO26 end-to-end predictions are not postprocessed with NMS, so do not expose
or tune an NMS IoU threshold.

- [ ] **Step 5: Implement checkpoint comparison and threshold freezing**

`tune_thresholds.py` must:

1. validate `best_map.pt` and `best_detection_f1.pt`;
2. cache validation predictions for each at `conf=0.001`;
3. load ground-truth per-image class counts from validation labels;
4. tune global and per-class thresholds for each candidate;
5. select the higher count micro-F1, breaking ties by macro per-class F1 and
   then detection mAP50-95;
6. atomically copy the winner to `weights/best_count_f1.pt`;
7. write `thresholds.json` with checkpoint source, SHA256, global threshold,
   nine class thresholds, count metrics, validation dataset fingerprint, and
   generation timestamp.

- [ ] **Step 6: Implement four validated submission outputs**

`submit.py` must:

1. verify `thresholds.json` matches `best_count_f1.pt` SHA256;
2. predict every repository test image at `conf=0.001`;
3. apply the frozen per-class thresholds;
4. write:
   - `submission_results_zeros.csv`;
   - `submission_results_blank.csv`;
   - `submission_correct_predictions_zeros.csv`;
   - `submission_correct_predictions_blank.csv`;
5. validate every output against the exact 1,000 test filenames;
6. print `submission_results_zeros.csv` as the default recommendation because
   it follows the concrete example.

- [ ] **Step 7: Run focused tests**

Run:

```bash
uv run pytest tests/test_workflow_clis.py -v
uv run python -m py_compile evaluate.py tune_thresholds.py submit.py
```

Expected: all tests and compilation pass.

- [ ] **Step 8: Commit**

```bash
git add 26SummerHackathon_GenAI/yolo26s_p2
git commit -m "feat: evaluate tune and submit YOLO26 predictions"
```

---

### Task 7: Local Integration Verification and Operator Documentation

**Files:**
- Create: `26SummerHackathon_GenAI/yolo26s_p2/README.md`
- Create: `26SummerHackathon_GenAI/yolo26s_p2/tests/test_integration_model.py`
- Modify: `26SummerHackathon_GenAI/yolo26s_p2/uv.lock`

**Interfaces:**
- Produces: locally verified installation, model construction, forward pass,
  smoke checkpoint reload, and miniature submission.
- Produces: exact server commands with override positions.

- [ ] **Step 1: Add model construction integration test**

Create `tests/test_integration_model.py`:

```python
from pathlib import Path

import torch
from ultralytics import YOLO


ROOT=Path(__file__).resolve().parents[1]


def test_local_yaml_builds_random_yolo26s_p2():
    torch.manual_seed(0)
    model=YOLO(str(ROOT/"configs/model.yaml"),task="detect")
    inner=model.model
    assert inner.nc==9
    assert inner.yaml["scale"]=="s"
    assert inner.end2end is True
    detect=inner.model[-1]
    assert len(detect.stride)==4
    assert sorted(int(value) for value in detect.stride)==[4,8,16,32]
    output=inner(torch.zeros(1,3,320,320))
    assert output is not None
```

- [ ] **Step 2: Run model integration test**

Run:

```bash
uv run pytest tests/test_integration_model.py -v
```

Expected: PASS without downloading a `.pt` file.

- [ ] **Step 3: Create and audit the smoke split**

Run:

```bash
uv run python -m yolo26_experiment.smoke_data \
  --source ../../bdd100k_selected \
  --target .artifacts/smoke \
  --train-size 18 --val-size 9
uv run python -m yolo26_experiment.data_audit \
  --dataset .artifacts/smoke --train-count 18 --val-count 9 --test-count 9
```

Expected: smoke audit passes and all nine classes occur in training labels.

- [ ] **Step 4: Run one-epoch smoke training**

Run:

```bash
uv run python train.py \
  --config configs/scratch_ref.yaml \
  --overlay configs/smoke.yaml
```

Expected:

- exit code 0;
- `runs/smoke/weights/last.pt`;
- `runs/smoke/weights/best_map.pt`;
- `runs/smoke/weights/best_detection_f1.pt`;
- both selected checkpoints reload successfully.

- [ ] **Step 5: Tune and generate a miniature submission**

Use the smoke validation directory as a temporary test source through the
Python workflow API. Expected: `best_count_f1.pt`, `thresholds.json`, and all
four CSV variants validate.

- [ ] **Step 6: Write operator README**

Document these exact command templates:

```bash
# Install
uv sync --python 3.11 --group dev

# Local verification
uv run pytest -v

# Single GPU
uv run python train.py --config configs/scratch_ref.yaml \
  --device 0 --batch 32 --workers 8 --name scratch_ref_gpu

# Multi GPU
uv run python train.py --config configs/scratch_ref.yaml \
  --device 0,1 --batch 64 --workers 8 --name scratch_ref_2gpu

# Resume
uv run python train.py --config configs/scratch_ref.yaml \
  --resume runs/scratch_ref_gpu/weights/last.pt

# Standard evaluation
uv run python evaluate.py \
  --weights runs/scratch_ref_gpu/weights/best_map.pt --device 0

# Count-F1 tuning and checkpoint selection
uv run python tune_thresholds.py --run-dir runs/scratch_ref_gpu --device 0

# Submission
uv run python submit.py --run-dir runs/scratch_ref_gpu --device 0
```

Explain that `.yaml` means random initialization, `.pt` new-run input is
forbidden, `submission_results_zeros.csv` is the current default, and the
platform sample submission overrides the header/empty choice if later supplied.

- [ ] **Step 7: Run the complete verification suite**

Run:

```bash
uv run pytest -v
uv run python -m py_compile \
  train.py evaluate.py experiment.py tune_thresholds.py submit.py \
  yolo26_experiment/*.py
git diff --check
```

Expected: all tests pass, all scripts compile, and no whitespace errors.

- [ ] **Step 8: Review implementation against the design**

Check:

- no file under `26SummerHackathon_GenAI/baseline` changed;
- no `.pt` file is tracked;
- no code path downloads model weights;
- all generated artifacts are ignored;
- submission tests cover both screenshot conflicts;
- README contains server override points.

- [ ] **Step 9: Commit**

```bash
git add 26SummerHackathon_GenAI/yolo26s_p2
git commit -m "docs: verify YOLO26s-P2 scratch workflow"
```
