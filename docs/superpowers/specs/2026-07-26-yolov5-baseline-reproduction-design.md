# YOLOv5 Baseline Reproduction Design

## Goal

Reproduce the supplied YOLOv5-style BDD100K object-detection baseline before
making model or hyperparameter improvements. The reproduction must demonstrate
that data loading, training, validation, checkpointing, and test inference work
end to end and produce repeatable validation metrics.

This phase does not attempt to improve leaderboard performance or redesign the
competition submission format.

## Competition Evaluation Metric

The competition evaluates submissions with the F1 score:

`F1 = 2 × precision × recall / (precision + recall)`.

Strict baseline reproduction keeps the supplied checkpoint-selection fitness,
which is dominated by mAP50-95, because changing it would change the baseline.
Standalone validation must additionally report the aggregate F1 derived from
its precision and recall so the reproduced run can be interpreted against the
competition metric. Confidence-threshold tuning for a competitive submission
is a follow-up experiment, not part of strict reproduction.

## Current Repository State

The dataset is stored at the repository root under `bdd100k_selected`:

- `images/train`: 7,000 JPEG images
- `images/val`: 2,000 JPEG images
- `images/test`: 1,000 JPEG images
- `labels/train`: 7,000 YOLO-format label files
- `labels/val`: 2,000 YOLO-format label files
- `labels/test`: no labels, as expected for a competition test split

The baseline is stored under `26SummerHackathon_GenAI/baseline`. It implements a
YOLOv5-style detector with nine classes:

1. person
2. rider
3. car
4. truck
5. bus
6. motorcycle
7. bicycle
8. traffic light
9. traffic sign

The documented baseline trains a YOLOv5s configuration from random
initialization for 30 epochs at 640-pixel input resolution.

## Known Reproduction Risks

1. `dataset/dataset.yaml` currently points to
   `dataset/bdd10k_selected/...`, which does not match the dataset's actual
   repository location.
2. The host is an Apple M4 MacBook Air with 24 GB memory. The currently
   installed PyTorch environments expose neither CUDA nor MPS, so full local
   training would run on CPU.
3. The system-default Python is 3.14, while the baseline dependency constraints
   and older YOLOv5-style code are safer to reproduce in an isolated Python 3.11
   environment.
4. The documented training command supplies no `.pt` weights, so the strict
   baseline starts from random initialization. Introducing pretrained weights
   would be a separate improvement experiment.
5. `test.py` filters predictions through matches to ground-truth boxes before
   writing class counts. Since the test split has no labels, this logic cannot
   generate a meaningful submission. Training reproduction and submission
   repair will remain separate scopes.

## Reproduction Architecture

The reproduction has two execution levels:

### Level 1: Local Smoke Test

Create an isolated Python 3.11 environment and a deterministic miniature data
split derived from the existing training and validation sets. Run one epoch on
CPU to verify:

- configuration and paths resolve correctly;
- images and labels load without corruption;
- class IDs and normalized boxes pass validation;
- forward and backward passes complete;
- validation returns P, R, mAP50, and mAP50-95;
- `last.pt` and `best.pt` are saved;
- a saved checkpoint can be loaded for a second validation run.

The smoke dataset is only an execution check. Its metrics are not baseline
quality measurements.

### Level 2: Full Baseline Run

Run the original YOLOv5s architecture on the full 7,000/2,000 train/validation
split using:

- random initialization;
- 30 epochs;
- 640 × 640 input;
- batch size 16 where memory permits;
- SGD and the supplied `hyper_parameter.yaml`;
- seed 0;
- the supplied data augmentations;
- model selection using the baseline fitness function dominated by
  mAP50-95.

Because the local host has no accelerator exposed, the preferred full-run
target is a CUDA machine. The local reproduction will still produce the exact
command and environment metadata needed to transfer the run.

## Components and Boundaries

### Environment Definition

An isolated environment definition records Python and package versions without
changing the user's global Python installations. It must support a CPU smoke
run locally and the same code on CUDA.

### Dataset Configuration

The canonical dataset YAML will resolve paths from the repository layout rather
than from a developer-specific absolute path. A separate smoke-test YAML will
reference a generated miniature split so the full dataset remains untouched.

### Dataset Audit

A small read-only audit command will verify:

- expected image and label counts;
- matching stems for labeled splits;
- label rows contain five fields;
- class IDs are integers in `[0, 8]`;
- normalized box values are finite and within valid bounds;
- widths and heights are positive.

Failures stop training and report the offending files.

### Training and Validation

The existing `train.py` and `val.py` remain the authoritative baseline entry
points. Changes should be limited to compatibility or correctness fixes needed
to execute the documented workflow. Model architecture and supplied
hyperparameters must not be tuned during reproduction.

### Run Artifacts

Each run uses a distinct name and produces:

- effective command and configuration;
- `last.pt` and `best.pt`;
- validation metrics;
- aggregate F1 computed from validation precision and recall;
- per-class metrics from the final `best.pt` validation;
- timing and device information;
- generated diagnostic plots when supported.

Smoke and full-run outputs must never share an output directory.

## Data Flow

1. Audit the source dataset.
2. Generate or reference the smoke split without modifying source images or
   labels.
3. Load the selected dataset YAML.
4. Build the YOLOv5s detector with nine output classes.
5. Train from random initialization using the supplied hyperparameters.
6. Validate after each epoch.
7. Save `last.pt` and update `best.pt` when fitness improves.
8. Reload `best.pt` and run an explicit validation pass.
9. Record the metrics and environment needed to reproduce the run elsewhere.

## Error Handling

- Missing images, labels, configuration, or dependencies fail before training.
- Invalid labels report exact file paths and line numbers.
- Unsupported device requests fail with a clear recommendation instead of
  silently falling back.
- Non-finite loss aborts the run and preserves logs for diagnosis.
- A smoke run is successful only if checkpoints exist and the saved checkpoint
  passes standalone validation.

## Verification Criteria

The reproduction phase is complete when:

1. The full dataset audit passes.
2. The one-epoch local smoke run completes end to end.
3. `best.pt` from the smoke run reloads successfully in `val.py`.
4. The exact 30-epoch full-run command is verified against the canonical
   dataset configuration.
5. If a CUDA machine is available, the full run completes and reports overall
   and per-class P, R, mAP50, and mAP50-95, plus aggregate F1.

Full-run metrics are not required to match an unpublished reference score
exactly, but all configuration, seed, dataset, and environment differences must
be recorded.

## Out of Scope

- pretrained-weight fine-tuning;
- alternative detectors;
- hyperparameter search;
- test-time augmentation or model ensembling;
- leaderboard submission-format repair;
- deployment as a real-time driving assistant.

These become follow-up phases after the strict baseline has been reproduced.
