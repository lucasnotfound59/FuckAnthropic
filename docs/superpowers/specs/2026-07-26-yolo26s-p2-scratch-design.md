# YOLO26s-P2 Scratch Training and Submission Design

## Goal

Build an independent YOLO26s-P2 experiment package for the nine-class BDD100K
subset. The model must train from random initialization, support controlled
architecture and hyperparameter iteration, select checkpoints against both
standard detection metrics and the competition's F1 objective, and generate a
validated competition submission.

The new implementation lives at:

`26SummerHackathon_GenAI/yolo26s_p2`

It is a sibling of `26SummerHackathon_GenAI/baseline`. It must not modify,
import, or write artifacts into the supplied baseline.

## Constraints

- Use the Ultralytics training engine and a local YOLO26s-P2 architecture YAML.
- Do not download or load COCO or other public pretrained weights.
- A new training run must start from a model YAML and random initialization.
- A `.pt` file is accepted only when explicitly resuming a checkpoint produced
  by this project.
- Use the existing repository dataset without modifying source images or labels.
- Preserve the nine-class order:
  `person`, `rider`, `car`, `truck`, `bus`, `motorcycle`, `bicycle`,
  `traffic light`, `traffic sign`.
- Keep every experiment isolated and preserve its effective configuration,
  environment metadata, metrics, and checkpoints.
- Make GPU-dependent settings overridable because the final CUDA server is not
  known yet.

## Repository Context

The dataset is stored at the repository root under `bdd100k_selected`:

- 7,000 training images and labels;
- 2,000 validation images and labels;
- 1,000 test images without annotation text files;
- nine object classes in YOLO text format.

The training labels are imbalanced. Cars dominate the training instances while
bus, motorcycle, truck, and rider have substantially fewer instances. Parameter
experiments must therefore report per-class metrics and include class weighting
as an explicit, isolated variable.

The supplied baseline is a local YOLOv5-style implementation trained from
random initialization. It remains the comparison target but is not a dependency
of the new package.

## Architecture

The detector is YOLO26s-P2:

- YOLO26 end-to-end detection mode;
- scale `s`;
- P2/4, P3/8, P4/16, and P5/32 detection outputs;
- nine output classes;
- 9,765,856 parameters and 27.8 GFLOPs according to the official `s`-scale
  architecture summary.

The P2 output is the primary architectural motivation because the road scenes
contain many small traffic lights, traffic signs, riders, and distant vehicles.
The model architecture is copied from the official AGPL-3.0 configuration with
the license notice retained and the scale made explicit.

References:

- <https://docs.ultralytics.com/models/yolo26>
- <https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/26/yolo26-p2.yaml>
- <https://docs.ultralytics.com/guides/yolo26-training-recipe>

## Package Layout

```text
26SummerHackathon_GenAI/yolo26s_p2/
├── README.md
├── pyproject.toml
├── configs/
│   ├── data.yaml
│   ├── model.yaml
│   ├── smoke.yaml
│   └── scratch_ref.yaml
├── yolo26_experiment/
│   ├── __init__.py
│   ├── config.py
│   ├── data_audit.py
│   ├── initialization.py
│   ├── metrics.py
│   ├── checkpoints.py
│   ├── submission.py
│   └── run_metadata.py
├── train.py
├── evaluate.py
├── experiment.py
├── tune_thresholds.py
├── submit.py
└── tests/
    ├── test_config.py
    ├── test_data_audit.py
    ├── test_initialization.py
    ├── test_model_architecture.py
    ├── test_metrics.py
    └── test_submission.py
```

Generated environments, caches, runs, checkpoints, plots, and submissions are
ignored by Git.

## Component Responsibilities

### Configuration

`config.py` loads a checked YAML experiment configuration, resolves all paths
relative to the project or repository root, validates supported keys, and
applies explicit command-line overrides. It writes the fully resolved
configuration into the run directory before training begins.

Configuration inheritance is limited to one base file plus one experiment
overlay. This permits single-factor experiments without introducing a complex
configuration framework.

### Dataset Audit

`data_audit.py` validates split counts, image-label stem matching, decodable
images, five-field YOLO label rows, integer class IDs in `[0, 8]`, finite
normalized coordinates, and positive boxes within normalized bounds. Test
images are required, while test labels are not.

An audit failure reports exact files and label line numbers and stops training.

### Initialization Guard

`initialization.py` distinguishes a new run from a resume:

- a new run accepts only this project's `configs/model.yaml`;
- it constructs `YOLO(model_yaml, task="detect")`;
- it rejects `.pt`, URLs, model aliases, and paths outside the project;
- a resume accepts only a local `last.pt` whose stored project metadata and
  architecture fingerprint match this package.

This guard prevents accidental pretrained-weight downloads. The run metadata
records the model source, random seed, initial parameter checksum, and the fact
that no pretrained checkpoint was loaded.

### Training

`train.py` performs the preflight audit, constructs the model, snapshots
metadata, and delegates training to Ultralytics. It supports a deterministic
seed, AMP, EMA, early stopping, periodic checkpoints, single-GPU or multi-GPU
devices, and explicit resume.

The first full reference run uses:

- 640-pixel input;
- 200 maximum epochs and early-stopping patience of 50 epochs;
- MuSGD with cosine learning-rate scheduling;
- `lr0=0.005`, `lrf=0.01`, `momentum=0.948`, and
  `weight_decay=0.0005`;
- three warmup epochs;
- `box=9.83`, `cls=0.65`, and `dfl=0.96`;
- road-scene-safe augmentation with no vertical flip;
- `mosaic=1.0`, `mixup=0.05`, `scale=0.9`, `translate=0.1`,
  `fliplr=0.5`, `flipud=0.0`, `degrees=0.0`, `shear=0.0`, and
  `perspective=0.0`;
- a 20-epoch mosaic-free stabilization phase;
- `cls_pw=0.0` so architecture replacement is measured before class weighting;
- seed 0 with deterministic mode enabled.

Batch size, workers, caching, and device IDs remain server overrides. Their
effective values are saved.

The public YOLO26 recipe is a starting reference, not a claimed reproduction.
Official YOLO26 checkpoints used intermediate initialization and internal
parameters unavailable in the public trainer. This project therefore treats
all training parameters as hypotheses to validate on this dataset.

### Checkpoint Selection

The run preserves:

- `last.pt` for recovery;
- `best_map.pt` selected by the standard Ultralytics detection fitness;
- `best_detection_f1.pt` selected by standard validation detection F1;
- `best_count_f1.pt` selected after comparing the count F1 of
  `best_map.pt` and `best_detection_f1.pt`.

The F1 checkpoint callback must operate on completed validation metrics and copy
a complete, reloadable checkpoint atomically. It records the selected epoch,
precision, recall, F1, and confidence operating point.

### Evaluation and Threshold Tuning

`evaluate.py` reloads a checkpoint and produces:

- aggregate and per-class precision, recall, F1, mAP50, and mAP50-95;
- confusion matrix and PR/F1 curves;
- inference timing;
- JSON and human-readable summaries.

`tune_thresholds.py` separates two objectives:

1. detection F1, which requires a correct class and bounding-box match;
2. competition count F1, which compares predicted and ground-truth class
   multiplicities per image.

For count F1, true positives are the per-image, per-class minimum of predicted
and true counts. Excess predictions are false positives and missing predictions
are false negatives. The report includes global micro F1 and per-class F1.

YOLO26 end-to-end inference uses its one-to-one head without NMS, so no NMS IoU
parameter is tuned. Threshold tuning first searches one global confidence
threshold and then searches nine per-class confidence thresholds using
coordinate descent. The validation set is the only source of threshold choices;
the test set is never used for tuning. Both checkpoint candidates are tuned
independently; the higher count-F1 candidate becomes `best_count_f1.pt`.

### Submission

`submit.py` loads `best_count_f1.pt`, uses the frozen threshold configuration,
runs inference on every test image, and converts detections into nine class
counts. Rows are deterministic and ordered by image filename.

The concrete example in the supplied competition instructions uses:

```csv
pic_name,results
cabc30fc-fd79926f.jpg,4;1;5;1;0;1;0;0;2
```

The prose in the same instructions instead names the second column
`correct_predictions`, and separately says no-detection rows should have an
empty prediction field. Because those instructions conflict, the generator
produces both header variants:

- `submission_results.csv`;
- `submission_correct_predictions.csv`.

It also supports both empty conventions:

- blank second field;
- `0;0;0;0;0;0;0;0;0`.

The default follows the concrete example: column `results`, nine semicolon
separated counts, and zero counts for an image with no detections. If an
official sample submission becomes available, its schema overrides this
default without changing inference.

Submission validation requires:

- exactly two CSV columns;
- exactly one row per test image;
- all 1,000 test filenames present exactly once;
- filename including `.jpg`;
- no unexpected filename;
- every non-empty result containing exactly nine non-negative integers;
- count order matching class IDs `0` through `8`;
- successful CSV round-trip parsing.

## Experiment Sequence

Experiments use the same train/validation split and seed unless the experiment
explicitly tests seed stability.

1. `smoke`: a tiny deterministic subset, reduced resolution, and one epoch.
2. `scratch_ref`: YOLO26s-P2 at 640 pixels with the reference scratch recipe.
3. `class_weight_025`: change only `cls_pw` from `0.0` to `0.25`.
4. `imgsz_768`: change only image size from 640 to 768.
5. `optimizer_lr`: compare a bounded set of MuSGD learning-rate and
   weight-decay pairs.
6. `augmentation`: adjust mosaic, scale, and close-mosaic values based on the
   small-object and overfitting diagnostics.
7. `best_recipe`: combine only individually validated improvements.
8. `seed_check`: repeat the best recipe with additional seeds if GPU budget
   permits.

Experiment comparison prioritizes competition count F1, then detection F1 and
mAP50-95. Training time and memory use are reported but are not the primary
objective.

## Data Flow

1. Load and validate the experiment configuration.
2. Audit source images and labels without modifying them.
3. Set deterministic seeds and record environment metadata.
4. Construct YOLO26s-P2 from the local model YAML.
5. Verify nine classes, P2–P5 outputs, and the absence of loaded checkpoint
   weights.
6. Train and validate each epoch.
7. Update `last.pt`, `best_map.pt`, and `best_detection_f1.pt`.
8. Reload selected checkpoints for standalone evaluation.
9. Tune both candidates' global and per-class thresholds using validation
   predictions, then select `best_count_f1.pt`.
10. Freeze the chosen threshold configuration.
11. Infer all test images and generate both submission schema variants.
12. Validate each CSV before reporting it as ready to upload.

## Error Handling

- Missing dependencies, images, labels, or configuration fail before model
  allocation.
- Requested CUDA devices must exist; the full run does not silently fall back
  to CPU.
- Out-of-memory errors report the effective batch and image size and recommend
  the exact override to retry.
- Non-finite loss aborts training and preserves the run directory for diagnosis.
- Resume rejects foreign, incompatible, or completed checkpoints.
- A checkpoint is considered valid only after it reloads and completes a
  validation pass.
- Submission generation fails on missing test images, duplicated rows, schema
  violations, negative counts, or an unvalidated threshold file.

## Testing

Unit tests cover:

- repository-relative data path resolution from arbitrary working directories;
- configuration overlay rules and unknown-key rejection;
- pretrained source rejection and valid local resume acceptance;
- dataset audit success and precise invalid-label failures;
- model scale `s`, nine classes, and P2/P3/P4/P5 detection outputs;
- F1 edge cases and count-F1 calculations;
- global and per-class threshold selection on synthetic predictions;
- submission class ordering, empty-image behavior, both headers, stable row
  ordering, and schema failures.

Integration checks cover:

- CPU model construction and one forward pass;
- one-epoch smoke training on a deterministic miniature split;
- `last.pt` reload and standalone validation;
- a miniature test inference producing a valid CSV.

## Completion Criteria

Before receiving the CUDA server:

1. The package installs in an isolated environment.
2. All unit tests pass.
3. The full dataset audit passes.
4. YOLO26s-P2 constructs from YAML with random initialization.
5. The one-epoch smoke run and checkpoint reload succeed.
6. A miniature submission passes schema validation.
7. README commands cover new training, resume, evaluation, threshold tuning,
   single-GPU execution, multi-GPU execution, and submission generation.

After receiving the CUDA server:

1. The full `scratch_ref` run completes.
2. `best_map.pt`, `best_detection_f1.pt`, and `best_count_f1.pt` reload and
   validate.
3. Standard and count-based overall and per-class metrics are recorded.
4. Controlled experiments identify a `best_recipe`.
5. The selected test submission contains all 1,000 test images and passes the
   schema validator.
6. Results are compared with the YOLOv5 baseline on the same validation split.

## Out of Scope

- public pretrained weights;
- external training images;
- ensembles or test-time augmentation in the first implementation;
- modifying the supplied baseline;
- deployment to a real-time application;
- claiming exact reproduction of the private YOLO26 pretraining pipeline.
