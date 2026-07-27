# RF-DETR-2XL BDD100K

This is the accuracy-first, non-YOLO training path. It uses RF-DETR-2XLarge,
whose backbone is DINOv2 and whose detector is based on Deformable DETR.

## License and environment

RF-DETR-2XL is distributed under the Platform Model License 1.0 and may require
a Roboflow account. Read and accept that license before using `--accept-pml`.
Use Python 3.10-3.12; the repository's current Python 3.13 environment is not
recommended for the PyTorch/CUDA training stack.

Create a clean environment and install a CUDA-compatible PyTorch build first,
then:

```bash
pip install "rfdetr[train,plus]>=1.8,<1.9"
```

## 1. Prepare the dataset layout

The source images are not duplicated on filesystems that support hard links.

```bash
python rfdetr/prepare_dataset.py
```

The 2000 source validation images are deterministically split into:

- 1500 `valid` images for checkpoint selection and threshold calibration
- 500 `holdout` images that training and threshold search never use

Expected total counts are 7000 train, 1500 valid, 500 holdout, and 1000 test.
The splitter uses label-count-balanced swaps so rare-class proportions remain
close between valid and holdout.

## 2. Train

```bash
python rfdetr/train.py --accept-pml
```

For AutoDL, the guarded launcher refuses to start until the license acceptance
environment variable is explicitly provided and then runs training in `screen`:

```bash
ACCEPT_RFDETR_PML=YES bash rfdetr/start_training.sh
```

The default run is deliberately aggressive:

- RF-DETR-2XLarge (126.9M parameters)
- 1360x1360 input for small traffic lights and signs; 1360 is divisible by the
  2XL patch/window factor of 40
- verified micro-batch 4 with gradient accumulation 4 (effective batch 16)
- BF16/FP16 automatic mixed precision
- EMA validation
- validation every epoch and periodic checkpointing every 5 epochs
- per-class validation metrics
- early stopping on EMA validation mAP
- reduced DINOv2 backbone learning rate
- DropPath, weight decay, and mild weather/lighting augmentations
- no random crop/scale jitter that can erase tiny objects

If the automatic memory probe is too optimistic, edit `train.py` to use a fixed
micro-batch and gradient accumulation while keeping the effective batch near 16.

Resume an interrupted run with:

```bash
python rfdetr/train.py --accept-pml \
  --resume rfdetr_runs/rfdetr_2xl_1360/checkpoint.pth
```

TensorBoard logs are written under the output directory.

## 3. Cache validation predictions

The aggressive inference path combines the 1360 full image with overlapping
720px tiles for person, rider, motorcycle, bicycle, traffic light, and traffic
sign. Class-aware NMS removes full/tile duplicates.

```bash
python rfdetr/cache_predictions.py --accept-pml \
  --images rfdetr_dataset/valid/images \
  --output rfdetr_prediction_cache/valid.jsonl

python rfdetr/cache_predictions.py --accept-pml \
  --images rfdetr_dataset/holdout/images \
  --output rfdetr_prediction_cache/holdout.jsonl
```

## 4. Tune thresholds without leaking into holdout

The competition submission contains counts, so threshold search maximizes
count-based micro-F1 directly rather than box mAP. If the organizer's scorer
uses macro-F1, append `--objective macro`:

```bash
python rfdetr/tune_thresholds.py \
  --calibration-cache rfdetr_prediction_cache/valid.jsonl \
  --calibration-labels rfdetr_dataset/valid/labels \
  --holdout-cache rfdetr_prediction_cache/holdout.jsonl \
  --holdout-labels rfdetr_dataset/holdout/labels
```

Only the reported holdout F1 is a trustworthy estimate. If calibration rises
but holdout falls, do not keep adding thresholds or post-processing rules.

## 5. Generate a submission

After threshold calibration:

```bash
python rfdetr/predict_submission.py --accept-pml
```

This uses the same full-image + tiled inference and writes
`submission_rfdetr.csv` plus a box-level `rfdetr_detections.csv`.

`submission_rfdetr.csv` is validated before the command succeeds. Its header is
exactly `pic_name,results`, and every non-empty `results` value contains nine
semicolon-separated counts in this order:

1. person
2. rider
3. car
4. truck
5. bus
6. motorcycle
7. bicycle
8. traffic light
9. traffic sign

The original BDD category 5 (`train`, the rail-vehicle class) is omitted by the
selected dataset. The remaining classes are reindexed, making `motorcycle`
submission class 5. An image with no accepted detections is still included and
uses nine zero counts: `image.jpg,0;0;0;0;0;0;0;0;0`. This avoids the competition
validator interpreting an empty CSV field as a null value.

The separate `rfdetr_detections.csv` is diagnostic only. It records each
detection's class, confidence, and bounding box and must not be uploaded as the
competition submission.

RF-DETR's built-in best checkpoint is selected by box mAP50-95. For the
competition, compare the per-epoch checkpoints with the exact count-based F1
scorer before choosing the final checkpoint.

The AutoDL data disk is currently 50GB. Keep the default
`--checkpoint-interval 5` unless the disk is expanded; saving a full 2XL
training checkpoint every epoch can exhaust the disk.

An F1 target of 0.95 is aspirational, not guaranteed. Treat it as reached only
if the untouched holdout also reaches it; calibration or training F1 alone is
not evidence of generalization.

## Accuracy refinement toward 0.95 count-F1

The stricter target is holdout **count-F1 >= 0.95**, not the dashboard's
box-matching F1. After the first stage has produced a stable best EMA
checkpoint:

```bash
python rfdetr/prepare_dataset_v2.py
```

This creates a hard-linked 8000/500/500 train/valid/holdout split. The original
500-image holdout remains untouched; 1000 previously labeled validation images
are promoted to training, and a new balanced 500-image calibration split is
selected from the remainder.

Copy the selected first-stage EMA checkpoint to
`rfdetr_runs/snapshots/refinement_source_best_ema.pth`, stop the first-stage
trainer, then run:

```bash
ACCEPT_RFDETR_PML=YES bash rfdetr/start_refinement.sh
```

The refinement stage uses a fresh optimizer, lower decoder/backbone learning
rates, lower label smoothing, the larger training set, EMA checkpointing, and
early stopping. Loss coefficients are not artificially reduced: a smaller
reported loss must come from better fitting rather than rescaling the loss.

The 0.95 target remains aspirational rather than guaranteed. Re-cache
calibration and holdout predictions from each candidate best EMA checkpoint,
then accept it only when the untouched holdout count-F1 reaches the target.
