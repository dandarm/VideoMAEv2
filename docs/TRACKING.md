# Cyclone Tracking

This document describes the cyclone-centre tracking task built on top of VideoMAEv2.

## Dataset
- [`MedicanesTrackDataset`](../dataset/tracking_dataset.py) loads only tiles labelled as cyclones (`label == 1`).
- Each video spans 80 minutes and is aligned so the last frame ends on the hour.
- Training targets are the cyclone centre coordinates from the **last frame** of each clip.

## Model
- [`create_tracking_model`](../models/tracking_model.py) builds a ViT backbone from a classification checkpoint passed with `--init_ckpt`.
- The binary classification head is removed and replaced by a LayerNorm + Linear layer predicting the `(x, y)` coordinates.

## Training
- [`tracking.py`](../tracking.py) configures distributed training analogously to the classification pipeline.
- [`engine_for_tracking.py`](../engine_for_tracking.py) implements `train_one_epoch` and `evaluate` using mean-squared error.
- During training, a new `checkpoint-best.pth` is saved whenever validation loss improves.

## Example
```bash
torchrun --nproc_per_node=8 tracking.py \
  --data_path <csv_file> \
  --init_ckpt <classification_checkpoint> \
  --output_dir <out_dir>
```
