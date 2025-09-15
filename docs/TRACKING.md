# Cyclone Tracking

This document describes the cyclone-centre tracking task built on top of VideoMAEv2.

## Dataset
- In tracking, we do coordinate regression (x, y), not classification.
- [`MedicanesTrackDataset`](../dataset/datasets.py) loads only tiles labelled as cyclones (`label == 1`).
- Each video spans 80 minutes and is aligned so the last frame ends on the hour, as in classification.
- Training targets are the cyclone centre coordinates from the **last frame** of each clip.

## Model
- [`create_tracking_model`](../models/tracking_model.py) builds a ViT backbone from a classification checkpoint passed with `--init_ckpt`.
- We build the backbone with `num_classes=0`, which sets `VisionTransformer.head = nn.Identity()`.
- The binary classification head is removed and replaced by a LayerNorm + Linear layer predicting the `(x, y)` coordinates.
- During checkpoint loading, only backbone weights are kept; classification head weights (`head.weight`/`head.bias`) from the checkpoint are dropped. 
- After loading, `create_tracking_model` replaces the head with a dedicated `RegressionHead(embed_dim -> 2)`, that runs and outputs a tensor of shape `[B, 2]`.

## Training
- The training loop optimizes an `MSELoss()` between predicted coords and the ground-truth `(x, y)`.
- [`tracking.py`](../tracking.py) configures distributed training analogously to the classification pipeline.
- [`engine_for_tracking.py`](../engine_for_tracking.py) implements `train_one_epoch` and `evaluate` using mean-squared error.

- During training, a new `checkpoint-best.pth` is saved whenever validation loss improves.

## Example
```bash
python tracking.py 
```
