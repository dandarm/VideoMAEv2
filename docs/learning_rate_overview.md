# Learning-Rate Mechanics in VideoMAEv2

## 1. Layer-Wise Scaling
Layer-wise decay hinges on the `lr_scale` attribute attached to every optimizer parameter group during model setup. Inside `optim_factory.get_parameter_groups`, each parameter is assigned to either a `decay` or `no_decay` group and optionally tagged with a layer index derived from `LayerDecayValueAssigner`. If that assigner is provided, it returns a scale factor per transformer depth; otherwise the scale defaults to `1.0`, which means no layer-wise adjustment. Lower (input-side) layers usually get smaller scales so they retain generic features, while upper layers receive larger scales that let them react faster to task-specific supervision. The beauty of the setup is that the rest of the training loop remains agnostic—`param_group["lr"]` already contains the right value once the scheduler multiplies by `lr_scale`.

You can inspect or override the scales at runtime:

```python
# Inspect the automatically generated scales
for idx, group in enumerate(optimizer.param_groups):
    print(f"group {idx}: decay={group['weight_decay']}, lr_scale={group['lr_scale']}")

# Force a uniform schedule (useful for aggressive fine-tuning)
uniform_scale = 1.0
for group in optimizer.param_groups:
    group["lr_scale"] = uniform_scale
```

For custom decay profiles, supply your own assigner before calling `create_optimizer`:

```python
from optim_factory import LayerDecayValueAssigner, create_optimizer

scales = [0.65] * (num_blocks // 2) + [0.85] * (num_blocks // 2)
assigner = LayerDecayValueAssigner(scales)
optimizer = create_optimizer(
    args,
    model_without_ddp,
    get_num_layer=assigner.get_layer_id,
    get_layer_scale=assigner.get_scale,
)
```

## 2. Step-Level Scheduling And The Recent Fix
Originally, `tracking.py` computed cosine schedules (`lr_schedule_values` and `wd_schedule_values`) but never forwarded them to the training loop, leaving the optimizer stuck at its initial LR. The fix wires those arrays through `train_one_epoch`, supplying the global step so every batch applies the correct value. The call site now looks like this:

```python
train_stats = train_one_epoch(
    model=model,
    criterion=criterion,
    data_loader=train_loader,
    optimizer=optimizer,
    device=device,
    epoch=epoch,
    start_steps=epoch * num_training_steps_per_epoch,
    lr_schedule_values=lr_schedule_values,
    wd_schedule_values=wd_schedule_values,
    num_training_steps_per_epoch=num_training_steps_per_epoch,
)
```

Inside `engine_for_tracking.train_one_epoch`, the first instructions of the batch loop compute `global_step` and use it to index the schedules. Each parameter group is then updated before the forward pass, ensuring that any `lr_scale` multiplier is respected:

```python
if lr_index is not None or wd_index is not None:
    for param_group in optimizer.param_groups:
        if lr_index is not None:
            scale = param_group.get("lr_scale", 1.0)
            param_group["lr"] = lr_schedule_values[lr_index] * scale
        if wd_index is not None and param_group.get("weight_decay", 0) > 0:
            param_group["weight_decay"] = wd_schedule_values[wd_index]
```

Classification and pretraining paths already performed a similar hand-off, so aligning tracking with them keeps every training workflow consistent and guarantees that the cosine schedules defined in `tracking.py` actually drive optimization.
