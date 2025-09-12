"""Training utilities for cyclone center tracking."""
from __future__ import annotations

from typing import Iterable, Optional

import torch

import utils


def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    max_norm: float = 0,
    log_writer: Optional[utils.TensorboardLogger] = None,
) -> dict:
    """Train for a single epoch."""
    # ensure different shuffles across workers when using DistributedSampler
    if hasattr(data_loader, "sampler") and hasattr(data_loader.sampler, "set_epoch"):
        data_loader.sampler.set_epoch(epoch)

    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = f"Epoch: [{epoch}]"
    for batch_idx, (samples, target, paths) in enumerate(metric_logger.log_every(data_loader, 20, header)):
        samples = samples.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # Guard: check inputs/targets are finite
        if not torch.isfinite(samples).all():
            print(f"[WARN] Non-finite samples at batch {batch_idx}: nan={torch.isnan(samples).sum().item()}, inf={torch.isinf(samples).sum().item()}")
        if not torch.isfinite(target).all():
            print(f"[WARN] Non-finite target at batch {batch_idx}: values={target}")

        output = model(samples)
        # Shape sanity check
        if output.ndim != target.ndim or output.shape[-1] != target.shape[-1]:
            try:
                print(f"[WARN] Output/target shape mismatch at batch {batch_idx}: output={tuple(output.shape)}, target={tuple(target.shape)}")
            except Exception:
                pass

        # Guard: check model outputs are finite
        if not torch.isfinite(output).all():
            o_nan = torch.isnan(output).sum().item()
            o_inf = torch.isinf(output).sum().item()
            o_min = float(torch.nanmin(output).detach().cpu()) if o_nan == 0 else float('nan')
            o_max = float(torch.nanmax(output).detach().cpu()) if o_nan == 0 else float('nan')
            print(f"[WARN] Non-finite output at batch {batch_idx}: nan={o_nan}, inf={o_inf}, min={o_min}, max={o_max}")

        loss = criterion(output, target)

        # Guard: skip update on non-finite loss to avoid poisoning training
        if not torch.isfinite(loss):
            t_min = float(torch.nanmin(target).detach().cpu()) if torch.isfinite(target).any() else float('nan')
            t_max = float(torch.nanmax(target).detach().cpu()) if torch.isfinite(target).any() else float('nan')
            print(f"[ERROR] Non-finite loss at batch {batch_idx}: loss={loss.item() if loss.numel()==1 else loss}\n"
                  f"        target[min,max]=({t_min:.4f},{t_max:.4f}), paths[0]={paths[0] if isinstance(paths, (list, tuple)) and len(paths)>0 else paths}")
            optimizer.zero_grad(set_to_none=True)
            continue

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss.item())
        if log_writer is not None:
            log_writer.update(loss=loss.item(), head="loss")
            log_writer.set_step()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    device: torch.device,
) -> dict:
    """Evaluate the model."""
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"
    for batch_idx, (samples, target, paths) in enumerate(metric_logger.log_every(data_loader, 20, header)):
        samples = samples.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        output = model(samples)
        loss = criterion(output, target)
        if not torch.isfinite(loss):
            print(f"[ERROR][EVAL] Non-finite loss at batch {batch_idx}: loss={loss.item() if loss.numel()==1 else loss}, paths[0]={paths[0] if isinstance(paths, (list, tuple)) and len(paths)>0 else paths}")

        metric_logger.update(loss=loss.item())

    metric_logger.synchronize_between_processes()
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
