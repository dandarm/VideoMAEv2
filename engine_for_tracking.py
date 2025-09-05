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
    for samples, target, _ in metric_logger.log_every(data_loader, 20, header):
        samples = samples.to(device)
        target = target.to(device)

        output = model(samples)
        loss = criterion(output, target)

        optimizer.zero_grad()
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
    for samples, target, _ in metric_logger.log_every(data_loader, 20, header):
        samples = samples.to(device)
        target = target.to(device)

        output = model(samples)
        loss = criterion(output, target)

        metric_logger.update(loss=loss.item())

    metric_logger.synchronize_between_processes()
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
