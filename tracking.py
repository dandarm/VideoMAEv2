"""Entry point for cyclone centre tracking training."""
import argparse
import json
import os
import time
import random

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, DistributedSampler
from arguments import prepare_tracking_args
from optim_factory import create_optimizer
from dataset.datasets import MedicanesTrackDataset
from dataset.data_manager import DataManager
from engine_for_tracking import train_one_epoch, evaluate
from models.tracking_model import create_tracking_model
import utils
from utils import setup_for_distributed


utils.suppress_transformers_pytree_warning()


def _format_log_value(key: str, value):
    """Keep learning-rate precision while rounding other floats for logs."""
    if not isinstance(value, float):
        return value
    if "lr" in key.lower():
        return float(f"{value:.8g}")
    return round(value, 4)


def launch_tracking(terminal_args: argparse.Namespace) -> None:
    """Launch the training process for the tracking task."""
    args = prepare_tracking_args(machine=terminal_args.on)

    # seed = args.seed
    # torch.manual_seed(seed)
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed(seed)
    # np.random.seed(seed)
    # random.seed(seed)
    # cudnn.benchmark = True

    # region --------------------------- distributed setup ---------------------------
    rank, local_rank, world_size, _, _ = utils.get_resources()
    if world_size > 1:
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        args.distributed = True
    else:
        args.distributed = False
    if args.device == "cuda":
        torch.cuda.set_device(local_rank)
        args.gpu = local_rank
        args.world_size = world_size
        args.rank = rank
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")
        args.gpu = None
        args.world_size = 1
        args.rank = 0

    if args.log_dir and not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    setup_for_distributed(rank == 0)
    # endregion ------------------------ distributed setup ------------------------

    #region logging
    if args.log_dir and not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    setup_for_distributed(rank == 0)
    # endregion logging

    # ------------------------------- datasets via DataManager -------------------------------
    train_m = DataManager(is_train=True, args=args, type_t='supervised', world_size=world_size, rank=rank)
    test_m = DataManager(is_train=False, args=args, type_t='supervised', world_size=world_size, rank=rank)
    val_m = None
    if getattr(args, 'val_path', None):
        val_m = DataManager(is_train=False, args=args, type_t='supervised', world_size=world_size, rank=rank, specify_data_path=args.val_path)

    train_loader = train_m.get_tracking_dataloader(args)
    test_loader = test_m.get_tracking_dataloader(args)
    val_loader = val_m.get_tracking_dataloader(args) if val_m is not None else None

    # region ------------------------------- model ---------------------------------
    #model_kwargs = args.__dict__.copy()
    #for k in ["model", "init_ckpt", "nb_classes"]:
    #    model_kwargs.pop(k, None)
    model = create_tracking_model(
        args.model,
        **args.__dict__
    )

    # Informative print: confirm regression head is active
    try:
        from models.tracking_model import RegressionHead  # local import for isinstance check
        head_is_reg = isinstance(model.head, RegressionHead)
    except Exception:
        head_is_reg = False
    print(f"[INFO] Model built: {args.model} | Head: {type(model.head).__name__} | Regression active: {head_is_reg}")

    model.to(device)
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], output_device=args.gpu
        )
        model_without_ddp = model.module

    # endregion ------------------------------ model ---------------------------------

    criterion = nn.MSELoss()
    optimizer = create_optimizer(args, model_without_ddp)

    total_batch_size = args.batch_size * world_size
    num_training_steps_per_epoch = train_m.dataset_len// total_batch_size
    if args.weight_decay_end is None:
        args.weight_decay_end = args.weight_decay
    lr_schedule_values = None
    wd_schedule_values = None
    if getattr(args, "disable_scheduler", False):
        print("[INFO] Learning-rate scheduler disabled: optimizer will use fixed lr.")
    else:
        lr_schedule_values = utils.cosine_scheduler(
            args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
            warmup_epochs=args.warmup_epochs, start_warmup_value=args.warmup_lr, warmup_steps=args.warmup_steps)
        print(f"lr_schedule_values {lr_schedule_values}")
        wd_schedule_values = utils.cosine_scheduler(
            args.weight_decay, args.weight_decay_end,
            args.epochs, num_training_steps_per_epoch
        )
        print(f"wd_schedule_values {wd_schedule_values}")



    best_loss = float("inf")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        print(f"start epoch{epoch}")
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
        print("train step")
        # Evaluate on test and optionally validation
        test_stats = evaluate(model, criterion, test_loader, device)
        val_stats = evaluate(model, criterion, val_loader, device) if val_loader is not None else None
        print("eval step")
        test_loss = test_stats.get("loss", float("inf"))
        val_loss = val_stats.get("loss", float("inf")) if val_stats is not None else float("inf")
        monitor_loss = min(test_loss, val_loss)
        if args.output_dir and monitor_loss < best_loss and args.rank == 0 and epoch > args.start_epoch_for_saving_best_ckpt:
            best_loss = monitor_loss
            checkpoint_path = os.path.join(args.output_dir, "checkpoint-tracking-best.pth")
            torch.save(
                {
                    "model": model_without_ddp.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "args": args.__dict__,
                },
                checkpoint_path,
            )
            print(f"[INFO] Best checkpoint saved at {checkpoint_path}")

        log_stats = {
            "epoch": epoch,
            **{f"train_{k}": v for k, v in train_stats.items()},
            **{f"test_{k}": v for k, v in test_stats.items()},
        }
        if val_stats is not None:
            log_stats.update({f"val_{k}": v for k, v in val_stats.items()})

        if args.output_dir and args.rank == 0:
            log_path = os.path.join(args.output_dir, "log.txt")
            with open(log_path, "a") as f:
                f.write(
                    json.dumps(
                        {k: _format_log_value(k, v) for k, v in log_stats.items()}
                    )
                    + "\n"
                )


    total_time = time.time() - start_time
    total_time_str = str(time.strftime('%H:%M:%S', time.gmtime(total_time)))
    print(f"Training time {total_time_str}")

    # ultimo salvataggio per riprendere da dove abbiamo lasciato
    last_checkpoint_path = os.path.join(args.output_dir, "last_checkpoint-tracking.pth")
    torch.save(
                {
                    "model": model_without_ddp.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "args": args.__dict__,
                },
                last_checkpoint_path,
            )



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Cyclone centre tracking")
    parser.add_argument('--on',
        type=str,
        default='leonardo',
        #metavar='NAME',
        help='[ewc, leonardo]'
    ),
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    launch_tracking(parsed)
