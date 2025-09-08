"""Entry point for cyclone centre tracking training."""
import argparse
import json
import os
import time

import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader, DistributedSampler
from arguments import prepare_finetuning_args
from optim_factory import create_optimizer
from dataset.tracking_dataset import MedicanesTrackDataset
from engine_for_tracking import train_one_epoch, evaluate
from models.tracking_model import create_tracking_model
import utils
from utils import setup_for_distributed


def launch_tracking(terminal_args: argparse.Namespace) -> None:
    """Launch the training process for the tracking task."""
    args = prepare_finetuning_args(machine=terminal_args.on)
    args.nb_classes = 2  # two regression outputs: lon and lat
    args.train_path = terminal_args.train_path
    args.test_path = terminal_args.test_path
    args.data_root = terminal_args.data_root
    if getattr(terminal_args, "init_ckpt", None):
        args.init_ckpt = terminal_args.init_ckpt

    seed = args.seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # --------------------------- distributed setup ---------------------------
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

    # ------------------------------- datasets -------------------------------
    train_dataset = MedicanesTrackDataset(
        args.train_path,
        data_root=args.data_root,
        clip_len=args.num_frames,
    )
    test_dataset = MedicanesTrackDataset(
        args.test_path,
        data_root=args.data_root,
        clip_len=args.num_frames,
    )

    sampler_train = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, shuffle=True
    )
    sampler_test = DistributedSampler(
        test_dataset, num_replicas=world_size, rank=rank, shuffle=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        sampler=sampler_train,
        persistent_workers=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
        sampler=sampler_test,
        persistent_workers=True,
    )

    # ------------------------------- model ---------------------------------
    model_kwargs = args.__dict__.copy()
    for k in ["model", "init_ckpt", "nb_classes"]:
        model_kwargs.pop(k, None)
    model = create_tracking_model(
        args.model,
        init_ckpt=getattr(args, "init_ckpt", ""),
        num_outputs=args.nb_classes,
        drop_path_rate=args.drop_path,
        **model_kwargs,
    )
    model.to(device)
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], output_device=args.gpu
        )
        model_without_ddp = model.module

    criterion = nn.MSELoss()
    optimizer = create_optimizer(args, model_without_ddp)

    best_loss = float("inf")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        train_stats = train_one_epoch(
            model, criterion, train_loader, optimizer, device, epoch
        )
        val_stats = evaluate(model, criterion, test_loader, device)

        val_loss = val_stats.get("loss", float("inf"))
        if args.output_dir and val_loss < best_loss and args.rank == 0:
            best_loss = val_loss
            checkpoint_path = os.path.join(args.output_dir, "checkpoint-best.pth")
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
            **{f"val_{k}": v for k, v in val_stats.items()},
        }
        if args.output_dir and args.rank == 0:
            log_path = os.path.join(args.output_dir, "log.txt")
            with open(log_path, "a") as f:
                f.write(
                    json.dumps(
                        {k: (round(v, 4) if isinstance(v, float) else v) for k, v in log_stats.items()}
                    )
                    + "\n"
                )

    if args.output_dir and args.rank == 0:
        ckpt = {
            "model": model_without_ddp.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": args.epochs,
            "args": args.__dict__,
        }
        torch.save(ckpt, os.path.join(args.output_dir, "checkpoint-last.pth"))
    total_time = time.time() - start_time
    total_time_str = str(time.strftime('%H:%M:%S', time.gmtime(total_time)))
    print(f"Training time {total_time_str}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Cyclone centre tracking")
    parser.add_argument("--train_path", required=True, help="CSV with training data")
    parser.add_argument("--test_path", required=True, help="CSV with test data")
    parser.add_argument("--data_root", default="", help="Root folder for video tiles")
    parser.add_argument("--init_ckpt", default="", help="Checkpoint to load model weights")
    parser.add_argument(
        "--on", default=None, help="Machine preset for argument utilities",
    )
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    launch_tracking(parsed)

