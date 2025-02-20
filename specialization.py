import numpy as np
import os
import datetime
import time
from PIL import Image
import json
import torch
import torch.nn.functional as F

from utils import multiple_pretrain_samples_collate
from functools import partial

import utils
from utils import NativeScalerWithGradNormCount as NativeScaler
from utils import multiple_pretrain_samples_collate
from optim_factory import create_optimizer
from dataset import build_pretraining_dataset
from run_mae_pretraining import get_model
from engine_for_pretraining import train_one_epoch, test
from arguments import prepare_args, Args  # NON TOGLIERE: serve a torch.load per caricare il mio modello addestrato
from model_analysis import get_dataloader, get_dataset_dataloader



def launch_specialization_training():
    args = prepare_args()
    device = torch.device(args.device)

    # LOAD MODEL
    pretrained_model = get_model(args)
    pretrained_model.to(device)
    model_without_ddp = pretrained_model
    n_parameters = sum(p.numel() for p in pretrained_model.parameters() if p.requires_grad)
    #print("Model = %s" % str(model_without_ddp))
    print('number of params: {} M'.format(n_parameters / 1e6))

    # LOAD DATASET
    patch_size = pretrained_model.encoder.patch_embed.patch_size
    data_loader_train, dataset_train = get_dataset_dataloader(args, patch_size)
    args.data_path = './test.csv'
    data_loader_test = get_dataloader(args, patch_size)


    # SET HYPER PARAMETERS
    num_tasks = utils.get_world_size()
    total_batch_size = args.batch_size * num_tasks
    num_training_steps_per_epoch = len(dataset_train) // total_batch_size
    # scale the lr
    args.lr = args.lr * total_batch_size / 256
    args.min_lr = args.min_lr * total_batch_size / 256
    args.warmup_lr = args.warmup_lr * total_batch_size / 256
    print("LR = %.8f" % args.lr)
    print("Batch size = %d" % total_batch_size)
    print("Number of training steps = %d" % num_training_steps_per_epoch)
    print("Number of training examples per epoch = %d" %
          (total_batch_size * num_training_steps_per_epoch))
    # if args.distributed:
    #     model = torch.nn.parallel.DistributedDataParallel(
    #         pretrained_model, device_ids=[args.gpu], find_unused_parameters=False)
    #     model_without_ddp = model.module

    optimizer = create_optimizer(args, model_without_ddp)
    loss_scaler = NativeScaler()

    print("Use step level LR & WD scheduler!")
    lr_schedule_values = utils.cosine_scheduler(
        args.lr,
        args.min_lr,
        args.epochs,
        num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs,
        warmup_steps=args.warmup_steps,
    )
    if args.weight_decay_end is None:
        args.weight_decay_end = args.weight_decay
    wd_schedule_values = utils.cosine_scheduler(args.weight_decay,
                                                args.weight_decay_end,
                                                args.epochs,
                                                num_training_steps_per_epoch)
    print("Max WD = %.7f, Min WD = %.7f" %
          (max(wd_schedule_values), min(wd_schedule_values)))

    utils.auto_load_model(
        args=args,
        model=pretrained_model,
        model_without_ddp=model_without_ddp,
        optimizer=optimizer,
        loss_scaler=loss_scaler)
    torch.cuda.empty_cache()

    # SET THE LOGGING
    global_rank = utils.get_rank()
    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = utils.TensorboardLogger(log_dir=args.log_dir)
    else:
        log_writer = None

    ############################################
    ########################### START TRAINING #
    ############################################
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        if log_writer is not None:
            log_writer.set_step(epoch * num_training_steps_per_epoch)
        train_stats = train_one_epoch(
            pretrained_model,
            data_loader_train,
            optimizer,
            device,
            epoch,
            loss_scaler,
            args.clip_grad,
            log_writer=log_writer,
            start_steps=epoch * num_training_steps_per_epoch,
            lr_schedule_values=lr_schedule_values,
            wd_schedule_values=wd_schedule_values,
            patch_size=patch_size[0],
            normlize_target=args.normlize_target)
        if args.output_dir:
            _epoch = epoch + 1
            if _epoch % args.save_ckpt_freq == 0 or _epoch == args.epochs:
                utils.save_model(
                    args=args,
                    model=pretrained_model,
                    model_without_ddp=model_without_ddp,
                    optimizer=optimizer,
                    loss_scaler=loss_scaler,
                    epoch=epoch)

        log_stats = {
            **{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': epoch,
            'n_parameters': n_parameters
        }

        if args.output_dir and utils.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

        if epoch % args.testing_epochs == 0:
            test_stats = test(pretrained_model, data_loader_test, device, epoch,
                        patch_size=patch_size[0], normlize_target=args.normlize_target, log_writer=log_writer)
            test_log_stats = {**{f'test_{k}': v for k, v in test_stats.items()}, 'epoch': epoch, 'n_parameters': n_parameters}
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(test_log_stats) + "\n")




    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))




if __name__ == '__main__':
    launch_specialization_training()