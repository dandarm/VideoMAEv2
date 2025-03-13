import numpy as np
import os
import datetime
import time
import random
import json
import torch
import torch.nn.functional as F

from utils import multiple_pretrain_samples_collate
from functools import partial

import utils
from engine_for_pretraining import train_one_epoch, test
from arguments import prepare_finetuning_args, Args  # NON TOGLIERE: serve a torch.load per caricare il mio modello addestrato
from model_analysis import get_dataloader, get_dataset_dataloader


# import dei moduli dal tuo codice
from dataset import build_dataset
import torch.backends.cudnn as cudnn
from engine_for_finetuning import train_one_epoch, validation_one_epoch, final_test
from optim_factory import create_optimizer
from timm.models import create_model
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from utils import NativeScalerWithGradNormCount as NativeScaler

import warnings
warnings.filterwarnings('ignore')


def launch_finetuning_classification():
    args = prepare_finetuning_args()

    device = torch.device(args.device)
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    #print("=============== ARGS ===============")
    # Se vuoi stampare i parametri
    #for attr, value in vars(args).items():
    #    print(f"{attr} = {value}")
    #print("=====================================")

    # -------------------------------------------
    print("BUILDING DATASET...") 
    # -------------------------------------------
    # Carichiamo train, val, test
    dataset_train, nb_classes_train = build_dataset(is_train=True, test_mode=False, args=args)
    dataset_val, nb_classes_val = build_dataset(is_train=False, test_mode=False, args=args)
    dataset_test, nb_classes_test = build_dataset(is_train=False, test_mode=True, args=args)
    print(f"DATASET: train: {len(dataset_train)}, val: {len(dataset_val)}, test: {len(dataset_test)}")

    # Dato che NB_CLASSES deve essere coerente...
    print(f"[INFO] dataset_train classes: {nb_classes_train}, dataset_val classes: {nb_classes_val}, dataset_test classes: {nb_classes_test}")

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=int(args.batch_size * 1.5),  # un po' più grande in val
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    # -------------------------------------------
    # build model
    # -------------------------------------------
    print(f"Creating model: {args.model} (nb_classes={args.nb_classes})")
    model = create_model(
        args.model,
        #pretrained=True,
        num_classes=args.nb_classes,
        drop_rate=0.0,
        drop_path_rate=args.drop_path,
        #attn_drop_rate=0.0,
        drop_block_rate=None,
        **args.__dict__
    )
    # Carichiamo i pesi pretrained
    # if args.finetune and os.path.isfile(args.finetune):
    #     checkpoint = torch.load(args.finetune, map_location='cpu')
    #     print(f"[INFO] Loading pretrained weights from {args.finetune}")
    #     state_dict = checkpoint
    #     # Se c'è 'model' o 'module' come chiave
    #     for model_key in ['model', 'module']:
    #         if model_key in checkpoint:
    #             state_dict = checkpoint[model_key]
    #             print(f"[INFO] Found key '{model_key}' in checkpoint")
    #             break
    #     msg = model.load_state_dict(state_dict, strict=False)
    #     print(f"[INFO] load_state_dict: {msg}")
    # else:
    #     print("[WARN] No pretrained weights loaded (args.finetune non trovato)")

    model.to(device)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of params:", n_parameters)

    # costruiamo l’optimizer
    optimizer = create_optimizer(args, model)
    loss_scaler = NativeScaler()  # per amp

    # scheduler lr e wd
    total_batch_size = args.batch_size
    num_training_steps_per_epoch = len(dataset_train) // total_batch_size
    args.lr = args.lr * total_batch_size / 256
    args.min_lr = args.min_lr * total_batch_size / 256
    args.warmup_lr = args.warmup_lr * total_batch_size / 256
    print(f"[INFO] LR scaled: {args.lr}, warmup_lr: {args.warmup_lr}, min_lr: {args.min_lr}")

    lr_schedule_values = utils.cosine_scheduler(
        args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps)
    print(f"lr_schedule_values {lr_schedule_values}")
    if args.weight_decay_end is None:
        args.weight_decay_end = args.weight_decay
    wd_schedule_values = utils.cosine_scheduler(
        args.weight_decay, args.weight_decay_end,
        args.epochs, num_training_steps_per_epoch
    )
    print(f"wd_schedule_values {wd_schedule_values}")

    # Prepariamo la loss
    mixup_active = (args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None)
    print(f"mixup_active? {mixup_active}", flush=True)
    mixup_active = False
    if mixup_active:
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    mixup_fn = None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup,
            cutmix_alpha=args.cutmix,
            cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob,
            switch_prob=args.mixup_switch_prob,
            mode=args.mixup_mode,
            label_smoothing=args.smoothing,
            num_classes=args.nb_classes
        )
        print("[INFO] Mixup is activated")

    # logging
    if args.log_dir and not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    max_accuracy = 0.0
    start_time = time.time()

    print("START TRAINING!", flush=True)
    for epoch in range(args.start_epoch, args.epochs):
        # TRAIN
        train_stats = train_one_epoch(
            model=model,
            criterion=criterion,
            data_loader=data_loader_train,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            loss_scaler=loss_scaler,
            max_norm=args.clip_grad,
            model_ema=None,
            mixup_fn=mixup_fn,
            log_writer=None,  # se vuoi un TensorboardLogger, passalo qui
            start_steps=epoch * num_training_steps_per_epoch,
            lr_schedule_values=lr_schedule_values,
            wd_schedule_values=wd_schedule_values,
            num_training_steps_per_epoch=num_training_steps_per_epoch,
            update_freq=1
        )

        # SALVATAGGIO CHECKPOINT
        # if args.output_dir and ((epoch + 1) % args.save_ckpt_freq == 0 or (epoch + 1) == args.epochs):
        #     ckpt_path = os.path.join(args.output_dir, f"checkpoint-{epoch}.pth")
        #     torch.save({
        #         "model": model.state_dict(),
        #         "optimizer": optimizer.state_dict(),
        #         "epoch": epoch,
        #         "scaler": loss_scaler.state_dict(),
        #         "args": vars(args)  # se vuoi salvare i param
        #     }, ckpt_path)
        #     print(f"[INFO] checkpoint saved at {ckpt_path}")

        # VAL
        val_stats = {}
        if (epoch + 1) % args.VAL_FREQ == 0:
            val_stats = validation_one_epoch(data_loader_val, model, device)
            print(f"[EPOCH {epoch + 1}] val acc1: {val_stats['acc1']:.2f}%")
            if val_stats["acc1"] > max_accuracy:
                max_accuracy = val_stats["acc1"]
                print(f"[INFO] New best acc1: {max_accuracy:.2f}%")
                # se vuoi salvare un "best" checkpoint
                best_ckpt_path = os.path.join(args.output_dir, "checkpoint-best.pth")
                torch.save({
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "scaler": loss_scaler.state_dict(),
                    "args": vars(args)
                }, best_ckpt_path)
                print(f"[INFO] Best checkpoint saved at {best_ckpt_path}")

        # logging su file
        log_stats = {
            'epoch': epoch,
            **{f'train_{k}': v for k, v in train_stats.items()},
            **{f'val_{k}': v for k, v in val_stats.items()},
        }
        if args.output_dir:
            with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    print(f"Training time: {str(datetime.timedelta(seconds=int(total_time)))}")

    # TEST finale   # COMMENTO PERCHé FA ERRORE SE BATCH SIZE è SOLTANTO 1 (?)
    #test_stats = final_test(data_loader_test, model, device, file=os.path.join(args.output_dir, "test_preds.txt"))
    #print(f"[TEST] final test Acc@1 {test_stats['acc1']:.2f}%, Acc@5 {test_stats['acc5']:.2f}%")

    return



if __name__ == '__main__':
    launch_finetuning_classification()