import numpy as np
import os
import datetime
import time
import random
import json
import torch
import torch.nn.functional as F
import torch.distributed as dist

from utils import multiple_pretrain_samples_collate, setup_for_distributed
from functools import partial

import utils
from engine_for_pretraining import train_one_epoch, test
from arguments import prepare_finetuning_args, Args  # NON TOGLIERE: serve a torch.load per caricare il mio modello addestrato


from dataset import build_dataset
import torch.backends.cudnn as cudnn
from engine_for_finetuning import train_one_epoch, validation_one_epoch, final_test
from optim_factory import create_optimizer
import models # NON TOGLIERE: serve a torch.load per caricare il mio modello addestrato
from timm.models import create_model
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from utils import NativeScalerWithGradNormCount as NativeScaler

from dataset.data_manager import DataManager

import warnings
warnings.filterwarnings('ignore')

def all_seeds():
    os.environ['PYTHONHASHSEED'] = str(0)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"
    import random
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)  # anche da richiamare tra una chiamata e l'altra del training
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
    #torch.use_deterministic_algorithms(True)

    
def launch_finetuning_classification(terminal_args):
    
    args = prepare_finetuning_args(machine=terminal_args.on)


    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True


    ############################# Distributed Training #############################

    rank, local_rank, world_size, local_size, num_workers = utils.get_resources()
    #print(f"rank, local_rank, world_size, local_size, num_workers: {rank, local_rank, world_size, local_size, num_workers}")

    if world_size > 1:
        dist.init_process_group("nccl", rank=rank, world_size=world_size)    
        args.distributed = True
    else:
        args.distributed = False   
    if args.device == 'cuda':    
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
    

    # logging
    if args.log_dir and not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    setup_for_distributed(rank == 0)

    #print("=============== ARGS ===============")
    # Se vuoi stampare i parametri
    #for attr, value in vars(args).items():
    #    print(f"{attr} = {value}")
    #print("=====================================")

    # -------------------------------------------
    #print("MANAGER DATASET...") 
    # -------------------------------------------
    train_m = DataManager(is_train=True, args=args, type_t='supervised', world_size=world_size, rank=rank)
    test_m = DataManager(is_train=False, args=args, type_t='supervised', world_size=world_size, rank=rank)
    val_m = DataManager(is_train=False, args=args, type_t='supervised', world_size=world_size, rank=rank, specify_data_path=args.val_path)

    train_m.create_classif_dataloader(args)
    test_m.create_classif_dataloader(args)
    val_m.create_classif_dataloader(args)

    # compute class weights for loss balancing
    class_weights = None
    if getattr(args, 'use_class_weight', False):
        if hasattr(train_m.dataset, 'df') and 'label' in train_m.dataset.df:
            labels = train_m.dataset.df['label'].to_numpy()
        else:
            labels = [lbl for _, lbl, _ in train_m.dataset]
            labels = np.array(labels)
        num_classes = labels.max() + 1
        class_counts = np.bincount(labels, minlength=num_classes)
        class_weights = len(labels) / (num_classes * class_counts)
        class_weights = torch.tensor(class_weights, dtype=torch.float32, device=device)

        # custom factor
        class_weights[1] *= 2

    #print(class_weights, class_counts, len(labels))


    # -------------------------------------------
    # build model
    # -------------------------------------------
    print(f"Creating model: {args.model} (nb_classes={args.nb_classes})")
    print(f"is pretrained? {args.pretrained}")
    pretrained_model = create_model(
        args.model,
        #pretrained=True,
        num_classes=args.nb_classes,
        drop_rate=0.0,
        drop_path_rate=args.drop_path,
        #attn_drop_rate=0.0,
        drop_block_rate=None,
        **args.__dict__
    )

    pretrained_model.to(device)
    model_without_ddp = pretrained_model
    n_parameters = sum(p.numel() for p in pretrained_model.parameters() if p.requires_grad)
    #print("number of params:", n_parameters)

    if args.distributed:
        pretrained_model = torch.nn.parallel.DistributedDataParallel(
            pretrained_model, device_ids=[args.gpu], output_device=args.gpu, 
            find_unused_parameters=False)
        model_without_ddp = pretrained_model.module


    # costruiamo l’optimizer
    optimizer = create_optimizer(args, model_without_ddp)
    loss_scaler = NativeScaler()  # per amp

    # scheduler lr e wd
    total_batch_size = args.batch_size * world_size
    num_training_steps_per_epoch = train_m.dataset_len// total_batch_size
    args.lr = args.lr * total_batch_size / 256
    args.min_lr = args.min_lr * total_batch_size / 256
    args.warmup_lr = args.warmup_lr * total_batch_size / 256
    print(f"[INFO] LR scaled: {args.lr}, warmup_lr: {args.warmup_lr}, min_lr: {args.min_lr}")

    lr_schedule_values = utils.cosine_scheduler(
        args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs, start_warmup_value=args.warmup_lr, warmup_steps=args.warmup_steps)
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
        print("Usiamo label smoothing")
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    #else:
    # if getattr(args, 'use_class_weight', False) and class_weights is not None:
    #     criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    #     print(f"Loss is weighted. Weights: {class_weights}")
    # else:
    #     criterion = torch.nn.CrossEntropyLoss()

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

    if args.auto_resume:
        utils.auto_load_model(
                args=args,
                model=pretrained_model,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                loss_scaler=loss_scaler)
    torch.cuda.empty_cache()

    

    max_bal_acc = 0.0
    start_time = time.time()

    print("START TRAINING!", flush=True)
    for epoch in range(args.start_epoch, args.epochs):
        # TRAIN
        train_stats = train_one_epoch(
            model=pretrained_model,
            criterion=criterion,
            data_loader=train_m.data_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            loss_scaler=loss_scaler,
            max_norm=args.clip_grad,
            model_ema=None,
            mixup_fn=mixup_fn,
            log_writer=None, 
            start_steps=epoch * num_training_steps_per_epoch,
            lr_schedule_values=lr_schedule_values,
            wd_schedule_values=wd_schedule_values,
            num_training_steps_per_epoch=num_training_steps_per_epoch,
            update_freq=1
        )

        # VAL
        val_stats = {}
        if (epoch + 1) % args.testing_epochs == 0:
            val_stats = validation_one_epoch(test_m.data_loader, pretrained_model, device, criterion)
            print(f"[EPOCH {epoch + 1}] val bal_acc: {val_stats['bal_acc']:.2f}%  - best bal_acc: {max_bal_acc:.2f}%")

            val2_stats = validation_one_epoch(val_m.data_loader, pretrained_model, device, criterion)

            if val_stats["bal_acc"] > max_bal_acc and epoch > args.start_epoch_for_saving_best_ckpt:
                max_bal_acc = val_stats["bal_acc"]
                print(f"[INFO] New best balanced accuracy: {max_bal_acc:.2f}%")

                # se vuoi salvare un "best" checkpoint
                ########################### SALVATAGGIO CHECKPOINT
                best_ckpt_path = os.path.join(args.output_dir, "checkpoint-best.pth")
                torch.save({
                    "model": model_without_ddp.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "scaler": loss_scaler.state_dict(),
                    "args": vars(args)
                }, best_ckpt_path)
                print(f"[INFO] Best checkpoint saved at {best_ckpt_path} , epoch {epoch}")


        # logging su file
        log_stats = {'epoch': epoch,
            **{f'train_{k}': v for k, v in train_stats.items()},
            **{f'val_{k}': v for k, v in val_stats.items()},
            **{f'val2_{k}': v for k, v in val2_stats.items()},
        }
        if args.output_dir and rank == 0:
            with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    print(f"Training time: {str(datetime.timedelta(seconds=int(total_time)))}")

    # TEST finale   # COMMENTO PERCHé FA ERRORE SE BATCH SIZE è SOLTANTO 1 (?)
    #test_stats = final_test(data_loader_test, model, device, file=os.path.join(args.output_dir, "test_preds.txt"))
    #print(f"[TEST] final test Acc@1 {test_stats['acc1']:.2f}%, Acc@5 {test_stats['acc5']:.2f}%")

    return



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        'Lancia il training supervisionato di classificazione',
        add_help=False)
    parser.add_argument('--on',
        type=str,
        default='leonardo',
        #metavar='NAME',
        help='[ewc, leonardo]'
    ),
    args =  parser.parse_args()


    launch_finetuning_classification(args)
