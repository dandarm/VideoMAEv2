import numpy as np
import os
import datetime
import time
import random
import json
import torch
import torch.nn.functional as F
import torch.distributed as dist

import utils
from utils import setup_for_distributed
from engine_for_pretraining import train_one_epoch, test
from arguments import prepare_finetuning_args

import torch.backends.cudnn as cudnn
from engine_for_finetuning import (
    train_one_epoch,
    validation_one_epoch,
    validation_one_epoch_collect,
    validation_one_epoch_collect_logits,
)
from optim_factory import create_optimizer
import models # NON TOGLIERE: serve a torch.load per caricare il mio modello addestrato
from timm.models import create_model
from timm.data.mixup import Mixup
#from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from utils import NativeScalerWithGradNormCount as NativeScaler

from dataset.data_manager import DataManager
from model_analysis import (
    create_df_predictions,
    merge_all_rank_merged,
    cleanup_npz_shards,
)
from glob import glob

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

    
def launch_inference_classification(terminal_args):
    
    args = prepare_finetuning_args(machine=terminal_args.on)

    if terminal_args.csvfile is not None:
        args.val_path = terminal_args.csvfile

    model_for_inference = terminal_args.inference_model
    args.init_ckpt = model_for_inference

    args.load_for_test_mode = True


    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True


    ############################# Distributed Training #############################
    # region distributed variabiles
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
    

    # logging dirs
    if args.log_dir and not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    # ensure output_dir exists to allow metrics logging
    if args.output_dir and not os.path.exists(args.output_dir):
        try:
            os.makedirs(args.output_dir, exist_ok=True)
        except Exception as e:
            if rank == 0:
                print(f"Warning: could not create output_dir {args.output_dir}: {e}")
    setup_for_distributed(rank == 0)

    # endregion#

    val_m = DataManager(is_train=False, args=args, type_t='supervised', world_size=world_size, rank=rank, specify_data_path=args.val_path)
    val_m.create_classif_dataloader(args)

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

    if args.distributed:
        pretrained_model = torch.nn.parallel.DistributedDataParallel(
            pretrained_model, device_ids=[args.gpu], output_device=args.gpu, 
            find_unused_parameters=False)
    
    pretrained_model.eval()

    torch.cuda.empty_cache()

    
    start_time = time.time()
    # VAL flow: either collect simple preds into CSV, or collect logits into NPZ shards
    if getattr(terminal_args, 'get_logits', False):
        # Choose directory to save logits shards/merged
        base_out = args.output_dir if args.output_dir else '.'
        logits_subdir = getattr(terminal_args, 'logits_dir', 'val_logits')
        save_dir = os.path.join(base_out, logits_subdir)
        os.makedirs(save_dir, exist_ok=True)

        prefix = getattr(terminal_args, 'logits_prefix', 'val')
        val_stats, all_paths, all_preds, all_labels = validation_one_epoch_collect_logits(
            val_m.data_loader,
            pretrained_model,
            device,
            save_dir=save_dir,
            prefix=prefix,
        )
        print(f"val bal_acc: {val_stats['bal_acc']:.2f}%  (logits saved under {save_dir})")

        # Sync all ranks before attempting cross-rank merge/cleanup
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
        # Perform final merge of per-rank merged files and cleanup shards on rank 0
        if rank == 0:
            try:
                # 1) Crea merged globale senza cancellare i per-rank (servono per pulire gli shard)
                global_path = merge_all_rank_merged(save_dir=save_dir, prefix=prefix, delete_inputs=False)
                print(f"Created global merged logits: {global_path}")
            except Exception as e:
                print(f"Warning: could not create global merged logits: {e}")
            # 2) Rimuovi sempre gli shard per-batch ora che i per-rank merged esistono
            try:
                # Delete all shard parts unconditionally for the chosen prefix
                removed = cleanup_npz_shards(save_dir=save_dir, prefix=prefix, only_if_merged=False, dry_run=False)
                print(f"Removed {len(removed)} shard files.")
            except Exception as e:
                print(f"Warning: shard cleanup failed: {e}")
            # 3) Rimuovi sempre anche i merged per-rank, lasciando solo l'all_merged
            try:
                rank_merged_files = sorted(glob(os.path.join(save_dir, f"{prefix}_rank*_merged.npz")))
                removed_cnt = 0
                for f in rank_merged_files:
                    try:
                        os.remove(f)
                        removed_cnt += 1
                    except Exception as e:
                        print(f"Warning: could not remove {f}: {e}")
                if removed_cnt:
                    print(f"Removed {removed_cnt} per-rank merged files.")
            except Exception as e:
                print(f"Warning: rank-merged cleanup failed: {e}")
                
        # Save predictions CSV (per-rank gather done inside collect_logits)
        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
        df = create_df_predictions(all_paths, all_preds, all_labels)
        preds_name = getattr(terminal_args, 'preds_csv', 'inference_predictions.csv')
        out_csv = preds_name if not args.output_dir else os.path.join(args.output_dir, preds_name)
        if rank == 0:
            try:
                df.to_csv(out_csv, index=False)
                print(f"Saved predictions to {out_csv}")
            except Exception as e:
                print(f"Warning: could not save predictions CSV: {e}")
    else:
        # Standard path: collect predictions and save a CSV
        val_stats, all_paths, all_preds, all_labels = validation_one_epoch_collect(
            val_m.data_loader, pretrained_model, device
        )
        print(f"val bal_acc: {val_stats['bal_acc']:.2f}% ")

        # Save predictions CSV (per-rank if distributed)
        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
        # build dataframe
        df = create_df_predictions(all_paths, all_preds, all_labels)
        # choose output filename
        preds_name = getattr(terminal_args, 'preds_csv', 'inference_predictions.csv')
        out_csv = preds_name if not args.output_dir else os.path.join(args.output_dir, preds_name)
        # save only on rank 0 (after distributed gather inside validation_one_epoch_collect)
        if rank == 0:
            try:
                df.to_csv(out_csv, index=False)
                print(f"Saved predictions to {out_csv}")
            except Exception as e:
                print(f"Warning: could not save predictions CSV: {e}")

    # logging su file (always)
    log_stats = {**{f'val2_{k}': v for k, v in val_stats.items()}}
    # round floats to 4 decimals for compact logs
    log_stats = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in log_stats.items()}
    if args.output_dir and rank == 0:
        metrics_path = os.path.join(args.output_dir, "inference_metrics.txt")
        try:
            with open(metrics_path, "a") as f:
                f.write(json.dumps(log_stats) + "\n")
        except Exception as e:
            print(f"Warning: could not write metrics to {metrics_path}: {e}")

    total_time = time.time() - start_time
    print(f"Inference time: {str(datetime.timedelta(seconds=int(total_time)))}")
    return



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser('Lancia il training supervisionato di classificazione',
        add_help=False)
    parser.add_argument('--on',
        type=str,
        default='leonardo',
        help='[ewc, leonardo]'
    )
    parser.add_argument('--csvfile', type=str, default='val_manos_w_2400.csv')
    
    parser.add_argument('--inference_model', type=str, default='output/checkpoint-best.pth')
    parser.add_argument('--preds_csv', type=str, default='inference_predictions.csv', help='Output CSV filename for predictions')

    parser.add_argument('--get_logits', action='store_true', help='Collect logits and save .npz shards')
    parser.add_argument('--logits_dir', type=str, default='val_logits', help='Subfolder under output_dir to store logits npz files')
    parser.add_argument('--logits_prefix', type=str, default='val', help='Prefix for saved logits files')
    

    args =  parser.parse_args()


    launch_inference_classification(args)
