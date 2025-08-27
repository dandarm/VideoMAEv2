from pathlib import Path
import numpy as np
import pandas as pd
from arguments import prepare_finetuning_args
from dataset.build_dataset import solve_paths, calc_avg_cld_idx
from dataset.data_manager import DataManager, BuildDataset
from model_analysis import predict_label, get_path_pred_label, create_df_predictions, get_only_labels

import utils
from utils import setup_for_distributed
import models
from timm.models import create_model
import torch
import torch.distributed as dist


if __name__ == "__main__":
    #input_dir = "$FAST/Medicanes_Data/from_gcloud"
    #output_dir = "$FAST/airmass/" 
    #manos_file = 'medicanes_new_windows.csv'


    args = prepare_finetuning_args()
    #output_dir = solve_paths(output_dir)
    #input_dir = solve_paths(input_dir)
    rank, local_rank, world_size, local_size, num_workers = utils.get_resources()
    dist.init_process_group("nccl", rank=rank, world_size=world_size)  
    torch.cuda.set_device(local_rank)    
    args.gpu = local_rank
    args.world_size = world_size
    args.rank = rank    
    device = torch.device(f"cuda:{local_rank}")
    setup_for_distributed(rank == 0)
    
    
    ###### carico il dataset (applico filtro nuvolosità sui video di validation)
    # Carico il csv di validation, calcolo avg_cloud_idx per ogni video usando
    # la funzione `calc_avg_cld_idx` definita in dataset/build_dataset.py
    try:
        df_val = pd.read_csv(args.val_path)
    except Exception:
        # se non riesco a leggere il csv uso comunque il path originale
        df_val = None

    if args.cloudy and df_val is not None and 'path' in df_val.columns:
        print('Calcolo indice di nuvolosità per i video di validation...')
        # Applichiamo direttamente la funzione ai percorsi
        df_val['avg_cloud_idx'] = df_val['path'].apply(lambda p: calc_avg_cld_idx(p))
        # soglia: manteniamo video con avg_cloud_idx <= 0.2 (come in build_dataset.filter_out_clear_sky)
        thresh = 0.2
        kept = df_val[df_val['avg_cloud_idx'] <= thresh].copy()
        print(f"Video totali: {len(df_val)} - dopo filtro nuvolosità (<= {thresh}): {len(kept)}")
        # salvo temporaneamente il csv filtrato e lo passo a DataManager
        filtered_csv = './val_filtered_for_prediction.csv'
        kept.to_csv(Path(args.csv_folder) / filtered_csv, index=False)
        val_csv_to_use = filtered_csv
    else:
        val_csv_to_use = args.val_path

    val_m = DataManager(is_train=False, args=args, specify_data_path=val_csv_to_use)
    val_m.create_classif_dataloader(args)

    # voglio prendere le predizioni
    get_prediction = True
    args.test_mode = True
    # istanzia l'oggetto del modello 
    print(f"Creating model: {args.model} (nb_classes={args.nb_classes})")
    args.init_ckpt = './output/checkpoint-best-90_25lug25.pth'
    model_ckpt = create_model(
        args.model,
        num_classes=args.nb_classes,
        drop_rate=0.0,
        drop_path_rate=args.drop_path,
        #attn_drop_rate=0.0,
        drop_block_rate=None,
        **args.__dict__
    )
    model_ckpt.to(device)

    model_ckpt = torch.nn.parallel.DistributedDataParallel(
            model_ckpt, device_ids=[local_rank], output_device=local_rank, 
            find_unused_parameters=False)
    
    model_ckpt.eval()  


    # calcolo e prendo le predizioni
    all_paths, all_preds, all_labels = get_path_pred_label(model_ckpt, val_m.data_loader)
    all_labels_array = np.array(all_labels)
    all_preds_array = np.array(all_preds)
    df_predictions = create_df_predictions(all_paths, all_preds, all_labels)

    df_predictions.to_csv(f"df_predictions.csv", index=False)

    print("Predizioni calcolate e salvate in df_predictions.csv")