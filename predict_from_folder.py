#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
from typing import Optional
import shutil
import subprocess

import pandas as pd
import torch
import torch.distributed as dist

import models  # necessario per create_model (timm registry)
from timm.models import create_model

import utils
from ffmpeg_utils import ensure_ffmpeg_in_path as _ensure_ffmpeg_in_path
from utils import setup_for_distributed
from arguments import prepare_finetuning_args
from dataset.data_manager import BuildDataset, DataManager
from dataset.build_dataset import create_final_df_csv, calc_tile_offsets
from medicane_utils.load_files import get_intervals_in_tracks_df
from engine_for_finetuning import validation_one_epoch_collect
from model_analysis import create_df_predictions, video_pred_2_img_pred


def _ensure_trailing_slash(path_str: str) -> str:
    return path_str if path_str.endswith("/") else path_str + "/"


def _load_tracks(manos_file: Optional[str]) -> tuple[pd.DataFrame, bool]:
    if manos_file:
        df = pd.read_csv(manos_file, parse_dates=["time", "start_time", "end_time"])
        return df, True
    cols = ["time", "x_pix", "y_pix", "source", "id_cyc_unico", "start_time", "end_time"]
    df = pd.DataFrame(columns=cols)
    df["time"] = pd.to_datetime(df["time"])
    df["start_time"] = pd.to_datetime(df["start_time"])
    df["end_time"] = pd.to_datetime(df["end_time"])
    return df, False


def _build_dataset_from_images(
    input_dir_images: str,
    output_dir_images: str,
    manos_file: Optional[str],
    tracks_df: pd.DataFrame,
    save_tiles: bool,
    create_csv: bool,
) -> tuple[BuildDataset, Optional[pd.DataFrame]]:
    if save_tiles:
        output_dir_images = _ensure_trailing_slash(output_dir_images)
        os.makedirs(output_dir_images, exist_ok=True)
    else:
        output_dir_images = None

    data = BuildDataset(type="SUPERVISED")
    data.create_master_df(manos_file=manos_file or "", input_dir_images=input_dir_images, tracks_df=tracks_df)

    start_dt = data.master_df["datetime"].min()
    end_dt = data.master_df["datetime"].max()
    print(f"[{input_dir_images}] Intervallo immagini: {start_dt} -> {end_dt}")
    if data.master_df.empty or pd.isna(start_dt) or pd.isna(end_dt):
        raise RuntimeError(
            f"Nessuna immagine trovata in '{input_dir_images}'. "
            "Controlla path e formato dei file (atteso: airmass_rgb_YYYYMMDD_HHMM.png)."
        )

    # Debug: statistiche sulle immagini coperte dagli intervalli Manos
    if not tracks_df.empty and "time" in tracks_df.columns:
        try:
            intervals = get_intervals_in_tracks_df(tracks_df)
            if not intervals.empty:
                # Limita agli intervalli che intersecano l'intervallo delle immagini
                intervals = intervals[
                    (intervals["max"] >= start_dt) & (intervals["min"] <= end_dt)
                ].reset_index(drop=True)
                print("Intervalli Manos (clippati all'intervallo immagini):")
                total_tiles = len(data.master_df)
                total_images = data.master_df["path"].nunique()
                covered_tiles = 0
                covered_images = 0
                for i, row in intervals.iterrows():
                    start_i = max(row["min"], start_dt)
                    end_i = min(row["max"], end_dt)
                    mask = (data.master_df["datetime"] >= start_i) & (data.master_df["datetime"] <= end_i)
                    count_tiles = int(mask.sum())
                    count_images = int(data.master_df.loc[mask, "path"].nunique())
                    covered_tiles += count_tiles
                    covered_images += count_images
                    print(f"  {i+1}) {start_i} -> {end_i} | immagini: {count_images} | tile: {count_tiles}")
                print(
                    f"Totale immagini folder: {total_images} | "
                    f"coperte da Manos (clippato): {covered_images}"
                )
                print(
                    f"Totale tile (righe master_df): {total_tiles} | "
                    f"coperte da Manos (clippato): {covered_tiles}"
                )
        except Exception as exc:
            print(f"Warning: impossibile calcolare stats Manos: {exc}")

    data.make_df_video(output_dir=output_dir_images, is_to_balance=False)
    print(f"Totale video tile creati: {len(data.df_video)}")
    if data.df_video is None or data.df_video.empty:
        raise RuntimeError(
            "df_video è vuoto: non posso creare il CSV. "
            "Verifica che l'intervallo contiguo abbia almeno 16 frame per tile."
        )

    if create_csv:
        if output_dir_images is None:
            raise RuntimeError("output_dir_images mancante per creare il CSV.")
        df_csv = create_final_df_csv(data.df_video, output_dir_images)
    else:
        df_csv = None
    return data, df_csv


def _concat_or_single(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    if not dfs:
        return pd.DataFrame([])
    if len(dfs) == 1:
        return dfs[0].copy()
    return pd.concat(dfs, ignore_index=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inferenza da folder immagini + video Mediterraneo.")
    parser.add_argument("--input_dir", required=True, help="Cartella immagini (png con date nel nome).")
    parser.add_argument("--output_dir", required=True, help="Cartella per salvare le tile video.")
    parser.add_argument("--model_path", required=True, help="Checkpoint del modello.")
    parser.add_argument(
        "--split_by_subfolder",
        action="store_true",
        help="Se true, processa ogni subfolder come sequenza separata (output unico).",
    )
    parser.add_argument(
        "--manos_file",
        default="medicane_data_input/medicanes_new_windows.csv",
        help="CSV Manos con etichette (opzionale). Se assente, solo predizioni.",
    )
    parser.add_argument(
        "--video_name",
        default="mediterraneo_predizioni",
        help="Nome base del video MP4 (senza estensione).",
    )
    parser.add_argument(
        "--ffmpeg_path",
        default=None,
        help="Path da aggiungere al PATH per trovare ffmpeg (opzionale).",
    )
    parser.add_argument(
        "--on",
        default=None,
        help="Nome macchina per caricare i preset di arguments.py (es. leonardo).",
    )
    parser.add_argument(
        "--make_video",
        action="store_true",
        help="Se presente, genera anche il video Mediterraneo.",
    )
    parser.add_argument(
        "--only_video",
        action="store_true",
        help="Se presente, salta il salvataggio dei frame e crea solo il video dai PNG esistenti.",
    )
    return parser.parse_args()


def _setup_distributed():
    if not torch.cuda.is_available():
        setup_for_distributed(True)
        return 0, 0, 1, False

    rank, local_rank, world_size, local_size, num_workers = utils.get_resources()
    if world_size > 1:
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        distributed = True
    else:
        distributed = False
    torch.cuda.set_device(local_rank)
    setup_for_distributed(rank == 0, silence_non_master=True)
    if rank == 0:
        env_keys = [
            "RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE",
            "OMPI_COMM_WORLD_RANK", "OMPI_COMM_WORLD_LOCAL_RANK", "OMPI_COMM_WORLD_SIZE",
            "SLURM_PROCID", "SLURM_LOCALID", "SLURM_NPROCS", "SLURM_NTASKS_PER_NODE",
        ]
        env_dump = {k: os.environ.get(k) for k in env_keys if os.environ.get(k) is not None}
        print(f"[env] distributed vars: {env_dump}")
    return rank, local_rank, world_size, distributed


def _prepare_dataset_csv(
    args_cli: argparse.Namespace,
    tracks_df: pd.DataFrame,
    save_tiles: bool,
    create_csv: bool,
):
    input_root = Path(args_cli.input_dir)
    output_root = Path(args_cli.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    builders: list[BuildDataset] = []
    csv_parts: list[pd.DataFrame] = []

    if args_cli.split_by_subfolder:
        subfolders = sorted([p for p in input_root.iterdir() if p.is_dir()])
        if not subfolders:
            data, df_csv = _build_dataset_from_images(
                str(input_root),
                str(output_root),
                args_cli.manos_file,
                tracks_df,
                save_tiles=save_tiles,
                create_csv=create_csv,
            )
            builders.append(data)
            csv_parts.append(df_csv)
        else:
            for sub in subfolders:
                out_dir = output_root / sub.name
                data, df_csv = _build_dataset_from_images(
                    str(sub),
                    str(out_dir),
                    args_cli.manos_file,
                    tracks_df,
                    save_tiles=save_tiles,
                    create_csv=create_csv,
                )
                builders.append(data)
                csv_parts.append(df_csv)
    else:
        data, df_csv = _build_dataset_from_images(
            str(input_root),
            str(output_root),
            args_cli.manos_file,
            tracks_df,
            save_tiles=save_tiles,
            create_csv=create_csv,
        )
        builders.append(data)
        csv_parts.append(df_csv)

    df_dataset_csv = _concat_or_single([c for c in csv_parts if c is not None])
    if create_csv:
        if df_dataset_csv.empty:
            raise RuntimeError("CSV dataset vuoto: nessun video generato.")

        dataset_csv = "output/general_inference_set.csv"
        os.makedirs(Path(dataset_csv).parent, exist_ok=True)
        df_dataset_csv.to_csv(dataset_csv, index=False)
        print(f"CSV dataset salvato in {dataset_csv} ({len(df_dataset_csv)} righe)")
    else:
        dataset_csv = None

    return builders, df_dataset_csv, dataset_csv


def _build_model(args_cli: argparse.Namespace, device: torch.device):
    args = prepare_finetuning_args(machine=args_cli.on)
    args.init_ckpt = args_cli.model_path
    args.load_for_test_mode = True
    args.val_path = "output/general_inference_set.csv"
    args.device = "cuda"

    model = create_model(
        args.model,
        num_classes=args.nb_classes,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        **args.__dict__,
    )
    model.to(device)
    model.eval()
    return args, model


def _fast_video_only(args_cli: argparse.Namespace) -> None:
    _ensure_ffmpeg_in_path(args_cli.ffmpeg_path)
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg non trovato nel PATH. "
            "Passa --ffmpeg_path /percorso/ffmpeg oppure installa ffmpeg."
        )

    folder = Path(f"./anim_frames_{args_cli.video_name}")
    frames_txt = folder / "frames.txt"
    if not folder.exists():
        raise RuntimeError(
            f"Cartella frame non trovata: {folder}. "
            "Rimuovi --only_video o genera prima i frame."
        )
    if not frames_txt.exists():
        raise RuntimeError(
            f"frames.txt non trovato in {folder}. "
            "Rimuovi --only_video o genera prima i frame."
        )

    nomefile = f"{args_cli.video_name}.mp4"
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(frames_txt),
        "-framerate",
        "10",
        "-vsync",
        "vfr",
        "-c:v",
        "libx264",
        "-crf",
        "18",
        "-preset",
        "medium",
        "-pix_fmt",
        "yuv420p",
        nomefile,
    ]
    print("\n>>> Creazione del video MP4 con ffmpeg (only_video)...")
    subprocess.run(cmd, check=True)
    print(f"\nVideo salvato: {nomefile}\n")


def _run_inference(args, model, device, world_size, rank, args_cli):
    val_m = DataManager(
        is_train=False,
        args=args,
        type_t="supervised",
        world_size=world_size,
        rank=rank,
        specify_data_path=args.val_path,
    )
    val_m.create_classif_dataloader(args)

    val_stats, all_paths, all_preds, all_labels = validation_one_epoch_collect(
        val_m.data_loader, model, device
    )
    df_predictions = create_df_predictions(all_paths, all_preds, all_labels)

    if rank == 0:
        preds_csv = "output/inference_predictions.csv"
        os.makedirs(Path(preds_csv).parent, exist_ok=True)
        df_predictions.to_csv(preds_csv, index=False)
        return df_predictions, val_stats
    return None, val_stats


def _make_video(args_cli, builders, df_predictions):
    _ensure_ffmpeg_in_path(args_cli.ffmpeg_path)
    from view_test_tiles import expand_group, make_animation_parallel_ffmpeg

    df_video_all = _concat_or_single([b.df_video for b in builders])
    master_df_all = _concat_or_single([b.master_df for b in builders])

    df_video_w_predictions = df_video_all.merge(df_predictions, on="path")
    df_mapping = video_pred_2_img_pred(df_video_w_predictions)
    df_data_merg = (
        df_mapping.merge(
            master_df_all,
            on=["path", "tile_offset_x", "tile_offset_y"],
            how="left",
        )
        .drop(columns="label")
        .rename(columns={"tmp_label": "label"})
    )

    offsets = calc_tile_offsets(stride_x=213, stride_y=196)
    df_offsets = pd.DataFrame(offsets, columns=["tile_offset_x", "tile_offset_y"])
    expanded_df = (
        df_data_merg.groupby("path", group_keys=False)
        .apply(lambda g: expand_group(g, df_offsets))
        .reset_index(drop=True)
    )
    expanded_df.predictions = expanded_df.predictions.astype("Int8")
    expanded_df.label = expanded_df.label.astype("Int8")

    make_animation_parallel_ffmpeg(
        expanded_df,
        nomefile=args_cli.video_name,
        only_video=args_cli.only_video,
    )


def main() -> None:
    args_cli = parse_args()

    if args_cli.only_video and not args_cli.make_video:
        raise RuntimeError("--only_video richiede --make_video.")

    if args_cli.only_video:
        _fast_video_only(args_cli)
        return

    rank, local_rank, world_size, distributed = _setup_distributed()
    device = torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else torch.device("cpu")

    preds_csv = "output/inference_predictions.csv"
    df_predictions = None
    val_stats = None

    preds_exist = Path(preds_csv).exists()
    need_inference = not preds_exist
    need_video = args_cli.make_video

    tracks_df, has_labels = _load_tracks(args_cli.manos_file)
    builders, _, dataset_csv = _prepare_dataset_csv(
        args_cli,
        tracks_df,
        save_tiles=need_inference,
        create_csv=need_inference,
    )

    if rank == 0 and preds_exist:
        print(
            f"Predizioni già presenti in {preds_csv}: "
            "salto il caricamento del modello e non calcolo nuove predizioni."
        )
        print("Uso solo master_df/df_video in memoria: nessuna tile verrà salvata su disco.")
        df_predictions = pd.read_csv(preds_csv)
    else:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA non disponibile: serve GPU per questo script.")

        args, model = _build_model(args_cli, device)
        if distributed:
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False
            )

        df_predictions, val_stats = _run_inference(
            args=args,
            model=model,
            device=device,
            world_size=world_size,
            rank=rank,
            args_cli=args_cli,
        )

        if rank == 0:
            if has_labels:
                print(f"Predizioni salvate in {preds_csv} - bal_acc={val_stats.get('bal_acc')}")
            else:
                print(f"Predizioni salvate in {preds_csv} (etichette assenti)")

    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    if need_video and rank == 0:
        if df_predictions is None:
            df_predictions = pd.read_csv(preds_csv)
        _make_video(args_cli, builders, df_predictions)


if __name__ == "__main__":
    main()
