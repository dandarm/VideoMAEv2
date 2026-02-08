#!/usr/bin/env python3
import argparse
import ast
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from timm.models import create_model

import models  # needed for timm registry
import utils
from utils import setup_for_distributed
from arguments import prepare_finetuning_args, prepare_tracking_args
from dataset.data_manager import DataManager
from dataset.build_dataset import calc_tile_offsets
from engine_for_finetuning import validation_one_epoch_collect
from model_analysis import create_df_predictions
from predict_from_folder import _load_tracks, _prepare_dataset_csv
from track_from_folder import (
    TrackFromFolderDataset,
    _parse_tile_folder_name,
    _build_gt_map_from_tracks,
)
from inference_tracking import run_tracking_inference, load_checkpoint, set_seeds
from models.tracking_model import create_tracking_model
from view_test_tiles import expand_group, make_animation_parallel_ffmpeg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unisce inferenza di classificazione e tracking da cartella immagini."
    )
    parser.add_argument("--input_dir", required=True, help="Cartella immagini (png con date nel nome).")
    parser.add_argument("--output_dir", required=True, help="Cartella per salvare le tile video.")
    parser.add_argument("--classification_model_path", required=True, help="Checkpoint modello classificazione.")
    parser.add_argument("--tracking_model_path", required=True, help="Checkpoint modello tracking.")
    parser.add_argument(
        "--split_by_subfolder",
        action="store_true",
        help="Se true, processa ogni subfolder come sequenza separata (output unico).",
    )
    parser.add_argument(
        "--manos_file",
        default="medicane_data_input/medicanes_new_windows.csv",
        help="CSV Manos per etichette/GT (opzionale).",
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
        help="Nome macchina per preset arguments.py (es. leonardo).",
    )
    parser.add_argument(
        "--make_video",
        action="store_true",
        help="Se presente, genera anche il video Mediterraneo.",
    )
    parser.add_argument(
        "--only_video",
        action="store_true",
        help="Se presente, genera solo il video finale dai PNG già esistenti.",
    )
    return parser.parse_args()


def _setup_distributed():
    if not torch.cuda.is_available():
        setup_for_distributed(True)
        return 0, 0, 1, False

    rank, local_rank, world_size, _, _ = utils.get_resources()
    if world_size > 1:
        if not dist.is_initialized():
            dist.init_process_group("nccl", rank=rank, world_size=world_size)
        distributed = True
    else:
        distributed = False
    torch.cuda.set_device(local_rank)
    setup_for_distributed(rank == 0, silence_non_master=True)
    if rank == 0:
        env_keys = [
            "RANK",
            "LOCAL_RANK",
            "WORLD_SIZE",
            "LOCAL_WORLD_SIZE",
            "OMPI_COMM_WORLD_RANK",
            "OMPI_COMM_WORLD_LOCAL_RANK",
            "OMPI_COMM_WORLD_SIZE",
            "SLURM_PROCID",
            "SLURM_LOCALID",
            "SLURM_NPROCS",
            "SLURM_NTASKS_PER_NODE",
        ]
        env_dump = {k: os.environ.get(k) for k in env_keys if os.environ.get(k) is not None}
        print(f"[env] distributed vars: {env_dump}")
    return rank, local_rank, world_size, distributed


def _ensure_ffmpeg_in_path(ffmpeg_path: Optional[str]) -> None:
    if ffmpeg_path:
        os.environ["PATH"] = ffmpeg_path + os.pathsep + os.environ.get("PATH", "")
        return
    local_ffmpeg = Path(__file__).resolve().parent / "ffmpeg-7.0.2-amd64-static"
    if local_ffmpeg.exists():
        os.environ["PATH"] = str(local_ffmpeg) + os.pathsep + os.environ.get("PATH", "")


def _find_tile_folders(root: Path) -> List[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Output dir not found: {root}")
    folders = []
    for path in root.rglob("*"):
        if path.is_dir() and _parse_tile_folder_name(path.name) is not None:
            folders.append(path)
    return sorted(folders)


def _run_classification_inference(
    args_cli: argparse.Namespace,
    dataset_csv: str,
    device: torch.device,
    world_size: int,
    rank: int,
    distributed: bool,
    preds_csv_path: Path,
):
    args = prepare_finetuning_args(machine=args_cli.on)
    args.init_ckpt = args_cli.classification_model_path
    args.load_for_test_mode = True
    args.val_path = dataset_csv
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

    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[device.index], output_device=device.index, find_unused_parameters=False
        )

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
        preds_csv_path.parent.mkdir(parents=True, exist_ok=True)
        df_predictions.to_csv(preds_csv_path, index=False)
    return df_predictions, val_stats


def _select_positive_tiles(df_predictions: pd.DataFrame) -> List[str]:
    if df_predictions.empty:
        return []
    pred_vals = pd.to_numeric(df_predictions.get("predictions"), errors="coerce")
    positive_mask = pred_vals >= 0.5
    return (
        df_predictions.loc[positive_mask, "path"]
        .astype(str)
        .dropna()
        .unique()
        .tolist()
    )


def _to_list(val):
    if val is None:
        return []
    if isinstance(val, (list, tuple)):
        return list(val)
    if isinstance(val, (np.ndarray, pd.Series)):
        return list(val)
    if isinstance(val, float) and np.isnan(val):
        return []
    if isinstance(val, str):
        stripped = val.strip()
        if not stripped:
            return []
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                parsed = ast.literal_eval(stripped)
                if isinstance(parsed, (list, tuple)):
                    return list(parsed)
            except Exception:
                return [stripped]
        return [stripped]
    return [val]


def _append_tracking_points(df: pd.DataFrame) -> pd.DataFrame:
    def _append_row(row):
        x_vals = _to_list(row.get("x_pix"))
        y_vals = _to_list(row.get("y_pix"))
        src_vals = _to_list(row.get("source"))

        tgt_x = row.get("track_target_x")
        tgt_y = row.get("track_target_y")
        if pd.notna(tgt_x) and pd.notna(tgt_y):
            x_vals.insert(0, float(tgt_x))
            y_vals.insert(0, float(tgt_y))
            src_vals.insert(0, "GT")

        pred_x = row.get("track_pred_x")
        pred_y = row.get("track_pred_y")
        if pd.notna(pred_x) and pd.notna(pred_y):
            x_vals.insert(0, float(pred_x))
            y_vals.insert(0, float(pred_y))
            src_vals.insert(0, "PRED")

        row["x_pix"] = x_vals
        row["y_pix"] = y_vals
        row["source"] = src_vals
        return row

    return df.apply(_append_row, axis=1)


def _build_timeframe_csv(
    tile_folders: List[Path],
    tracking_df: pd.DataFrame,
    manos_available: bool,
    output_csv: Path,
) -> None:
    time_list: List[pd.Timestamp] = []
    for folder in tile_folders:
        parsed = _parse_tile_folder_name(folder.name)
        if parsed is None:
            continue
        dt, _, _ = parsed
        time_list.append(dt)
    if not time_list:
        raise RuntimeError("Nessuna tile valida per costruire il CSV finale.")

    all_times = sorted(set(time_list))

    if not tracking_df.empty:
        tracking_df = tracking_df.copy()
        tracking_df["tile_time"] = tracking_df["path"].apply(
            lambda name: _parse_tile_folder_name(str(name))[0]
            if _parse_tile_folder_name(str(name)) is not None
            else pd.NaT
        )

    rows = []
    for dt in all_times:
        chosen = None
        if not tracking_df.empty:
            subset = tracking_df[tracking_df["tile_time"] == dt]
            if manos_available:
                gt_subset = subset[subset["target_x"].notna() & subset["target_y"].notna()]
                if not gt_subset.empty:
                    chosen = gt_subset.sort_values("path").iloc[0]
            if chosen is None and not subset.empty:
                chosen = subset.sort_values("path").iloc[0]

        if chosen is None:
            rows.append(
                {
                    "datetime": dt,
                    "has_cyclone": 0,
                    "pred_lat": np.nan,
                    "pred_lon": np.nan,
                }
            )
        else:
            rows.append(
                {
                    "datetime": dt,
                    "has_cyclone": 1,
                    "pred_lat": chosen.get("pred_lat"),
                    "pred_lon": chosen.get("pred_lon"),
                }
            )

    out_df = pd.DataFrame(rows)
    out_df["datetime"] = pd.to_datetime(out_df["datetime"])
    out_df = out_df.sort_values("datetime").reset_index(drop=True)
    out_df["has_cyclone"] = out_df["has_cyclone"].astype("Int8")
    out_df["pred_lat"] = pd.to_numeric(out_df["pred_lat"], errors="coerce").round(2)
    out_df["pred_lon"] = pd.to_numeric(out_df["pred_lon"], errors="coerce").round(2)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_csv, index=False)


def _build_tracking_frames_df(
    builders,
    tracking_df: pd.DataFrame,
    df_predictions: Optional[pd.DataFrame],
) -> pd.DataFrame:
    df_video_all = pd.concat([b.df_video for b in builders], ignore_index=True)
    master_df_all = pd.concat([b.master_df for b in builders], ignore_index=True)

    for df in (df_video_all, master_df_all):
        if "path" in df.columns:
            df["path"] = df["path"].astype(str)
        for col in ["tile_offset_x", "tile_offset_y"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").round().astype("Int64")
    if "path" in master_df_all.columns:
        master_df_all["path_key"] = master_df_all["path"].apply(lambda p: os.path.basename(str(p)))

    preds_df = df_predictions.copy() if df_predictions is not None else pd.DataFrame([])
    if not preds_df.empty and "path" in preds_df.columns:
        preds_df["path"] = preds_df["path"].astype(str)
        df_video_all = df_video_all.merge(
            preds_df[["path", "predictions"]],
            on="path",
            how="left",
        )

    track_df = tracking_df.copy() if tracking_df is not None else pd.DataFrame([])
    if "path" not in track_df.columns:
        track_df = pd.DataFrame(
            columns=[
                "path",
                "pred_x_global",
                "pred_y_global",
                "target_x_global",
                "target_y_global",
            ]
        )
    if not track_df.empty:
        track_df["path"] = track_df["path"].apply(lambda p: os.path.basename(str(p)))
        track_df = track_df.drop(columns=["tile_offset_x", "tile_offset_y"], errors="ignore")

    df_joined = df_video_all.merge(track_df, on="path", how="left")

    records: List[Dict[str, object]] = []
    for _, row in df_joined.iterrows():
        orig_paths = _to_list(row.get("orig_paths"))
        for orig_path in orig_paths:
            records.append(
                {
                    "path": orig_path,
                    "tile_offset_x": row.get("tile_offset_x"),
                    "tile_offset_y": row.get("tile_offset_y"),
                    "predictions": row.get("predictions"),
                    "track_pred_x": row.get("pred_x_global"),
                    "track_pred_y": row.get("pred_y_global"),
                    "track_target_x": row.get("target_x_global"),
                    "track_target_y": row.get("target_y_global"),
                }
            )

    df_track_map = pd.DataFrame(records)
    if not df_track_map.empty:
        df_track_map["path"] = df_track_map["path"].astype(str)
        df_track_map["path_key"] = df_track_map["path"].apply(lambda p: os.path.basename(str(p)))
        for col in ["tile_offset_x", "tile_offset_y"]:
            df_track_map[col] = pd.to_numeric(df_track_map[col], errors="coerce").round().astype("Int64")

    for col in ["tile_offset_x", "tile_offset_y"]:
        if col in master_df_all.columns:
            master_df_all[col] = pd.to_numeric(master_df_all[col], errors="coerce").round().astype("Int64")
    df_frames = master_df_all.merge(
        df_track_map,
        on=["path_key", "tile_offset_x", "tile_offset_y"],
        how="left",
        suffixes=("", "_track"),
    )

    for col in ["predictions", "track_pred_x", "track_pred_y", "track_target_x", "track_target_y"]:
        if col in df_frames.columns:
            df_frames[col] = pd.to_numeric(df_frames[col], errors="coerce")

    for col in ["x_pix", "y_pix", "source"]:
        if col not in df_frames.columns:
            df_frames[col] = np.nan

    df_frames = _append_tracking_points(df_frames)
    df_frames["keep_tile_boxes"] = True
    df_frames["keep_all_tracks"] = True

    offsets = calc_tile_offsets(stride_x=213, stride_y=196)
    df_offsets = pd.DataFrame(offsets, columns=["tile_offset_x", "tile_offset_y"])
    expanded_df = (
        df_frames.groupby("path", group_keys=False)
        .apply(lambda g: expand_group(g, df_offsets))
        .reset_index(drop=True)
    )
    return expanded_df


def _make_tracking_video(
    args_cli: argparse.Namespace,
    builders,
    tracking_df: pd.DataFrame,
    df_predictions: Optional[pd.DataFrame],
) -> None:
    _ensure_ffmpeg_in_path(args_cli.ffmpeg_path)
    if not args_cli.only_video:
        frames_dir = Path(f"./anim_frames_{args_cli.video_name}")
        if frames_dir.exists():
            shutil.rmtree(frames_dir)
    expanded_df = _build_tracking_frames_df(builders, tracking_df, df_predictions)
    make_animation_parallel_ffmpeg(
        expanded_df,
        nomefile=args_cli.video_name,
        only_video=args_cli.only_video,
    )


def main() -> None:
    args_cli = parse_args()

    if args_cli.only_video and not args_cli.make_video:
        raise RuntimeError("--only_video richiede --make_video.")

    rank, local_rank, world_size, distributed = _setup_distributed()
    device = torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else torch.device("cpu")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA non disponibile: serve GPU per questo script.")

    class_preds_csv = Path(args_cli.output_dir) / "_tmp_inference_predictions.csv"
    track_tiles_csv = Path(args_cli.output_dir) / "_tmp_tracking_inference_predictions_tiles.csv"
    final_time_csv = Path(args_cli.output_dir) / "tracking_inference_predictions.csv"

    # 1) Classification: build tiles + inference (skip if predictions already exist)
    tracks_df, _ = _load_tracks(args_cli.manos_file)
    preds_exist = class_preds_csv.exists()
    builders, _, dataset_csv = _prepare_dataset_csv(
        args_cli,
        tracks_df,
        save_tiles=not preds_exist,
        create_csv=not preds_exist,
    )

    df_predictions = None
    val_stats = None
    if preds_exist:
        if rank == 0:
            print(
                f"[INFO] Predizioni classificazione già presenti in {class_preds_csv}: "
                "salto inferenza."
            )
        df_predictions = pd.read_csv(class_preds_csv)
    else:
        df_predictions, val_stats = _run_classification_inference(
            args_cli=args_cli,
            dataset_csv=dataset_csv,
            device=device,
            world_size=world_size,
            rank=rank,
            distributed=distributed,
            preds_csv_path=class_preds_csv,
        )

    if rank == 0:
        if val_stats is not None and "bal_acc" in val_stats:
            print(f"[INFO] Classificazione completata - bal_acc={val_stats.get('bal_acc')}")
        if not preds_exist:
            print(f"[INFO] Predizioni classificazione salvate in {class_preds_csv}")

    # 2) Tracking: solo tile positive
    tile_folders = _find_tile_folders(Path(args_cli.output_dir))
    if not tile_folders:
        raise RuntimeError(f"Nessuna tile trovata in {args_cli.output_dir}")

    positive_names = _select_positive_tiles(df_predictions)
    name_to_folder = {p.name: p for p in tile_folders}
    positive_folders = [name_to_folder[n] for n in positive_names if n in name_to_folder]

    if rank == 0:
        print(f"[INFO] Tile totali: {len(tile_folders)} | positive: {len(positive_folders)}")

    track_exist = track_tiles_csv.exists()
    if track_exist:
        if rank == 0:
            print(
                f"[INFO] Predizioni tracking già presenti in {track_tiles_csv}: "
                "salto inferenza."
            )
    elif positive_folders:
        args_tracking = prepare_tracking_args(machine=args_cli.on)
        args_tracking.output_dir = args_cli.output_dir
        args_tracking.pretrained = False
        args_tracking.init_ckpt = ""
        args_tracking.load_for_test_mode = True

        set_seeds(args_tracking.seed)

        tile_infos: List[Tuple[str, pd.Timestamp, float, float]] = []
        for folder in positive_folders:
            parsed = _parse_tile_folder_name(folder.name)
            if parsed is None:
                print(f"[WARN][combined] Nome cartella non valido (saltato GT): {folder.name}")
                continue
            dt_floor, off_x, off_y = parsed
            tile_infos.append((str(folder), dt_floor, off_x, off_y))

        gt_map = _build_gt_map_from_tracks(args_cli.manos_file, tile_infos)
        dataset = TrackFromFolderDataset(
            folders=positive_folders,
            target_map=gt_map,
            clip_len=args_tracking.num_frames,
        )

        sampler = None
        if distributed:
            sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)

        data_loader = DataLoader(
            dataset,
            batch_size=args_tracking.batch_size,
            num_workers=args_tracking.num_workers,
            pin_memory=args_tracking.pin_mem,
            sampler=sampler,
        )

        model = create_tracking_model(args_tracking.model, **args_tracking.__dict__)
        model.to(device)
        if rank == 0:
            print(f"[INFO] Loading tracking checkpoint: {args_cli.tracking_model_path}")
        load_checkpoint(model, args_cli.tracking_model_path, device)

        if distributed:
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[device.index], output_device=device.index
            )

        _ = run_tracking_inference(
            model=model,
            data_loader=data_loader,
            device=device,
            output_dir=args_cli.output_dir,
            preds_csv=track_tiles_csv.name,
        )
    else:
        if rank == 0:
            print("[WARN][combined] Nessuna tile positiva: salto tracking.")

    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    # 3) CSV finale per timeframe (datetime, has_cyclone, pred_lat, pred_lon)
    if rank == 0:
        if track_tiles_csv.exists():
            track_df = pd.read_csv(track_tiles_csv)
        else:
            track_df = pd.DataFrame([])
        manos_available = bool(args_cli.manos_file and Path(args_cli.manos_file).exists())
        _build_timeframe_csv(tile_folders, track_df, manos_available, final_time_csv)
        print(f"[INFO] CSV finale salvato in {final_time_csv}")

    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    # 4) Video finale (tracking view con logica notebook)
    if args_cli.make_video and rank == 0:
        if track_tiles_csv.exists():
            track_df = pd.read_csv(track_tiles_csv)
        else:
            track_df = pd.DataFrame([])
        if df_predictions is None or df_predictions.empty:
            df_predictions = pd.read_csv(class_preds_csv)
        _make_tracking_video(args_cli, builders, track_df, df_predictions)


if __name__ == "__main__":
    main()
