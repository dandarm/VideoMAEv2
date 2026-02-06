#!/usr/bin/env python3
"""Inference for cyclone tracking from a folder of video tiles."""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torchvision import transforms

import utils
from arguments import prepare_tracking_args
from ffmpeg_utils import ensure_ffmpeg_in_path as _ensure_ffmpeg_in_path
from models.tracking_model import create_tracking_model
import engine_for_tracking as tracking_engine
from utils import setup_for_distributed


IMG_EXTS = {".png"} #, ".jpg", ".jpeg", ".tif", ".tiff"}
FRAMERATE = 10
TILE_NAME_RE = re.compile(
    r"^(?P<date>\d{2}-\d{2}-\d{4})_(?P<time>\d{4})_(?P<x>-?\d+(?:\.\d+)?)_(?P<y>-?\d+(?:\.\d+)?)$"
)


def _collect_tile_folders(root: Path) -> List[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Input dir not found: {root}")
    if root.is_dir():
        subdirs = [p for p in root.iterdir() if p.is_dir()]
        if subdirs:
            return sorted(subdirs)
        image_files = [p for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
        if image_files:
            return [root]
    return []


def _first_val(val):
    """Extract first numeric value from list-like or string representation."""
    import ast
    import math
    if isinstance(val, (list, tuple)):
        return float(val[0]) if len(val) > 0 else math.nan
    if isinstance(val, str):
        s = val.strip()
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, (list, tuple)) and len(parsed) > 0:
                return float(parsed[0])
            return float(parsed)
        except Exception:
            try:
                return float(s)
            except Exception:
                return math.nan
    try:
        return float(val)
    except Exception:
        return float("nan")


def _parse_tile_folder_name(folder_name: str) -> Optional[Tuple[pd.Timestamp, float, float]]:
    match = TILE_NAME_RE.match(folder_name)
    if not match:
        return None
    date_str = match.group("date")
    time_str = match.group("time")
    try:
        dt = datetime.strptime(f"{date_str}_{time_str}", "%d-%m-%Y_%H%M")
    except ValueError:
        return None
    try:
        offset_x = float(match.group("x"))
        offset_y = float(match.group("y"))
    except ValueError:
        return None
    return pd.Timestamp(dt), offset_x, offset_y


def _build_gt_map_from_tracks(
    manos_file: Optional[str],
    tile_infos: List[Tuple[str, pd.Timestamp, float, float]],
    tile_width: int = 224,
    tile_height: int = 224,
) -> Dict[str, Tuple[float, float]]:
    if not manos_file:
        return {}
    manos_path = Path(manos_file)
    if not manos_path.exists():
        print(f"[WARN][track_from_folder] GT file not found: {manos_file}. Output senza GT.")
        return {}

    tracks_df = pd.read_csv(manos_path)
    required_cols = {"time", "x_pix", "y_pix"}
    if not required_cols.issubset(tracks_df.columns):
        print("[WARN][track_from_folder] GT CSV senza colonne time/x_pix/y_pix. Output senza GT.")
        return {}

    tracks_df = tracks_df.copy()
    tracks_df["time"] = pd.to_datetime(tracks_df["time"], errors="coerce", dayfirst=True)
    tracks_df["x_pix"] = tracks_df["x_pix"].apply(_first_val)
    tracks_df["y_pix"] = tracks_df["y_pix"].apply(_first_val)
    tracks_df = tracks_df[np.isfinite(tracks_df["x_pix"]) & np.isfinite(tracks_df["y_pix"])].copy()
    tracks_df = tracks_df[tracks_df["time"].notna()].copy()
    if tracks_df.empty:
        return {}

    tracks_df["time_floor"] = tracks_df["time"].dt.round("h")
    grouped = {t: df for t, df in tracks_df.groupby("time_floor")}

    gt_map: Dict[str, Tuple[float, float]] = {}
    for folder_path, dt_floor, offset_x, offset_y in tile_infos:
        df_t = grouped.get(dt_floor)
        if df_t is None or df_t.empty:
            continue
        cond_x = (df_t["x_pix"] >= offset_x) & (df_t["x_pix"] < offset_x + tile_width)
        cond_y = (df_t["y_pix"] >= offset_y) & (df_t["y_pix"] < offset_y + tile_height)
        hits = df_t[cond_x & cond_y]
        if hits.empty:
            continue
        row = hits.iloc[0]
        gt_map[folder_path] = (float(row["x_pix"]) - offset_x, float(row["y_pix"]) - offset_y)
    return gt_map


class TrackFromFolderDataset(Dataset):
    def __init__(
        self,
        folders: Sequence[Path],
        target_map: Dict[str, Tuple[float, float]],
        clip_len: int = 16,
        transform: Optional[transforms.Compose] = None,
    ) -> None:
        self.folders = [Path(p) for p in folders]
        self.clip_len = clip_len
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = transform
        self.target_map = target_map
        self._placeholder_clip_cache: Optional[torch.Tensor] = None

    def __len__(self) -> int:
        return len(self.folders)

    def _placeholder_clip(self) -> torch.Tensor:
        if self._placeholder_clip_cache is None:
            dummy_img = Image.new("RGB", (224, 224))
            frame_tensor = self.transform(dummy_img) if self.transform else transforms.ToTensor()(dummy_img)
            frames = [frame_tensor.clone() for _ in range(self.clip_len)]
            video = torch.stack(frames, dim=0).permute(1, 0, 2, 3)
            self._placeholder_clip_cache = video
        return self._placeholder_clip_cache.clone()

    def _load_clip(self, folder_path: Path) -> Tuple[bool, Optional[torch.Tensor], str]:
        try:
            frame_files = sorted([p for p in folder_path.iterdir() if p.suffix.lower() in IMG_EXTS])
        except Exception as exc:
            return False, None, f"os.listdir failed: {exc}"

        if not frame_files:
            return False, None, "directory contains no frames"

        frames = []
        for file in frame_files[: self.clip_len]:
            try:
                img = Image.open(file).convert("RGB")
                img = self.transform(img)
                frames.append(img)
            except Exception as exc:
                print(f"[WARN][TrackFromFolderDataset] Problema nel caricamento di {file}: {exc}")

        if not frames:
            return False, None, "no frames could be loaded from disk"

        if len(frames) != self.clip_len:
            frames = frames + [frames[-1]] * (self.clip_len - len(frames))

        video = torch.stack(frames, dim=0).permute(1, 0, 2, 3)
        return True, video, ""

    def _lookup_target(self, folder_path: Path) -> Optional[Tuple[float, float]]:
        return self.target_map.get(str(folder_path))

    def __getitem__(self, idx: int):
        folder_path = self.folders[idx]
        ok, video, err = self._load_clip(folder_path)
        if not ok or video is None:
            print(f"[WARN][TrackFromFolderDataset] Skipping {folder_path}: {err}")
            video = self._placeholder_clip()
        target = self._lookup_target(folder_path)
        if target is None or not np.isfinite(target).all():
            coords = torch.zeros(2, dtype=torch.float32)
            has_target = False
        else:
            coords = torch.tensor(target, dtype=torch.float32)
            has_target = True
        return video, coords, str(folder_path), has_target


def _compute_sample_record(
    path: str,
    pred_xy: torch.Tensor,
    target_xy: Optional[torch.Tensor],
) -> Dict[str, Union[float, str, None]]:
    pred_np = pred_xy.detach().cpu().numpy().astype(float)
    record: Dict[str, Union[float, str, None]] = {
        "path": path,
        "pred_x": float(pred_np[0]),
        "pred_y": float(pred_np[1]),
        "target_x": None,
        "target_y": None,
        "err_px": None,
        "err_km": None,
    }

    if target_xy is not None:
        target_np = target_xy.detach().cpu().numpy().astype(float)
        err_vec = pred_np - target_np
        record["target_x"] = float(target_np[0])
        record["target_y"] = float(target_np[1])
        record["err_px"] = float(np.linalg.norm(err_vec))

    try:
        offset_x, offset_y = tracking_engine._parse_tile_offsets(path)
        pred_abs = pred_np + np.array([offset_x, offset_y], dtype=float)
        record["tile_offset_x"] = float(offset_x)
        record["tile_offset_y"] = float(offset_y)
        record["pred_x_global"] = float(pred_abs[0])
        record["pred_y_global"] = float(pred_abs[1])

        lat_pred, lon_pred = tracking_engine._pixels_to_latlon(
            np.array([pred_abs[0]]), np.array([pred_abs[1]])
        )
        record["pred_lat"] = float(lat_pred[0])
        record["pred_lon"] = float(lon_pred[0])

        if target_xy is not None:
            target_np = target_xy.detach().cpu().numpy().astype(float)
            target_abs = target_np + np.array([offset_x, offset_y], dtype=float)
            record["target_x_global"] = float(target_abs[0])
            record["target_y_global"] = float(target_abs[1])
            lat_true, lon_true = tracking_engine._pixels_to_latlon(
                np.array([target_abs[0]]), np.array([target_abs[1]])
            )
            record["target_lat"] = float(lat_true[0])
            record["target_lon"] = float(lon_true[0])
            record["err_km"] = float(
                tracking_engine._haversine_km(lat_pred, lon_pred, lat_true, lon_true)[0]
            )
    except Exception as exc:
        print(f"[WARN][track_from_folder] Unable to recover geo coords for {path}: {exc}")

    return record


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = True


def load_checkpoint(model: torch.nn.Module, checkpoint_path: str, device: torch.device) -> None:
    if not checkpoint_path:
        raise ValueError("checkpoint path must be provided for inference")
    map_location = device if device.type == "cpu" else torch.device("cpu")
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    state_dict = checkpoint.get("model") or checkpoint.get("state_dict")
    if state_dict is None:
        raise KeyError(
            f"Checkpoint at {checkpoint_path} does not contain 'model' or 'state_dict' keys; "
            f"available keys: {list(checkpoint.keys()) if isinstance(checkpoint, dict) else 'N/A'}"
        )
    msg = model.load_state_dict(state_dict, strict=False)
    print(f"[INFO] Loaded checkpoint from {checkpoint_path}: {msg}")


def _format_log_value(key: str, value):
    if not isinstance(value, float):
        return value
    if "lr" in key.lower():
        return float(f"{value:.8g}")
    return round(value, 4)


def _render_tracking_video(
    folder_path: Path,
    pred_xy: Tuple[float, float],
    target_xy: Optional[Tuple[float, float]],
    output_dir: Path,
    radius: int = 3,
) -> None:
    frame_files = sorted([p for p in folder_path.iterdir() if p.suffix.lower() in IMG_EXTS])
    if not frame_files:
        print(f"[WARN][track_from_folder] No frames found in {folder_path}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    frame_dir = output_dir / folder_path.name
    frame_dir.mkdir(parents=True, exist_ok=True)

    pred_x, pred_y = pred_xy
    tgt_x, tgt_y = target_xy if target_xy is not None else (None, None)

    for idx, frame_path in enumerate(frame_files):
        img = Image.open(frame_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        if tgt_x is not None and tgt_y is not None:
            draw.ellipse((tgt_x - radius, tgt_y - radius, tgt_x + radius, tgt_y + radius), fill=(0, 255, 0))
        draw.ellipse((pred_x - radius, pred_y - radius, pred_x + radius, pred_y + radius), fill=(255, 0, 0))
        img.save(frame_dir / f"frame_{idx:05d}.png")

    if shutil.which("ffmpeg") is None:
        print("[WARN][track_from_folder] ffmpeg non trovato nel PATH: salto creazione video")
        return

    frames_txt = frame_dir / "frames.txt"
    dur = 1.0 / FRAMERATE
    ordered_paths = sorted(frame_dir.glob("frame_*.png"))
    if not ordered_paths:
        print(f"[WARN][track_from_folder] Nessun frame annotato trovato in {frame_dir}")
        return
    with open(frames_txt, "w") as handle:
        for p in ordered_paths:
            handle.write(f"file '{os.path.abspath(p)}'\n")
            handle.write(f"duration {dur}\n")
        handle.write(f"file '{os.path.abspath(ordered_paths[-1])}'\n")

    video_path = output_dir / f"{folder_path.name}.mp4"
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
        str(FRAMERATE),
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
        str(video_path),
    ]
    subprocess.run(cmd, check=False)


def inference_epoch(
    model: torch.nn.Module,
    data_loader: Iterable,
    device: torch.device,
) -> Tuple[Dict[str, float], List[Dict[str, Union[float, str, None]]]]:
    criterion = torch.nn.MSELoss()
    metric_logger = utils.MetricLogger(delimiter="  ")
    results: List[Dict[str, Union[float, str, None]]] = []

    for samples, target, paths, has_target in metric_logger.log_every(data_loader, 20, header="Infer:"):
        samples = samples.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        has_target = has_target.to(device, non_blocking=True)

        output = model(samples)

        valid_mask = has_target.bool()
        loss_value = None
        px_err = None
        geo_err = None
        if valid_mask.any():
            valid_out = output[valid_mask]
            valid_tgt = target[valid_mask]
            loss = criterion(valid_out, valid_tgt)
            loss_value = loss.item()
            px_err = float(torch.norm(valid_out - valid_tgt, dim=-1).mean().item())
            valid_paths = [paths[i] for i in range(len(paths)) if bool(valid_mask[i].item())]
            geo_err = tracking_engine.batch_geo_distance_km(valid_out, valid_tgt, valid_paths)

        metric_logger.update(loss=loss_value, geo_km=geo_err, px_err=px_err)

        for idx, sample_path in enumerate(paths):
            tgt = target[idx] if bool(valid_mask[idx].item()) else None
            results.append(_compute_sample_record(sample_path, output[idx], tgt))

    metric_logger.synchronize_between_processes()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    return stats, results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inferenza tracking da folder di video tile")
    parser.add_argument("--input_dir", required=True, help="Cartella con subfolder dei video tile.")
    parser.add_argument("--model_path", required=True, help="Checkpoint del modello di tracking.")
    parser.add_argument("--output_dir", default=None, help="Cartella output (default: args.output_dir).")
    parser.add_argument(
        "--manos_file",
        default="medicane_data_input/medicanes_new_windows.csv",
        help="CSV Manos con GT (time, x_pix, y_pix). Se non presente, output solo predizioni.",
    )
    parser.add_argument("--on", type=str, default=None, help="Nome macchina per preset arguments.py (es. leonardo).")
    parser.add_argument("--make_video", action="store_true", help="Se presente, genera mp4 per ogni tile.")
    parser.add_argument("--ffmpeg_path", default=None, help="Path per ffmpeg (opzionale).")
    return parser.parse_args()


def launch_track_from_folder(cli_args: argparse.Namespace) -> None:
    args = prepare_tracking_args(machine=cli_args.on)
    if cli_args.output_dir:
        args.output_dir = cli_args.output_dir
    # Avoid loading init_ckpt from default args; we want CLI model_path to be authoritative.
    args.pretrained = False
    args.init_ckpt = ""

    set_seeds(args.seed)

    rank, local_rank, world_size, _, _ = utils.get_resources()
    if world_size > 1:
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        args.distributed = True
    else:
        args.distributed = False

    if args.device == "cuda" and torch.cuda.is_available():
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

    setup_for_distributed(rank == 0)

    input_dir = Path(cli_args.input_dir)
    folders = _collect_tile_folders(input_dir)
    if not folders:
        raise RuntimeError(f"Nessuna tile folder trovata in {input_dir}")

    tile_infos: List[Tuple[str, pd.Timestamp, float, float]] = []
    for folder in folders:
        parsed = _parse_tile_folder_name(folder.name)
        if parsed is None:
            print(f"[WARN][track_from_folder] Nome cartella non valido (saltato GT): {folder.name}")
            continue
        dt_floor, off_x, off_y = parsed
        tile_infos.append((str(folder), dt_floor, off_x, off_y))

    gt_map = _build_gt_map_from_tracks(cli_args.manos_file, tile_infos)

    dataset = TrackFromFolderDataset(
        folders=folders,
        target_map=gt_map,
        clip_len=args.num_frames,
    )

    sampler = None
    if args.distributed:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)

    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        sampler=sampler,
    )

    model = create_tracking_model(args.model, **args.__dict__)
    model.to(device)
    if utils.is_main_process():
        print(f"[INFO] Loading tracking checkpoint from CLI: {cli_args.model_path}")
    load_checkpoint(model, cli_args.model_path, device)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], output_device=args.gpu
        )

    model.eval()
    torch.cuda.empty_cache()

    stats, local_results = inference_epoch(model, data_loader, device)

    gathered_results = local_results
    if dist.is_available() and dist.is_initialized():
        gathered: List[List[Dict[str, Union[float, str, None]]]] = [None] * dist.get_world_size()
        dist.all_gather_object(gathered, local_results)
        if utils.is_main_process():
            gathered_results = [item for sublist in gathered for item in sublist]
        else:
            gathered_results = []

    if utils.is_main_process():
        os.makedirs(args.output_dir, exist_ok=True)
        df = pd.DataFrame(gathered_results)
        out_csv = os.path.join(args.output_dir, "tracking_inference_predictions.csv")
        df.to_csv(out_csv, index=False)
        print(f"Saved tracking predictions to {out_csv}")

        log_stats = {f"test_{k}": v for k, v in stats.items()}
        log_stats = {k: _format_log_value(k, v) for k, v in log_stats.items()}
        metrics_path = os.path.join(args.output_dir, "inference_tracking_metrics.txt")
        try:
            with open(metrics_path, "a") as handle:
                handle.write(json.dumps(log_stats) + "\n")
        except Exception as exc:
            print(f"Warning: could not write metrics to {metrics_path}: {exc}")

        if cli_args.make_video:
            _ensure_ffmpeg_in_path(cli_args.ffmpeg_path)
            video_dir = Path(args.output_dir)
            for record in gathered_results:
                path = record.get("path")
                if not path:
                    continue
                pred_xy = (record.get("pred_x"), record.get("pred_y"))
                if pred_xy[0] is None or pred_xy[1] is None:
                    continue
                target_xy = None
                if record.get("target_x") is not None and record.get("target_y") is not None:
                    target_xy = (record.get("target_x"), record.get("target_y"))
                _render_tracking_video(Path(path), pred_xy, target_xy, video_dir)

    dist.barrier() if dist.is_available() and dist.is_initialized() else None


if __name__ == "__main__":
    cli_args = parse_args()
    launch_track_from_folder(cli_args)
