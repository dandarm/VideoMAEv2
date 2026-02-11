"""Inference entry point for cyclone centre tracking."""

from __future__ import annotations

import argparse
import datetime
import json
import os
import random
import time
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import warnings

import engine_for_tracking as tracking_engine
import utils
from arguments import prepare_tracking_args
from dataset.data_manager import DataManager
from models.tracking_model import create_tracking_model
from utils import setup_for_distributed

warnings.filterwarnings("ignore")
utils.suppress_transformers_pytree_warning()


def _format_log_value(key: str, value):
    """Format numeric stats preserving LR precision."""
    if not isinstance(value, float):
        return value
    if "lr" in key.lower():
        return float(f"{value:.8g}")
    return round(value, 4)


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = True


def _ensure_path_list(paths) -> List[str]:
    if isinstance(paths, (list, tuple)):
        return [str(p) for p in paths]
    return [str(paths)]


def _compute_sample_record(
    path: str,
    pred_xy: torch.Tensor,
    target_xy: Union[torch.Tensor, None],
    has_target: Optional[bool] = None,
) -> Dict[str, Union[float, int, str, None]]:
    if has_target is None:
        has_target = target_xy is not None
    else:
        has_target = bool(has_target)
    pred_np = pred_xy.detach().cpu().numpy().astype(float)
    target_np = None
    err_px = None
    if has_target and target_xy is not None:
        target_np = target_xy.detach().cpu().numpy().astype(float)
        err_vec = pred_np - target_np
        err_px = float(np.linalg.norm(err_vec))

    offset_x = offset_y = None
    pred_abs = target_abs = None
    pred_lat = pred_lon = target_lat = target_lon = None
    err_km = None

    try:
        offset_x, offset_y = tracking_engine._parse_tile_offsets(path)
        pred_abs = pred_np + np.array([offset_x, offset_y], dtype=float)
        lat_pred, lon_pred = tracking_engine._pixels_to_latlon(
            np.array([pred_abs[0]]), np.array([pred_abs[1]])
        )
        pred_lat = float(lat_pred[0])
        pred_lon = float(lon_pred[0])
        if target_np is not None:
            target_abs = target_np + np.array([offset_x, offset_y], dtype=float)
            lat_true, lon_true = tracking_engine._pixels_to_latlon(
                np.array([target_abs[0]]), np.array([target_abs[1]])
            )
            target_lat = float(lat_true[0])
            target_lon = float(lon_true[0])
            err_km = float(
                tracking_engine._haversine_km(lat_pred, lon_pred, lat_true, lon_true)[0]
            )
    except Exception as exc:  # pragma: no cover - defensive path for malformed names
        print(f"[WARN][tracking_inference] Unable to recover geo coords for {path}: {exc}")

    record = {
        "path": path,
        "has_target": int(has_target),
        "pred_x": float(pred_np[0]),
        "pred_y": float(pred_np[1]),
        "target_x": float(target_np[0]) if target_np is not None else None,
        "target_y": float(target_np[1]) if target_np is not None else None,
        "err_px": err_px,
        "err_km": err_km,
    }

    if offset_x is not None and pred_abs is not None:
        record.update(
            {
                "tile_offset_x": float(offset_x),
                "tile_offset_y": float(offset_y),
                "pred_x_global": float(pred_abs[0]),
                "pred_y_global": float(pred_abs[1]),
                "pred_lat": pred_lat,
                "pred_lon": pred_lon,
            }
        )
        if target_np is not None:
            record.update(
                {
                    "target_x_global": float(target_abs[0]),
                    "target_y_global": float(target_abs[1]),
                    "target_lat": target_lat,
                    "target_lon": target_lon,
                }
            )

    return record


def _extract_state_dict(checkpoint_obj) -> Dict[str, torch.Tensor]:
    if isinstance(checkpoint_obj, dict):
        for key in ("state_dict", "model", "module"):
            val = checkpoint_obj.get(key)
            if isinstance(val, dict):
                return val
        # Fallback: if dict itself already looks like a state_dict
        if checkpoint_obj and all(isinstance(v, torch.Tensor) for v in checkpoint_obj.values()):
            return checkpoint_obj
    raise KeyError(
        "Checkpoint senza state_dict/model/module valido per il caricamento."
    )


def _normalize_state_dict_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    cleaned: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        new_key = key.replace("backbone.", "").replace("module.", "").replace("_orig_mod.", "")
        cleaned[new_key] = value
    return cleaned


def _adapt_legacy_head_keys(
    state_dict: Dict[str, torch.Tensor],
    model_state: Dict[str, torch.Tensor],
) -> Tuple[Dict[str, torch.Tensor], List[str]]:
    adapted = dict(state_dict)
    mapped: List[str] = []

    # Legacy tracking checkpoints might store a linear head as head.weight/head.bias.
    # Current tracking head is RegressionHead: head.mlp.1.{weight,bias}.
    if "head.weight" in adapted and "head.mlp.1.weight" in model_state:
        src = adapted.get("head.weight")
        dst_shape = tuple(model_state["head.mlp.1.weight"].shape)
        if isinstance(src, torch.Tensor) and tuple(src.shape) == dst_shape:
            adapted["head.mlp.1.weight"] = src
            mapped.append("head.weight->head.mlp.1.weight")
    if "head.bias" in adapted and "head.mlp.1.bias" in model_state:
        src = adapted.get("head.bias")
        dst_shape = tuple(model_state["head.mlp.1.bias"].shape)
        if isinstance(src, torch.Tensor) and tuple(src.shape) == dst_shape:
            adapted["head.mlp.1.bias"] = src
            mapped.append("head.bias->head.mlp.1.bias")

    return adapted, mapped


@torch.no_grad()
def inference_epoch(model, data_loader, device) -> tuple[Dict[str, float], List[Dict[str, Union[float, str, None]]]]:
    criterion = torch.nn.MSELoss()
    metric_logger = utils.MetricLogger(delimiter="  ")
    results: List[Dict[str, Union[float, str, None]]] = []

    for batch in metric_logger.log_every(data_loader, 20, header="Infer:"):
        if len(batch) == 3:
            samples, target, paths = batch
            has_target = None
        elif len(batch) == 4:
            samples, target, paths, has_target = batch
        else:
            raise ValueError(f"Expected batch of len 3 or 4, got {len(batch)}")

        samples = samples.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        if has_target is not None:
            has_target = has_target.to(device, non_blocking=True)

        output = model(samples)

        if has_target is None:
            valid_mask = torch.ones(target.shape[0], dtype=torch.bool, device=target.device)
        else:
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
            geo_err = tracking_engine.batch_geo_distance_km(valid_out, valid_tgt, _ensure_path_list(valid_paths))

        metric_logger.update(loss=loss_value, geo_km=geo_err, px_err=px_err)

        path_list = _ensure_path_list(paths)
        for idx, sample_path in enumerate(path_list):
            sample_has_target = bool(valid_mask[idx].item())
            tgt = target[idx] if sample_has_target else None
            results.append(
                _compute_sample_record(
                    sample_path,
                    output[idx],
                    tgt,
                    has_target=sample_has_target,
                )
            )

    metric_logger.synchronize_between_processes()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    return stats, results


def load_checkpoint(model: torch.nn.Module, checkpoint_path: str, device: torch.device) -> None:
    if not checkpoint_path:
        raise ValueError("checkpoint path must be provided for inference")
    map_location = device if device.type == "cpu" else torch.device("cpu")
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    raw_state_dict = _extract_state_dict(checkpoint)
    cleaned_state_dict = _normalize_state_dict_keys(raw_state_dict)

    model_state = model.state_dict()
    cleaned_state_dict, mapped_head_keys = _adapt_legacy_head_keys(cleaned_state_dict, model_state)

    filtered_state: Dict[str, torch.Tensor] = {}
    unexpected_keys: List[str] = []
    skipped_shape: List[str] = []
    for key, value in cleaned_state_dict.items():
        if key not in model_state:
            unexpected_keys.append(key)
            continue
        if tuple(value.shape) != tuple(model_state[key].shape):
            skipped_shape.append(
                f"{key}: ckpt{tuple(value.shape)} != model{tuple(model_state[key].shape)}"
            )
            continue
        filtered_state[key] = value

    msg = model.load_state_dict(filtered_state, strict=False)

    missing = sorted(msg.missing_keys)
    loaded_count = len(filtered_state)
    total_model_keys = len(model_state)
    print(
        f"[INFO] Loaded checkpoint from {checkpoint_path} | "
        f"loaded={loaded_count}/{total_model_keys}, missing={len(missing)}, "
        f"unexpected={len(unexpected_keys)}, shape_mismatch={len(skipped_shape)}"
    )
    if mapped_head_keys:
        print(f"[INFO] Legacy head mapping applied: {mapped_head_keys}")

    # Explicit diagnostic: confirm regression head parameters are actually loaded.
    head_keys = sorted([k for k in model_state.keys() if k.startswith("head.")])
    loaded_head = [k for k in head_keys if k in filtered_state]
    if head_keys:
        print(
            f"[INFO] Regression head load status: loaded {len(loaded_head)}/{len(head_keys)} params"
        )
        if len(loaded_head) == 0:
            print(
                "[WARN] Nessun parametro della regression head caricato dal checkpoint: "
                "la testa puo' restare random."
            )

    if skipped_shape:
        print("[WARN] Parametri saltati per mismatch shape (prime 8):")
        for row in skipped_shape[:8]:
            print(f"  - {row}")
    if missing:
        print(f"[WARN] Missing keys after load (prime 8): {missing[:8]}")
    if unexpected_keys:
        print(f"[WARN] Unexpected checkpoint keys (prime 8): {unexpected_keys[:8]}")


def _validate_tracking_dataset_paths(dataset) -> None:
    resolved_paths = getattr(dataset, "_resolved_paths", None)
    if not isinstance(resolved_paths, list):
        return
    total = len(resolved_paths)
    if total == 0:
        raise RuntimeError("Tracking dataset vuoto: nessun sample disponibile.")

    existing = sum(1 for p in resolved_paths if isinstance(p, str) and os.path.isdir(p))
    missing = total - existing
    print(f"[INFO] Tracking dataset paths: existing={existing}/{total}, missing={missing}")
    if existing == 0:
        raise RuntimeError(
            "Nessuna cartella clip del tracking esiste sul filesystem. "
            "Controlla il CSV input e i path assoluti/relativi."
        )
    if missing > 0:
        print(
            "[WARN] Alcune cartelle clip del tracking non esistono. "
            "Il dataset usera' fallback/skip per quei sample."
        )


def run_tracking_inference(
    model: torch.nn.Module,
    data_loader,
    device: torch.device,
    output_dir: str,
    preds_csv: str,
) -> tuple[Dict[str, float], List[Dict[str, Union[float, str, None]]]]:
    model.eval()
    torch.cuda.empty_cache()

    start = time.time()
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
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        df = pd.DataFrame(gathered_results)
        if "path" in df.columns:
            df["path"] = df["path"].apply(lambda p: os.path.basename(str(p)) if p else p)

        def _summary_stats(series: pd.Series) -> Optional[Dict[str, float]]:
            s = pd.to_numeric(series, errors="coerce")
            s = s[np.isfinite(s)]
            if s.empty:
                return None
            return {
                "min": float(s.min()),
                "max": float(s.max()),
                "mean": float(s.mean()),
                "std": float(s.std()),
                "p05": float(s.quantile(0.05)),
                "p50": float(s.quantile(0.50)),
                "p95": float(s.quantile(0.95)),
            }

        pixel_cols = [
            "pred_x",
            "pred_y",
            "target_x",
            "target_y",
            "pred_x_global",
            "pred_y_global",
            "target_x_global",
            "target_y_global",
        ]
        for col in pixel_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").round().astype("Int64")
        if "has_target" in df.columns:
            df["has_target"] = pd.to_numeric(df["has_target"], errors="coerce").fillna(0).astype("Int8")
            n_has_target = int((df["has_target"] == 1).sum())
            print(f"[INFO][tracking_inference] rows with target: {n_has_target}/{len(df)}")

        # Quick sanity stats on predictions (useful for debugging collapsed outputs)
        if "pred_x" in df.columns and "pred_y" in df.columns:
            pred_x_stats = _summary_stats(df["pred_x"])
            pred_y_stats = _summary_stats(df["pred_y"])
            pred_xg_stats = _summary_stats(df["pred_x_global"]) if "pred_x_global" in df.columns else None
            pred_yg_stats = _summary_stats(df["pred_y_global"]) if "pred_y_global" in df.columns else None
            print(f"[INFO][tracking_inference] pred_x stats: {pred_x_stats}")
            print(f"[INFO][tracking_inference] pred_y stats: {pred_y_stats}")
            if pred_xg_stats is not None or pred_yg_stats is not None:
                print(f"[INFO][tracking_inference] pred_x_global stats: {pred_xg_stats}")
                print(f"[INFO][tracking_inference] pred_y_global stats: {pred_yg_stats}")

        latlon_cols = ["pred_lat", "pred_lon", "target_lat", "target_lon"]
        for col in latlon_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").round(2)

        out_csv = preds_csv if not output_dir else os.path.join(output_dir, preds_csv)
        try:
            df.to_csv(out_csv, index=False)
            print(f"Saved tracking predictions to {out_csv}")
        except Exception as exc:
            print(f"Warning: could not save predictions CSV: {exc}")

        log_stats = {f"test_{k}": v for k, v in stats.items()}
        log_stats = {k: _format_log_value(k, v) for k, v in log_stats.items()}
        metrics_path = os.path.join(output_dir, "inference_tracking_metrics.txt")
        try:
            with open(metrics_path, "a") as handle:
                handle.write(json.dumps(log_stats) + "\n")
        except Exception as exc:
            print(f"Warning: could not write metrics to {metrics_path}: {exc}")

    dist.barrier() if dist.is_available() and dist.is_initialized() else None

    elapsed = time.time() - start
    if utils.is_main_process():
        print(f"Inference time: {datetime.timedelta(seconds=int(elapsed))}")

    return stats, gathered_results


def launch_inference_tracking(terminal_args: argparse.Namespace) -> None:
    args = prepare_tracking_args(machine=terminal_args.on)

    if terminal_args.csvfile:
        args.test_path = terminal_args.csvfile

    args.init_ckpt = terminal_args.inference_model
    args.load_for_test_mode = True
    args.distributed = False

    set_seeds(args.seed)

    rank, local_rank, world_size, _, _ = utils.get_resources()

    if world_size > 1:
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        args.distributed = True
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

    if args.log_dir and not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir, exist_ok=True)
    if args.output_dir and not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    setup_for_distributed(rank == 0)

    data_manager = DataManager(
        is_train=False,
        args=args,
        type_t="supervised",
        world_size=world_size,
        rank=rank,
        specify_data_path=args.test_path,
    )
    data_loader = data_manager.get_tracking_dataloader(args)
    _validate_tracking_dataset_paths(data_manager.dataset)

    model = create_tracking_model(args.model, **args.__dict__)
    model.to(device)
    load_checkpoint(model, args.init_ckpt, device)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], output_device=args.gpu
        )

    run_tracking_inference(model, data_loader, device, args.output_dir, terminal_args.preds_csv)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Cyclone tracking inference", add_help=False)
    parser.add_argument("--on", type=str, default="leonardo", help="[ewc, leonardo]")
    parser.add_argument("--csvfile", type=str, default="val_tracking.csv", help="CSV con clip di tracking")
    parser.add_argument(
        "--inference_model",
        type=str,
        default="output/checkpoint-tracking-best.pth",
        help="Checkpoint da utilizzare per l'inferenza",
    )
    parser.add_argument(
        "--preds_csv",
        type=str,
        default="tracking_inference_predictions.csv",
        help="Nome del CSV con le predizioni salvate",
    )
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    launch_inference_tracking(cli_args)
