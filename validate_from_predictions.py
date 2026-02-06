#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd

from dataset.data_manager import BuildDataset
from medicane_utils.load_files import get_intervals_in_tracks_df
from model_analysis import evaluate_binary_classifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calcola metriche a partire dal CSV di predizioni (inference_predictions)."
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Cartella immagini (png con date nel nome).",
    )
    parser.add_argument(
        "--preds_csv",
        required=True,
        help="CSV delle predizioni (output di inference).",
    )
    parser.add_argument(
        "--manos_file",
        default="medicane_data_input/medicanes_new_windows.csv",
        help="CSV Manos con etichette (default: medicanes_new_windows.csv).",
    )
    parser.add_argument(
        "--merged_out",
        default="output/df_video_w_predictions.csv",
        help="CSV output con df_video + predictions (opzionale).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    preds_path = Path(args.preds_csv)
    if not preds_path.exists():
        raise FileNotFoundError(f"Predictions CSV not found: {preds_path}")

    manos_path = Path(args.manos_file)
    if not manos_path.exists():
        raise FileNotFoundError(f"Manos CSV not found: {manos_path}")

    # Ricostruisci df_video con etichette
    tracks_df = pd.read_csv(manos_path, parse_dates=["time", "start_time", "end_time"])
    data = BuildDataset(type="SUPERVISED")
    data.create_master_df(manos_file=str(manos_path), input_dir_images=args.input_dir, tracks_df=tracks_df)

    start_dt = data.master_df["datetime"].min()
    end_dt = data.master_df["datetime"].max()
    print(f"[{args.input_dir}] Intervallo immagini: {start_dt} -> {end_dt}")

    # Info intervalli Manos (clippati)
    try:
        intervals = get_intervals_in_tracks_df(tracks_df)
        if not intervals.empty:
            intervals = intervals[
                (intervals["max"] >= start_dt) & (intervals["min"] <= end_dt)
            ].reset_index(drop=True)
            print("Intervalli Manos (clippati all'intervallo immagini):", len(intervals))
    except Exception as exc:
        print(f"Warning: impossibile calcolare intervalli Manos: {exc}")

    data.make_df_video(output_dir=None, is_to_balance=False)
    if data.df_video is None or data.df_video.empty:
        raise RuntimeError("df_video è vuoto: impossibile calcolare metriche.")

    df_pred = pd.read_csv(preds_path)
    if "predictions" not in df_pred.columns:
        raise ValueError("Il CSV predizioni non ha la colonna 'predictions'.")
    if "path" not in df_pred.columns:
        raise ValueError("Il CSV predizioni non ha la colonna 'path'.")

    # Deduplica predizioni per path (se necessario)
    dup = df_pred.duplicated("path").sum()
    if dup:
        print(f"Warning: {dup} duplicati in predictions.csv, tengo la prima occorrenza.")
        df_pred = df_pred.drop_duplicates("path", keep="first")

    df_video = data.df_video.copy()
    df_video_w_predictions = df_video.merge(df_pred, on="path", how="inner")

    missing = len(df_pred) - len(df_video_w_predictions)
    if missing:
        print(f"Warning: {missing} righe di predizioni non hanno match in df_video.")

    # Confronta labels se presenti nel CSV predizioni
    if "labels" in df_pred.columns:
        mismatch = (df_video_w_predictions["label"].astype(int) != df_video_w_predictions["labels"].astype(int)).sum()
        if mismatch:
            print(f"Warning: {mismatch} mismatch tra label df_video e labels del CSV predizioni.")

    y_true = df_video_w_predictions["label"].astype(int).values
    y_pred = df_video_w_predictions["predictions"].astype(int).values
    score_cols = ["scores", "score", "probs", "prob", "logits"]
    y_score = None
    for col in score_cols:
        if col in df_video_w_predictions.columns:
            y_score = df_video_w_predictions[col].values
            print(f"Uso '{col}' come score per ROC/PR AUC.")
            break

    metrics = evaluate_binary_classifier(y_true, y_pred, y_score=y_score, show_report=False)

    print("Metriche:")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.6f}")
        else:
            print(f"  {k}: {v}")

    if "recall" in metrics:
        print(f"  pod (recall): {metrics['recall']:.6f}")

    if args.merged_out:
        out_path = Path(args.merged_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df_video_w_predictions.to_csv(out_path, index=False)
        print(f"df_video_w_predictions salvato in {out_path}")


if __name__ == "__main__":
    main()
