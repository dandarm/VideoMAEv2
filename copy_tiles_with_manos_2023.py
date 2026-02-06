#!/usr/bin/env python3
"""Copy 2023 tile folders that contain a Manos cyclone center."""
from __future__ import annotations

import argparse
import ast
import csv
import math
import re
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple


TILE_NAME_RE = re.compile(
    r"^(?P<date>\d{2}-\d{2}-\d{4})_(?P<time>\d{4})_(?P<x>-?\d+(?:\.\d+)?)_(?P<y>-?\d+(?:\.\d+)?)$"
)
TILE_SIZE = 224


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copia le tile 2023 che contengono un centro ciclone (Manos)."
    )
    parser.add_argument("--input_dir", required=True, help="Cartella con subfolder tile (nome: DD-MM-YYYY_HHMM_offsetX_offsetY).")
    parser.add_argument("--output_dir", required=True, help="Cartella di destinazione per le tile copiate.")
    parser.add_argument(
        "--manos_file",
        default="medicane_data_input/medicanes_new_windows.csv",
        help="CSV Manos con colonne time, x_pix, y_pix (default: medicane_data_input/medicanes_new_windows.csv).",
    )
    parser.add_argument("--year", type=int, default=2023, help="Anno da filtrare (default: 2023).")
    return parser.parse_args()


def first_val(val):
    if val is None:
        return None
    if isinstance(val, (list, tuple)):
        return float(val[0]) if len(val) > 0 else None
    if isinstance(val, str):
        s = val.strip()
        if not s:
            return None
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, (list, tuple)):
                return float(parsed[0]) if len(parsed) > 0 else None
            return float(parsed)
        except Exception:
            try:
                return float(s)
            except Exception:
                return None
    try:
        return float(val)
    except Exception:
        return None


def round_hour(dt: datetime) -> datetime:
    base = dt.replace(minute=0, second=0, microsecond=0)
    if dt.minute > 30 or (dt.minute == 30 and (dt.second > 0 or dt.microsecond > 0)):
        return base + timedelta(hours=1)
    if dt.minute < 30:
        return base
    # exactly half: round to even hour
    if base.hour % 2 == 1:
        return base + timedelta(hours=1)
    return base


def load_tracks_by_time(manos_file: Path, year: int) -> Dict[datetime, List[Tuple[float, float]]]:
    tracks_by_time: Dict[datetime, List[Tuple[float, float]]] = {}
    with manos_file.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            time_str = row.get("time")
            if not time_str:
                continue
            try:
                dt = datetime.strptime(time_str.strip(), "%Y-%m-%d %H:%M:%S")
            except ValueError:
                continue
            dt_floor = round_hour(dt)
            if dt_floor.year != year:
                continue
            x = first_val(row.get("x_pix"))
            y = first_val(row.get("y_pix"))
            if x is None or y is None:
                continue
            if not (math.isfinite(x) and math.isfinite(y)):
                continue
            tracks_by_time.setdefault(dt_floor, []).append((x, y))
    return tracks_by_time


def main() -> None:
    args = parse_args()

    src_root = Path(args.input_dir).resolve()
    dst_root = Path(args.output_dir).resolve()
    manos_csv = Path(args.manos_file).resolve()

    if not src_root.exists():
        raise SystemExit(f"Input dir not found: {src_root}")
    if not manos_csv.exists():
        raise SystemExit(f"Manos CSV not found: {manos_csv}")

    tracks_by_time = load_tracks_by_time(manos_csv, args.year)

    folders = []
    for p in src_root.iterdir():
        if not p.is_dir():
            continue
        m = TILE_NAME_RE.match(p.name)
        if not m:
            continue
        date_str = m.group("date")
        time_str = m.group("time")
        if not date_str.endswith(str(args.year)):
            continue
        try:
            dt = datetime.strptime(f"{date_str}_{time_str}", "%d-%m-%Y_%H%M")
        except ValueError:
            continue
        off_x = float(m.group("x"))
        off_y = float(m.group("y"))
        folders.append((p, round_hour(dt), off_x, off_y))

    print(f"Found {len(folders)} tile folders for {args.year} in {src_root}")

    copied = 0
    skipped = 0
    missing_time = 0
    for p, t_floor, off_x, off_y in folders:
        points = tracks_by_time.get(t_floor)
        if not points:
            missing_time += 1
            continue
        hit = False
        for x, y in points:
            if off_x <= x < off_x + TILE_SIZE and off_y <= y < off_y + TILE_SIZE:
                hit = True
                break
        if not hit:
            skipped += 1
            continue
        dst = dst_root / p.name
        dst_root.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            copied += 1
            continue
        shutil.copytree(p, dst)
        copied += 1

    print(f"Copied {copied} folders into {dst_root}")
    print(f"Skipped (no cyclone in tile): {skipped}")
    print(f"Skipped (no Manos time match): {missing_time}")


if __name__ == "__main__":
    main()
