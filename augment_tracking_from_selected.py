#!/usr/bin/env python3
"""Build an augmented tracking dataset starting from selected tracking CSV rows.

Given a CSV like ``train_tracking_selezionati.csv`` (rows with columns
``path,start,end,x_pix,y_pix``), this script:

1. Parses each tile folder name ``DD-MM-YYYY_HHMM_offsetX_offsetY``.
2. Reconstructs the original 16-frame clip from ``--source_dataset``.
3. Recreates the original tile and generates ``N`` random offset variants.
4. Saves all clips under ``--output_dir``.
5. Writes a new tracking CSV suitable for VideoMAE tracking training.

By default, random offsets are constrained so the cyclone center remains
inside the 224x224 tile (positive tracking samples).
"""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image


FRAME_NAME_RE = re.compile(r"^airmass_rgb_(\d{8})_(\d{4})\.png$")
TILE_NAME_RE = re.compile(
    r"^(?P<date>\d{2}-\d{2}-\d{4})_(?P<time>\d{4})_(?P<x>-?\d+)_(?P<y>-?\d+)$"
)


@dataclass(frozen=True)
class TileRef:
    timestamp: datetime
    offset_x: int
    offset_y: int
    name: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Augment a selected tracking dataset by creating random offset tiles "
            "from original full-frame images."
        )
    )
    parser.add_argument(
        "--selected_csv",
        required=True,
        help="Input selected tracking CSV (e.g. train_tracking_selezionati.csv).",
    )
    parser.add_argument(
        "--source_dataset",
        required=True,
        help=(
            "Folder with source full frames named like "
            "airmass_rgb_YYYYMMDD_HHMM.png (flat or nested)."
        ),
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Output folder where augmented tile clips are written.",
    )
    parser.add_argument(
        "--output_csv",
        default=None,
        help=(
            "Output CSV path. Default: <output_dir>/train_tracking_selezionati_augmented.csv"
        ),
    )
    parser.add_argument(
        "--random_per_tile",
        type=int,
        default=10,
        help="Number of random offsets generated for each source tile.",
    )
    parser.add_argument(
        "--max_shift_px",
        type=int,
        default=100,
        help="Maximum offset displacement (in px) from the source tile offset.",
    )
    parser.add_argument(
        "--min_shift_px",
        type=int,
        default=40,
        help=(
            "Minimum offset displacement (in px) from the source tile offset. "
            "Use 0 to allow small shifts."
        ),
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=16,
        help="Frames per clip (default: 16).",
    )
    parser.add_argument(
        "--frame_step_minutes",
        type=int,
        default=5,
        help="Temporal spacing (minutes) between consecutive clip frames.",
    )
    parser.add_argument(
        "--timestamp_position",
        choices=("last", "center"),
        default="last",
        help=(
            "Interpretation of timestamp in tile folder name: "
            "'last' means frame 16, 'center' means frame 8."
        ),
    )
    parser.add_argument(
        "--tile_size",
        type=int,
        default=224,
        help="Tile side in pixels (default: 224).",
    )
    parser.add_argument(
        "--image_width",
        type=int,
        default=1290,
        help="Source full-frame width (default: 1290).",
    )
    parser.add_argument(
        "--image_height",
        type=int,
        default=420,
        help="Source full-frame height (default: 420).",
    )
    parser.add_argument(
        "--max_attempts_per_tile",
        type=int,
        default=5000,
        help="Maximum random sampling attempts per source tile.",
    )
    parser.add_argument(
        "--allow_center_outside",
        action="store_true",
        help="Allow random tiles where the center is outside the tile.",
    )
    parser.add_argument(
        "--skip_original",
        action="store_true",
        help="If set, do not recreate the original tile, only random variants.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed. If omitted, randomness is not fixed.",
    )
    return parser.parse_args()


def _parse_tile_name_from_path(path_str: str) -> TileRef:
    base = Path(str(path_str)).name
    m = TILE_NAME_RE.match(base)
    if not m:
        raise ValueError(
            f"Path basename does not match tile format DD-MM-YYYY_HHMM_x_y: {base}"
        )
    ts = datetime.strptime(
        f"{m.group('date')}_{m.group('time')}",
        "%d-%m-%Y_%H%M",
    )
    return TileRef(
        timestamp=ts,
        offset_x=int(m.group("x")),
        offset_y=int(m.group("y")),
        name=base,
    )


class FrameLocator:
    """Resolve source frame paths by frame filename, with lazy recursive index."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self._indexed = False
        self._name_to_path: Dict[str, Path] = {}

    def _build_index(self) -> None:
        print(f"[INFO] Building recursive frame index under: {self.root}")
        matched = 0
        for p in self.root.rglob("*.png"):
            if not FRAME_NAME_RE.match(p.name):
                continue
            matched += 1
            if p.name not in self._name_to_path:
                self._name_to_path[p.name] = p
        self._indexed = True
        print(
            f"[INFO] Indexed {matched} frame files "
            f"({len(self._name_to_path)} unique names)"
        )

    def find(self, frame_name: str) -> Optional[Path]:
        direct = self.root / frame_name
        if direct.exists():
            return direct
        if not self._indexed:
            self._build_index()
        return self._name_to_path.get(frame_name)


def _format_frame_name(ts: datetime) -> str:
    return f"airmass_rgb_{ts:%Y%m%d_%H%M}.png"


def _clip_timestamps(
    end_or_center_ts: datetime,
    num_frames: int,
    step_minutes: int,
    timestamp_position: str,
) -> List[datetime]:
    if timestamp_position == "last":
        center_index = num_frames - 1
    else:  # center
        center_index = (num_frames // 2) - 1

    start_ts = end_or_center_ts - timedelta(minutes=center_index * step_minutes)
    return [start_ts + timedelta(minutes=i * step_minutes) for i in range(num_frames)]


def _resolve_clip_frames(
    locator: FrameLocator,
    ts_reference: datetime,
    num_frames: int,
    step_minutes: int,
    timestamp_position: str,
) -> Optional[List[Path]]:
    dts = _clip_timestamps(
        end_or_center_ts=ts_reference,
        num_frames=num_frames,
        step_minutes=step_minutes,
        timestamp_position=timestamp_position,
    )
    out: List[Path] = []
    for dt in dts:
        name = _format_frame_name(dt)
        frame_path = locator.find(name)
        if frame_path is None:
            return None
        out.append(frame_path)
    return out


def _sample_random_offsets(
    base_offset: Tuple[int, int],
    center_global: Tuple[float, float],
    n_random: int,
    min_shift_px: float,
    max_shift_px: float,
    tile_size: int,
    image_width: int,
    image_height: int,
    max_attempts: int,
    rng: np.random.Generator,
    require_center_inside: bool,
) -> List[Tuple[int, int]]:
    base_x, base_y = base_offset
    cx, cy = center_global
    x_min, x_max = 0, image_width - tile_size
    y_min, y_max = 0, image_height - tile_size

    if require_center_inside:
        center_x_min = math.floor(cx - tile_size + 1)
        center_x_max = math.floor(cx)
        center_y_min = math.floor(cy - tile_size + 1)
        center_y_max = math.floor(cy)
        x_min = max(x_min, center_x_min)
        x_max = min(x_max, center_x_max)
        y_min = max(y_min, center_y_min)
        y_max = min(y_max, center_y_max)
        if x_min > x_max or y_min > y_max:
            return []

    selected: List[Tuple[int, int]] = []
    used = {(base_x, base_y)}
    attempts = 0
    while len(selected) < n_random and attempts < max_attempts:
        attempts += 1
        ox = int(rng.integers(x_min, x_max + 1))
        oy = int(rng.integers(y_min, y_max + 1))
        if (ox, oy) in used:
            continue
        dist = float(math.hypot(ox - base_x, oy - base_y))
        if dist < min_shift_px or dist > max_shift_px:
            continue
        used.add((ox, oy))
        selected.append((ox, oy))
    return selected


def _save_cropped_clip(
    frame_paths: Sequence[Path],
    dst_folder: Path,
    offset_x: int,
    offset_y: int,
    tile_size: int,
) -> None:
    dst_folder.mkdir(parents=True, exist_ok=True)
    for idx, src in enumerate(frame_paths, start=1):
        with Image.open(src).convert("RGB") as img:
            crop = img.crop((offset_x, offset_y, offset_x + tile_size, offset_y + tile_size))
            crop.save(dst_folder / f"img_{idx:05d}.png")


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    selected_csv = Path(args.selected_csv).resolve()
    source_dataset = Path(args.source_dataset).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    output_csv = (
        Path(args.output_csv).resolve()
        if args.output_csv
        else output_dir / "train_tracking_selezionati_augmented.csv"
    )
    metadata_csv = output_dir / "augmentation_metadata.csv"

    if not selected_csv.exists():
        raise FileNotFoundError(f"selected csv not found: {selected_csv}")
    if not source_dataset.exists():
        raise FileNotFoundError(f"source dataset not found: {source_dataset}")

    df = pd.read_csv(selected_csv)
    required = {"path", "x_pix", "y_pix"}
    if not required.issubset(df.columns):
        raise ValueError(
            f"Input CSV missing required columns. Found {list(df.columns)}, "
            f"required {sorted(required)}"
        )

    # Keep only rows with finite targets.
    x_num = pd.to_numeric(df["x_pix"], errors="coerce")
    y_num = pd.to_numeric(df["y_pix"], errors="coerce")
    valid_mask = np.isfinite(x_num.values) & np.isfinite(y_num.values)
    dropped = int((~valid_mask).sum())
    if dropped > 0:
        print(f"[WARN] Dropping {dropped} rows with non-finite x_pix/y_pix.")
    df = df.loc[valid_mask].copy().reset_index(drop=True)
    df["x_pix"] = pd.to_numeric(df["x_pix"], errors="coerce")
    df["y_pix"] = pd.to_numeric(df["y_pix"], errors="coerce")

    locator = FrameLocator(source_dataset)

    out_rows: List[dict] = []
    meta_rows: List[dict] = []
    skipped_parse = 0
    skipped_frames = 0
    generated_clips = 0
    expected_per_row = args.random_per_tile + (0 if args.skip_original else 1)
    print("\nSalvataggio videotile in corso...", end="\t")

    for idx, row in df.iterrows():
        src_path = str(row["path"])
        try:
            tile_ref = _parse_tile_name_from_path(src_path)
        except ValueError:
            skipped_parse += 1
            continue

        clip_frames = _resolve_clip_frames(
            locator=locator,
            ts_reference=tile_ref.timestamp,
            num_frames=args.num_frames,
            step_minutes=args.frame_step_minutes,
            timestamp_position=args.timestamp_position,
        )
        if clip_frames is None:
            skipped_frames += 1
            continue

        base_offset = (tile_ref.offset_x, tile_ref.offset_y)
        center_global = (
            float(tile_ref.offset_x + float(row["x_pix"])),
            float(tile_ref.offset_y + float(row["y_pix"])),
        )

        random_offsets = _sample_random_offsets(
            base_offset=base_offset,
            center_global=center_global,
            n_random=args.random_per_tile,
            min_shift_px=float(args.min_shift_px),
            max_shift_px=float(args.max_shift_px),
            tile_size=args.tile_size,
            image_width=args.image_width,
            image_height=args.image_height,
            max_attempts=args.max_attempts_per_tile,
            rng=rng,
            require_center_inside=not args.allow_center_outside,
        )

        variants: List[Tuple[str, int, int]] = []
        if not args.skip_original:
            variants.append(("orig", base_offset[0], base_offset[1]))
        for j, (ox, oy) in enumerate(random_offsets, start=1):
            variants.append((f"rand{j:02d}", ox, oy))

        if len(variants) < expected_per_row:
            print(
                f"[WARN] Row {idx}: generated {len(variants)} variants "
                f"(expected {expected_per_row})"
            )

        sample_root = output_dir / "tiles" / f"row_{idx:06d}"
        for variant_name, ox, oy in variants:
            folder_name = f"{tile_ref.timestamp:%d-%m-%Y_%H%M}_{ox}_{oy}"
            dst_folder = sample_root / variant_name / folder_name
            _save_cropped_clip(
                frame_paths=clip_frames,
                dst_folder=dst_folder,
                offset_x=ox,
                offset_y=oy,
                tile_size=args.tile_size,
            )
            generated_clips += 1
            if generated_clips % 50 == 0:
                print(generated_clips, end=" ", flush=True)

            rel_x = float(center_global[0] - ox)
            rel_y = float(center_global[1] - oy)
            shift_x = int(ox - base_offset[0])
            shift_y = int(oy - base_offset[1])
            shift_dist = float(math.hypot(shift_x, shift_y))

            out_rows.append(
                {
                    "path": str(dst_folder),
                    "start": 1,
                    "end": args.num_frames,
                    "x_pix": rel_x,
                    "y_pix": rel_y,
                    "label": 1,
                }
            )
            meta_rows.append(
                {
                    "path": str(dst_folder),
                    "variant": variant_name,
                    "source_row_idx": idx,
                    "source_path": src_path,
                    "timestamp": tile_ref.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                    "offset_x": ox,
                    "offset_y": oy,
                    "base_offset_x": base_offset[0],
                    "base_offset_y": base_offset[1],
                    "shift_x_from_base": shift_x,
                    "shift_y_from_base": shift_y,
                    "shift_dist_from_base_px": shift_dist,
                    "center_x_global": center_global[0],
                    "center_y_global": center_global[1],
                    "target_x_rel": rel_x,
                    "target_y_rel": rel_y,
                }
            )

        if (idx + 1) % 50 == 0:
            print(
                f"[INFO] Processed {idx + 1}/{len(df)} rows | "
                f"generated clips: {generated_clips}",
                end="\t",
                flush=True,
            )
    print()

    out_df = pd.DataFrame(out_rows)
    meta_df = pd.DataFrame(meta_rows)
    out_df.to_csv(output_csv, index=False)
    meta_df.to_csv(metadata_csv, index=False)

    print("\n=== Done ===")
    print(f"Input rows (finite targets): {len(df)}")
    print(f"Generated clips: {generated_clips}")
    print(f"Skipped rows (bad tile name): {skipped_parse}")
    print(f"Skipped rows (missing source frames): {skipped_frames}")
    print(f"Output CSV: {output_csv}")
    print(f"Metadata CSV: {metadata_csv}")


if __name__ == "__main__":
    main()
