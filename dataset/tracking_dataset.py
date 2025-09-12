import os
from typing import Tuple, Optional

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from .datasets import MedicanesClsDataset
from PIL import Image


class MedicanesTrackDataset(MedicanesClsDataset):
    """Dataset for cyclone tracking (pixel coordinates only).

    - Expects CSV columns: ``path``, ``x_pix``, ``y_pix``. Optionally ``label``.
    - If ``label`` exists, keeps only rows with ``label == 1``.
    - Returns: ``video`` tensor and ``coords`` tensor of shape ``[2]`` as ``(x, y)``.
    """

    def __init__(
        self,
        anno_path: str,
        data_root: str = "",
        clip_len: int = 16,
        transform: Optional[transforms.Compose] = None,
        # Column names for pixel-coordinates (required)
        x_col: str = "x_pix",
        y_col: str = "y_pix",
        ) -> None:
        super().__init__(
            anno_path=anno_path,
            data_root=data_root,
            mode="train",
            clip_len=clip_len,
            transform=transform,
        )

        # Keep only the cyclonic tiles (label == 1) if label exists
        if "label" in self.df.columns:
            self.df = self.df[self.df["label"] == 1].reset_index(drop=True)

        # Require pixel coordinate columns
        if x_col not in self.df.columns or y_col not in self.df.columns:
            raise ValueError(f"CSV must contain pixel coordinate columns '{x_col}' and '{y_col}'")
        self.x_col = x_col
        self.y_col = y_col

        # Coerce to numeric and drop rows with missing/non-finite coordinates
        before = len(self.df)
        self.df[self.x_col] = pd.to_numeric(self.df[self.x_col], errors='coerce')
        self.df[self.y_col] = pd.to_numeric(self.df[self.y_col], errors='coerce')
        mask_finite = np.isfinite(self.df[self.x_col].values) & np.isfinite(self.df[self.y_col].values)
        self.df = self.df[mask_finite].reset_index(drop=True)
        dropped = before - len(self.df)
        if dropped > 0:
            print(f"[INFO][TrackingDataset] Dropped {dropped} rows with missing/non-finite coordinates from {anno_path} (kept {len(self.df)}/{before}).")

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """Return video tensor, coords tensor (x,y), and folder path.

        Overrides parent to avoid requiring a 'label' column, since this is
        a regression target with pixel coordinates.
        """
        row = self.df.iloc[idx]
        folder_path = row['path']

        # If path is relative, compose with data_root
        if not os.path.isabs(folder_path):
            folder_path = os.path.join(self.data_root, folder_path)

        # Load up to clip_len frames from the folder
        frame_files = sorted(os.listdir(folder_path))
        frames = []
        for file in frame_files[: self.clip_len]:
            file_path = os.path.join(folder_path, file)
            try:
                img = Image.open(file_path).convert("RGB")
                img = self.transform(img)
                frames.append(img)
            except Exception:
                print(f"Problema nel caricamento di {file_path}")

        # Pad by repeating last frame if needed, to ensure clip_len
        if len(frames) != self.clip_len and len(frames) > 0:
            frames = frames + [frames[-1]] * (self.clip_len - len(frames))

        # [T, C, H, W] -> [C, T, H, W]
        video = torch.stack(frames, dim=0).permute(1, 0, 2, 3)

        # Build target coords as float tensor
        coords = torch.tensor([row[self.x_col], row[self.y_col]], dtype=torch.float32)
        return video, coords, folder_path
