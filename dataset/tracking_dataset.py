import os
from typing import Tuple, Optional

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from .datasets import MedicanesClsDataset


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

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        video, _label_unused, folder_path = super().__getitem__(idx)
        row = self.df.iloc[idx]
        coords = torch.tensor([row[self.x_col], row[self.y_col]], dtype=torch.float32)
        return video, coords, folder_path
