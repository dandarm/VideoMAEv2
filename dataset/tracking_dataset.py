import os
from typing import Tuple

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from .datasets import MedicanesClsDataset


class MedicanesTrackDataset(MedicanesClsDataset):
    """Dataset for cyclone tracking.

    This dataset reuses :class:`MedicanesClsDataset` to load the video
    tiles but filters the entries so that only cyclonic tiles
    (``label == 1``) are kept.  Each sample returns the video tensor and
    the coordinates of the cyclone centre at the **last frame** of the
    tile.

    The annotation CSV is expected to contain at least the columns
    ``path``, ``label`` and two coordinate columns (default
    ``lon`` and ``lat``).  Coordinate values are returned as a tensor of
    shape ``[2]`` ordered as ``(lon, lat)``.
    """

    def __init__(
        self,
        anno_path: str,
        data_root: str = "",
        clip_len: int = 16,
        transform: transforms.Compose | None = None,
        lon_col: str = "lon",
        lat_col: str = "lat",
    ) -> None:
        super().__init__(
            anno_path=anno_path,
            data_root=data_root,
            mode="train",
            clip_len=clip_len,
            transform=transform,
        )

        # Keep only the cyclonic tiles (label == 1)
        if "label" in self.df.columns:
            self.df = self.df[self.df["label"] == 1].reset_index(drop=True)

        # Columns containing the coordinates of the last frame
        if lon_col not in self.df.columns or lat_col not in self.df.columns:
            raise ValueError(
                f"CSV must contain columns '{lon_col}' and '{lat_col}' for coordinates"
            )
        self.lon_col = lon_col
        self.lat_col = lat_col

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        video, _label_unused, folder_path = super().__getitem__(idx)
        row = self.df.iloc[idx]
        coords = torch.tensor(
            [row[self.lon_col], row[self.lat_col]], dtype=torch.float32
        )
        return video, coords, folder_path
