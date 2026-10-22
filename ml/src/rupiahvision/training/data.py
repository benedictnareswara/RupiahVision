"""Images, labels and transforms for the v1 model.

The model predicts the app's outputs directly: index 0 is "unknown", then the 7
denominations in labels.json order.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import cv2
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2

from rupiahvision.data.inventory import DATA_DIR, SPLITS, read_csv
from rupiahvision.labels import load_schema
from rupiahvision.public import INDEX_COLUMNS, index_path

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
OUTPUTS: tuple[str, ...] = load_schema().outputs


def load_split_rows(data_dir: Path = DATA_DIR) -> dict[str, list[dict]]:
    rows = read_csv(index_path(data_dir), INDEX_COLUMNS)
    return {split: [r for r in rows if r["split"] == split] for split in SPLITS}


def read_rgb(path: Path) -> torch.Tensor:
    """uint8 CHW RGB tensor, the input torchvision transforms expect."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise OSError(f"cannot read {path}")
    return torch.from_numpy(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).permute(2, 0, 1)


def train_transform(size: int) -> v2.Compose:
    """Augmentations for how a blind user's phone sees a note: any rotation, partial framing,
    tilt, poor light, blur and fingers in the way. No flips: a mirrored note never occurs."""
    return v2.Compose(
        [
            v2.RandomResizedCrop(size, scale=(0.35, 1.0), ratio=(0.6, 1.6), antialias=True),
            v2.RandomRotation(180),
            v2.RandomPerspective(distortion_scale=0.25, p=0.3),
            v2.ColorJitter(brightness=0.5, contrast=0.4, saturation=0.3, hue=0.03),
            v2.RandomApply([v2.GaussianBlur(kernel_size=7, sigma=(0.1, 2.5))], p=0.3),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            v2.RandomErasing(p=0.3, scale=(0.05, 0.25)),
        ]
    )


def eval_transform(size: int) -> v2.Compose:
    """Shorter side to `size`, then a centre square: the same crop the app takes of a frame."""
    return v2.Compose(
        [
            v2.Resize(size, antialias=True),
            v2.CenterCrop(size),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


class ImageDataset(Dataset):
    def __init__(self, rows: list[dict], transform: Callable, data_dir: Path = DATA_DIR):
        self.rows = rows
        self.transform = transform
        self.data_dir = data_dir
        self.targets = [OUTPUTS.index(r["label"]) for r in rows]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image = read_rgb(self.data_dir / self.rows[index]["path"])
        return self.transform(image), self.targets[index]
