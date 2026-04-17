from __future__ import annotations

import json
from pathlib import Path

import cv2
import torch
from torch.utils.data import Dataset

from utils.preprocess import preprocess_frame


class ExcitementDataset(Dataset):
    """
    JSON schema:
    [
      {
        "prev_frame": "path/to/prev.png",
        "frame": "path/to/current.png",
        "instruction_id": 1,
        "target": 0.85
      }
    ]
    """

    def __init__(self, annotation_file: str, image_size: int = 224) -> None:
        self.image_size = image_size
        path = Path(annotation_file)
        if not path.exists():
            raise FileNotFoundError(f"annotation file not found: {annotation_file}")

        with path.open("r", encoding="utf-8") as f:
            self.samples = json.load(f)

        if not isinstance(self.samples, list):
            raise ValueError("annotation file must contain a list of samples")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = self.samples[idx]

        prev_path = item["prev_frame"]
        curr_path = item["frame"]

        prev = cv2.imread(prev_path)
        curr = cv2.imread(curr_path)
        if prev is None or curr is None:
            raise FileNotFoundError(f"failed to read frame pair: {prev_path}, {curr_path}")

        prev_t = preprocess_frame(prev, self.image_size)
        curr_t = preprocess_frame(curr, self.image_size)

        instruction = torch.tensor(item["instruction_id"], dtype=torch.long)
        target = torch.tensor(item["target"], dtype=torch.float32)

        return {
            "prev_frame": prev_t,
            "frame": curr_t,
            "instruction": instruction,
            "target": target,
        }


class RandomExcitementDataset(Dataset):
    """Fallback dataset for quick smoke tests."""

    def __init__(self, size: int = 512, image_size: int = 224, vocab_size: int = 50) -> None:
        self.size = size
        self.image_size = image_size
        self.vocab_size = vocab_size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        prev = torch.rand(3, self.image_size, self.image_size)
        curr = torch.rand(3, self.image_size, self.image_size)
        instruction = torch.randint(0, self.vocab_size, (1,), dtype=torch.long).squeeze(0)
        target = torch.rand((), dtype=torch.float32)
        return {
            "prev_frame": prev,
            "frame": curr,
            "instruction": instruction,
            "target": target,
        }
