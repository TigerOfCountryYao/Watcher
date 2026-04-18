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
        "instruction": "watch for popup",
        "label": 1,
        "event_map": "path/to/event_map.png"
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

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        item = self.samples[idx]

        prev_path = item["prev_frame"]
        curr_path = item["frame"]

        prev = cv2.imread(prev_path)
        curr = cv2.imread(curr_path)
        if prev is None or curr is None:
            raise FileNotFoundError(f"failed to read frame pair: {prev_path}, {curr_path}")

        prev_t = preprocess_frame(prev, self.image_size)
        curr_t = preprocess_frame(curr, self.image_size)

        instruction = str(item["instruction"])
        label = torch.tensor(item["label"], dtype=torch.float32)
        sample: dict[str, torch.Tensor | str] = {
            "prev_frame": prev_t,
            "frame": curr_t,
            "instruction": instruction,
            "label": label,
        }
        if "event_map" in item and item["event_map"]:
            event_map_img = cv2.imread(str(item["event_map"]), cv2.IMREAD_GRAYSCALE)
            if event_map_img is None:
                raise FileNotFoundError(f"failed to read event_map: {item['event_map']}")
            event_map = cv2.resize(
                event_map_img,
                (self.image_size, self.image_size),
                interpolation=cv2.INTER_NEAREST,
            )
            sample["event_map"] = torch.from_numpy(event_map).float().unsqueeze(0) / 255.0
        return sample


class RandomExcitementDataset(Dataset):
    """Fallback dataset for quick smoke tests."""

    def __init__(self, size: int = 512, image_size: int = 224) -> None:
        self.size = size
        self.image_size = image_size
        self.instructions = [
            "watch for popup",
            "watch submit button state",
            "watch icon status",
            "watch target color",
        ]

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        prev = torch.rand(3, self.image_size, self.image_size)
        curr = torch.rand(3, self.image_size, self.image_size)
        instruction = self.instructions[idx % len(self.instructions)]
        label = torch.randint(0, 2, (), dtype=torch.int64).to(torch.float32)
        return {
            "prev_frame": prev,
            "frame": curr,
            "instruction": instruction,
            "label": label,
        }
