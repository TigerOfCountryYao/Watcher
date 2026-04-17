from __future__ import annotations

import time
from pathlib import Path
from typing import Iterator

import torch
import yaml

from model.model import ExcitementModel
from pipeline.screen_capture import ScreenCapture
from utils.runtime import step_event_detector


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class WatcherRunner:
    def __init__(self, cfg: dict, checkpoint: str = "", use_device: bool = True) -> None:
        self.cfg = cfg
        self.device = torch.device(cfg["infer"].get("device", "cpu")) if use_device else None

        self.model = ExcitementModel(**cfg["model"])
        if self.device is not None:
            self.model = self.model.to(self.device)
        self.model.eval()

        ckpt_path = checkpoint or cfg["infer"].get("checkpoint", "")
        if ckpt_path:
            map_location = self.device if self.device is not None else "cpu"
            ckpt = torch.load(ckpt_path, map_location=map_location)
            self.model.load_state_dict(ckpt["model"])
            print(f"loaded checkpoint: {ckpt_path}")

        self.capture = ScreenCapture(
            mode=cfg["infer"].get("capture_mode", "desktop"),
            monitor=cfg["infer"].get("monitor", 1),
            window_title=cfg["infer"].get("window_title"),
        )
        self.instruction = cfg["infer"].get("instruction", "watch for popup")
        self.threshold = cfg["infer"].get("threshold", 0.5)
        self.image_size = cfg["data"].get("image_size", 224)
        self.sleep_seconds = cfg["infer"].get("sleep_seconds", 0.03)
        self.prev_frame_np = None

    def step(self) -> float | None:
        frame = self.capture.get_frame()
        result = step_event_detector(
            model=self.model,
            prev_frame_np=self.prev_frame_np,
            curr_frame_np=frame,
            instruction=self.instruction,
            image_size=self.image_size,
            device=self.device,
        )
        self.prev_frame_np = result.prev_frame_np
        if result.skipped:
            return None
        return result.score

    def run_forever(self) -> None:
        while True:
            score = self.step()
            if score is not None:
                print(f"event_prob: {score:.4f}")
                if score > self.threshold:
                    print("TRIGGER EVENT")
            time.sleep(self.sleep_seconds)

    def iter_scores(self) -> Iterator[float]:
        while True:
            score = self.step()
            time.sleep(self.sleep_seconds)
            if score is not None:
                yield score
