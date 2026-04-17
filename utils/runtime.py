from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from utils.preprocess import preprocess_frame


@dataclass
class StepResult:
    prev_frame_np: np.ndarray
    score: float | None
    skipped: bool


def has_any_pixel_change(prev_frame: np.ndarray, curr_frame: np.ndarray) -> bool:
    return bool(np.any(prev_frame != curr_frame))


def score_frame_pair(
    model,
    prev_frame_np: np.ndarray,
    curr_frame_np: np.ndarray,
    instruction: str,
    image_size: int,
    device: torch.device | None = None,
) -> float:
    prev_frame_t = preprocess_frame(prev_frame_np, image_size=image_size).unsqueeze(0)
    curr_frame_t = preprocess_frame(curr_frame_np, image_size=image_size).unsqueeze(0)

    if device is not None:
        prev_frame_t = prev_frame_t.to(device)
        curr_frame_t = curr_frame_t.to(device)

    with torch.no_grad():
        logit = model(prev_frame_t, curr_frame_t, [instruction])
        return torch.sigmoid(logit).item()


def step_event_detector(
    model,
    prev_frame_np: np.ndarray | None,
    curr_frame_np: np.ndarray,
    instruction: str,
    image_size: int,
    device: torch.device | None = None,
) -> StepResult:
    if prev_frame_np is None:
        return StepResult(prev_frame_np=curr_frame_np, score=None, skipped=True)

    if not has_any_pixel_change(prev_frame_np, curr_frame_np):
        return StepResult(prev_frame_np=curr_frame_np, score=None, skipped=True)

    score = score_frame_pair(
        model=model,
        prev_frame_np=prev_frame_np,
        curr_frame_np=curr_frame_np,
        instruction=instruction,
        image_size=image_size,
        device=device,
    )
    return StepResult(prev_frame_np=curr_frame_np, score=score, skipped=False)
