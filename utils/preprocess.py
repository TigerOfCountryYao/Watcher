from __future__ import annotations

import cv2
import numpy as np
import torch


def preprocess_frame(frame: np.ndarray, image_size: int = 224) -> torch.Tensor:
    """Convert BGR numpy frame to normalized CHW float tensor."""
    resized = cv2.resize(frame, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    return tensor
