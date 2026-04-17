from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.model import ExcitementModel
from pipeline.screen_capture import ScreenCapture
from utils.preprocess import preprocess_frame


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--checkpoint", default="")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(cfg["infer"].get("device", "cpu"))

    model = ExcitementModel(**cfg["model"]).to(device)
    model.eval()

    ckpt_path = args.checkpoint or cfg["infer"].get("checkpoint", "")
    if ckpt_path:
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        print(f"loaded checkpoint: {ckpt_path}")

    instruction_id = cfg["infer"].get("instruction_id", 0)
    threshold = cfg["infer"].get("threshold", 0.5)
    image_size = cfg["data"].get("image_size", 224)

    instruction = torch.tensor([instruction_id], dtype=torch.long, device=device)
    capture = ScreenCapture(monitor=cfg["infer"].get("monitor", 1))

    prev_z = None
    while True:
        frame = capture.get_frame()
        frame_t = preprocess_frame(frame, image_size=image_size).unsqueeze(0).to(device)

        with torch.no_grad():
            z, excite = model(frame_t, instruction, prev_z)

        if excite is not None:
            score = excite.item()
            print(f"excite: {score:.4f}")
            if score > threshold:
                print("TRIGGER EVENT")

        prev_z = z
        time.sleep(0.03)


if __name__ == "__main__":
    main()
