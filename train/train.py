from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.dataset import ExcitementDataset, RandomExcitementDataset
from model.model import ExcitementModel
from utils.logger import setup_logger


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_dataloader(cfg: dict, logger):
    data_cfg = cfg["data"]
    annotation_file = data_cfg.get("annotation_file", "")

    if annotation_file and Path(annotation_file).exists():
        dataset = ExcitementDataset(
            annotation_file=annotation_file,
            image_size=data_cfg["image_size"],
        )
        logger.info("Using ExcitementDataset: %s samples", len(dataset))
    else:
        dataset = RandomExcitementDataset(
            size=data_cfg["random_size"],
            image_size=data_cfg["image_size"],
        )
        logger.info("Using RandomExcitementDataset: %s samples", len(dataset))

    return DataLoader(
        dataset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["train"]["num_workers"],
        drop_last=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = setup_logger("watcher-train", cfg["train"].get("log_file"))

    device = torch.device(cfg["train"].get("device", "cpu"))
    model = ExcitementModel(**cfg["model"]).to(device)

    dataloader = build_dataloader(cfg, logger)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["train"]["lr"])

    epochs = cfg["train"]["epochs"]
    out_dir = Path(cfg["train"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    model.train()
    global_step = 0

    for epoch in range(epochs):
        running_loss = 0.0
        for batch in dataloader:
            prev_frame = batch["prev_frame"].to(device)
            frame = batch["frame"].to(device)
            instructions = list(batch["instruction"])
            label = batch["label"].to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(prev_frame, frame, instructions)
            loss = F.binary_cross_entropy_with_logits(logits.squeeze(1), label)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            global_step += 1

        avg_loss = running_loss / max(len(dataloader), 1)
        logger.info("epoch=%s step=%s loss=%.6f", epoch + 1, global_step, avg_loss)

    ckpt_path = out_dir / "watcher_latest.pt"
    torch.save({"model": model.state_dict(), "config": cfg}, ckpt_path)
    logger.info("Saved checkpoint: %s", ckpt_path)


if __name__ == "__main__":
    main()
