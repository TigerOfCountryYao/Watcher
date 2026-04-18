from __future__ import annotations

import argparse
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset, random_split

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.dataset import ExcitementDataset, RandomExcitementDataset
from model.model import ExcitementModel
from utils.logger import setup_logger


@dataclass
class EpochMetrics:
    loss: float
    accuracy: float
    precision: float
    recall: float
    f1: float


class BinaryFocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: float | None = None) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        pt = probs * targets + (1.0 - probs) * (1.0 - targets)
        focal = (1.0 - pt).pow(self.gamma) * bce
        if self.alpha is not None:
            alpha_factor = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
            focal = alpha_factor * focal
        return focal.mean()


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_dataset(cfg: dict, logger) -> Dataset:
    data_cfg = cfg["data"]
    annotation_file = data_cfg.get("annotation_file", "")

    if annotation_file and Path(annotation_file).exists():
        dataset = ExcitementDataset(
            annotation_file=annotation_file,
            image_size=data_cfg["image_size"],
        )
        logger.info("Using ExcitementDataset: %s samples", len(dataset))
        return dataset

    dataset = RandomExcitementDataset(
        size=data_cfg["random_size"],
        image_size=data_cfg["image_size"],
    )
    logger.info("Using RandomExcitementDataset: %s samples", len(dataset))
    return dataset


def split_dataset(dataset: Dataset, cfg: dict) -> tuple[Dataset, Dataset]:
    val_split = float(cfg["train"].get("val_split", 0.2))
    if not 0.0 < val_split < 1.0:
        raise ValueError(f"train.val_split must be between 0 and 1, got {val_split}")

    total_size = len(dataset)
    if total_size < 2:
        raise ValueError("dataset must contain at least 2 samples for train/val split")

    val_size = max(1, int(total_size * val_split))
    train_size = total_size - val_size
    if train_size < 1:
        train_size = 1
        val_size = total_size - 1

    generator = torch.Generator().manual_seed(int(cfg["train"].get("seed", 42)))
    return random_split(dataset, [train_size, val_size], generator=generator)


def build_dataloader(dataset: Dataset, cfg: dict, shuffle: bool, device: torch.device) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=shuffle,
        num_workers=cfg["train"]["num_workers"],
        drop_last=shuffle,
        pin_memory=device.type == "cuda",
    )


def build_scheduler(optimizer: AdamW, cfg: dict, steps_per_epoch: int) -> LambdaLR:
    epochs = int(cfg["train"]["epochs"])
    warmup_epochs = float(cfg["train"].get("warmup_epochs", 0.0))
    min_lr_ratio = float(cfg["train"].get("min_lr_ratio", 0.1))

    total_steps = max(1, epochs * steps_per_epoch)
    warmup_steps = min(total_steps - 1, int(warmup_epochs * steps_per_epoch)) if total_steps > 1 else 0

    def lr_lambda(current_step: int) -> float:
        if warmup_steps > 0 and current_step < warmup_steps:
            return float(current_step + 1) / float(warmup_steps)

        if total_steps == warmup_steps:
            return 1.0

        progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def build_loss_fn(cfg: dict, train_dataset: Dataset) -> nn.Module:
    loss_name = str(cfg["train"].get("loss", "bce")).lower()
    if loss_name == "focal":
        return BinaryFocalLoss(
            gamma=float(cfg["train"].get("focal_gamma", 2.0)),
            alpha=cfg["train"].get("focal_alpha"),
        )

    pos_weight_value = cfg["train"].get("pos_weight")
    if pos_weight_value is not None:
        pos_weight = torch.tensor([float(pos_weight_value)], dtype=torch.float32)
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    return nn.BCEWithLogitsLoss()


def compute_metrics(logits: torch.Tensor, labels: torch.Tensor, threshold: float) -> EpochMetrics:
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).to(torch.int64)
    targets = labels.to(torch.int64)

    correct = (preds == targets).sum().item()
    total = max(1, targets.numel())

    tp = ((preds == 1) & (targets == 1)).sum().item()
    fp = ((preds == 1) & (targets == 0)).sum().item()
    fn = ((preds == 0) & (targets == 1)).sum().item()

    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-8, precision + recall)

    return EpochMetrics(
        loss=0.0,
        accuracy=correct / total,
        precision=precision,
        recall=recall,
        f1=f1,
    )


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    threshold: float,
    use_amp: bool,
    loss_fn: nn.Module,
) -> EpochMetrics:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    total_tp = 0
    total_fp = 0
    total_fn = 0

    autocast_device = device.type if device.type in ("cuda", "cpu") else "cpu"
    with torch.no_grad():
        for batch in dataloader:
            prev_frame = batch["prev_frame"].to(device, non_blocking=True)
            frame = batch["frame"].to(device, non_blocking=True)
            instructions = list(batch["instruction"])
            label = batch["label"].to(device, non_blocking=True)

            with torch.amp.autocast(device_type=autocast_device, enabled=use_amp):
                logits = model(prev_frame, frame, instructions).squeeze(1)
                loss = loss_fn(logits, label)

            probs = torch.sigmoid(logits)
            preds = (probs >= threshold).to(torch.int64)
            targets = label.to(torch.int64)

            total_loss += loss.item() * label.size(0)
            total_correct += (preds == targets).sum().item()
            total_count += targets.numel()
            total_tp += ((preds == 1) & (targets == 1)).sum().item()
            total_fp += ((preds == 1) & (targets == 0)).sum().item()
            total_fn += ((preds == 0) & (targets == 1)).sum().item()

    precision = total_tp / max(1, total_tp + total_fp)
    recall = total_tp / max(1, total_tp + total_fn)
    f1 = 2 * precision * recall / max(1e-8, precision + recall)

    return EpochMetrics(
        loss=total_loss / max(1, total_count),
        accuracy=total_correct / max(1, total_count),
        precision=precision,
        recall=recall,
        f1=f1,
    )


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: AdamW,
    scheduler: LambdaLR,
    cfg: dict,
    epoch: int,
    best_val_loss: float,
) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "config": cfg,
            "epoch": epoch,
            "best_val_loss": best_val_loss,
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = setup_logger("watcher-train", cfg["train"].get("log_file"))

    seed = int(cfg["train"].get("seed", 42))
    set_seed(seed)

    device = torch.device(cfg["train"].get("device", "cpu"))
    use_amp = bool(cfg["train"].get("use_amp", device.type == "cuda")) and device.type == "cuda"
    threshold = float(cfg["train"].get("metric_threshold", 0.5))

    model = ExcitementModel(**cfg["model"]).to(device)

    dataset = build_dataset(cfg, logger)
    train_dataset, val_dataset = split_dataset(dataset, cfg)
    train_loader = build_dataloader(train_dataset, cfg, shuffle=True, device=device)
    val_loader = build_dataloader(val_dataset, cfg, shuffle=False, device=device)

    optimizer = AdamW(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=float(cfg["train"].get("weight_decay", 0.01)),
    )
    loss_fn = build_loss_fn(cfg, train_dataset)
    if isinstance(loss_fn, nn.BCEWithLogitsLoss) and loss_fn.pos_weight is not None:
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=loss_fn.pos_weight.to(device))
    scheduler = build_scheduler(optimizer, cfg, steps_per_epoch=max(1, len(train_loader)))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    epochs = int(cfg["train"]["epochs"])
    out_dir = Path(cfg["train"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    best_ckpt_path = out_dir / "watcher_best.pt"
    latest_ckpt_path = out_dir / "watcher_latest.pt"

    best_val_loss = float("inf")
    patience = int(cfg["train"].get("early_stopping_patience", 5))
    min_delta = float(cfg["train"].get("early_stopping_min_delta", 0.0))
    epochs_without_improvement = 0
    global_step = 0
    autocast_device = device.type if device.type in ("cuda", "cpu") else "cpu"

    logger.info(
        "Training setup | train_samples=%s val_samples=%s device=%s use_amp=%s",
        len(train_dataset),
        len(val_dataset),
        device,
        use_amp,
    )

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for batch in train_loader:
            prev_frame = batch["prev_frame"].to(device, non_blocking=True)
            frame = batch["frame"].to(device, non_blocking=True)
            instructions = list(batch["instruction"])
            label = batch["label"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=autocast_device, enabled=use_amp):
                logits = model(prev_frame, frame, instructions).squeeze(1)
                loss = loss_fn(logits, label)

            scaler.scale(loss).backward()

            grad_clip_norm = float(cfg["train"].get("grad_clip_norm", 1.0))
            if grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running_loss += loss.item() * label.size(0)
            global_step += 1

        train_loss = running_loss / max(1, len(train_dataset))
        val_metrics = evaluate(
            model=model,
            dataloader=val_loader,
            device=device,
            threshold=threshold,
            use_amp=use_amp,
            loss_fn=loss_fn,
        )

        logger.info(
            "epoch=%s step=%s train_loss=%.6f val_loss=%.6f val_acc=%.4f val_precision=%.4f val_recall=%.4f val_f1=%.4f lr=%.8f",
            epoch + 1,
            global_step,
            train_loss,
            val_metrics.loss,
            val_metrics.accuracy,
            val_metrics.precision,
            val_metrics.recall,
            val_metrics.f1,
            optimizer.param_groups[0]["lr"],
        )

        save_checkpoint(
            latest_ckpt_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            cfg=cfg,
            epoch=epoch + 1,
            best_val_loss=best_val_loss,
        )

        if val_metrics.loss < best_val_loss - min_delta:
            best_val_loss = val_metrics.loss
            epochs_without_improvement = 0
            save_checkpoint(
                best_ckpt_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                cfg=cfg,
                epoch=epoch + 1,
                best_val_loss=best_val_loss,
            )
            logger.info("Saved best checkpoint: %s", best_ckpt_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                logger.info(
                    "Early stopping triggered at epoch %s after %s unimproved epochs.",
                    epoch + 1,
                    epochs_without_improvement,
                )
                break

    logger.info("Saved latest checkpoint: %s", latest_ckpt_path)


if __name__ == "__main__":
    main()
