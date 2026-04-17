from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class TextEncoder(nn.Module):
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        output_dim: int = 128,
        freeze: bool = True,
        cache_dir: str | None = None,
    ) -> None:
        super().__init__()
        resolved_cache_dir = str(Path(cache_dir).resolve()) if cache_dir else None
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=resolved_cache_dir)
        self.model = AutoModel.from_pretrained(model_name, cache_dir=resolved_cache_dir)
        self.proj = nn.Linear(self.model.config.hidden_size, output_dim)

        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, texts: list[str], device: torch.device) -> torch.Tensor:
        batch = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        batch = {key: value.to(device) for key, value in batch.items()}

        outputs = self.model(**batch)
        hidden = outputs.last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        return self.proj(pooled)
