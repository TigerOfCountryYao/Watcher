from __future__ import annotations

import torch
import torch.nn as nn

from model.backbone import Backbone
from model.excite_head import ExciteHead
from model.film import FiLM
from model.text_encoder import TextEncoder


class ExcitementModel(nn.Module):
    def __init__(
        self,
        emb_dim: int = 128,
        feat_dim: int = 128,
        text_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        freeze_text_encoder: bool = True,
        text_model_cache_dir: str | None = None,
    ) -> None:
        super().__init__()
        self.encoder = Backbone(out_dim=feat_dim)
        self.text_encoder = TextEncoder(
            model_name=text_model_name,
            output_dim=emb_dim,
            freeze=freeze_text_encoder,
            cache_dir=text_model_cache_dir,
        )
        self.film = FiLM(feat_dim=feat_dim, cond_dim=emb_dim)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = ExciteHead(dim=feat_dim)

    def forward(
        self,
        prev_frame: torch.Tensor,
        curr_frame: torch.Tensor,
        instructions: list[str],
    ) -> torch.Tensor:
        prev_feat = self.encoder(prev_frame)
        curr_feat = self.encoder(curr_frame)
        delta_feat = curr_feat - prev_feat

        cond = self.text_encoder(instructions, device=curr_frame.device)
        conditioned = self.film(delta_feat, cond)
        pooled = self.pool(conditioned).flatten(1)
        return self.head(pooled)


if __name__ == "__main__":
    model = ExcitementModel()
    prev_frame = torch.randn(2, 3, 224, 224)
    curr_frame = torch.randn(2, 3, 224, 224)
    instructions = ["watch popup", "watch submit button"]

    logits = model(prev_frame, curr_frame, instructions)
    print("logits:", logits.shape)
