from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.backbone import Backbone
from model.film import FiLM
from model.text_encoder import TextEncoder


class ScaleFusionBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, cond_dim: int, dropout: float) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
        )
        self.film = FiLM(feat_dim=out_dim, cond_dim=cond_dim)
        self.refine = nn.Sequential(
            nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
        )

    def forward(self, feat: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        feat = self.proj(feat)
        feat = self.film(feat, cond)
        return self.refine(feat)


class SpatialEventHead(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.event_map_head = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 1, kernel_size=1),
        )
        self.score_head = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        event_logits = self.event_map_head(feat)
        b, _, h, w = event_logits.shape
        weights = torch.softmax(event_logits.view(b, 1, h * w), dim=-1).view(b, 1, h, w)
        pooled = (feat * weights).sum(dim=(2, 3))
        logits = self.score_head(pooled)
        return logits, event_logits


class ExcitementModel(nn.Module):
    def __init__(
        self,
        emb_dim: int = 128,
        feat_dim: int = 128,
        text_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        freeze_text_encoder: bool = True,
        text_model_cache_dir: str | None = None,
        fusion_dropout: float = 0.1,
        head_hidden_dim: int = 128,
        head_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder = Backbone(out_dim=feat_dim)
        self.text_encoder = TextEncoder(
            model_name=text_model_name,
            output_dim=emb_dim,
            freeze=freeze_text_encoder,
            cache_dir=text_model_cache_dir,
        )

        stage1_dim = self.encoder.feature_dims["stage1"]
        stage2_dim = self.encoder.feature_dims["stage2"]
        stage3_dim = self.encoder.feature_dims["stage3"]

        self.scale_fusion = nn.ModuleDict(
            {
                "stage1": ScaleFusionBlock(
                    in_dim=stage1_dim * 2,
                    out_dim=feat_dim,
                    cond_dim=emb_dim,
                    dropout=fusion_dropout,
                ),
                "stage2": ScaleFusionBlock(
                    in_dim=stage2_dim * 2,
                    out_dim=feat_dim,
                    cond_dim=emb_dim,
                    dropout=fusion_dropout,
                ),
                "stage3": ScaleFusionBlock(
                    in_dim=stage3_dim * 2,
                    out_dim=feat_dim,
                    cond_dim=emb_dim,
                    dropout=fusion_dropout,
                ),
            }
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(feat_dim * 3, feat_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(feat_dim),
            nn.ReLU(inplace=True),
        )
        self.head = SpatialEventHead(
            dim=feat_dim,
            hidden_dim=head_hidden_dim,
            dropout=head_dropout,
        )

    def forward(
        self,
        prev_frame: torch.Tensor,
        curr_frame: torch.Tensor,
        instructions: list[str],
        return_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        prev_feats = self.encoder(prev_frame)
        curr_feats = self.encoder(curr_frame)
        cond = self.text_encoder(instructions, device=curr_frame.device)

        fused_scales: list[torch.Tensor] = []
        target_size = curr_feats["stage1"].shape[-2:]

        for scale_name in ("stage1", "stage2", "stage3"):
            curr_feat = curr_feats[scale_name]
            prev_feat = prev_feats[scale_name]
            diff_feat = torch.abs(curr_feat - prev_feat)
            pair_feat = torch.cat([curr_feat, diff_feat], dim=1)
            fused = self.scale_fusion[scale_name](pair_feat, cond)
            if fused.shape[-2:] != target_size:
                fused = F.interpolate(fused, size=target_size, mode="bilinear", align_corners=False)
            fused_scales.append(fused)

        multi_scale_feat = self.fuse(torch.cat(fused_scales, dim=1))
        logits, event_map_logits = self.head(multi_scale_feat)

        if not return_aux:
            return logits
        return logits, {
            "event_map_logits": event_map_logits,
            "feature_map": multi_scale_feat,
        }


if __name__ == "__main__":
    model = ExcitementModel()
    prev_frame = torch.randn(2, 3, 224, 224)
    curr_frame = torch.randn(2, 3, 224, 224)
    instructions = ["watch popup", "watch submit button"]

    logits, aux = model(prev_frame, curr_frame, instructions, return_aux=True)
    print("logits:", logits.shape)
    print("event_map:", aux["event_map_logits"].shape)
