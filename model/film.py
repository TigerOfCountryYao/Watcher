import torch
import torch.nn as nn


class FiLM(nn.Module):
    """Feature-wise linear modulation."""

    def __init__(self, feat_dim: int = 128, cond_dim: int = 128) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(cond_dim, feat_dim),
            nn.Hardswish(inplace=True),
            nn.Linear(feat_dim, feat_dim * 2),
        )

    def forward(self, feat: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        feat: (B, C, H, W)
        cond: (B, cond_dim)
        output: (B, C, H, W)
        """
        gb = self.mlp(cond)
        gamma, beta = gb.chunk(2, dim=-1)

        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return gamma * feat + beta
