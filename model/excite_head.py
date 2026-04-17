import torch
import torch.nn as nn


class ExciteHead(nn.Module):
    def __init__(self, dim: int = 128, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, delta_z: torch.Tensor) -> torch.Tensor:
        return self.net(delta_z)
