from __future__ import annotations

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int, stride: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_c,
            out_c,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class Backbone(nn.Module):
    """Lightweight multi-scale CNN backbone for screen event detection."""

    def __init__(self, out_dim: int = 128) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            ConvBlock(3, 32, stride=2),
            ConvBlock(32, 64, stride=2),
        )
        self.stage1 = nn.Sequential(
            ConvBlock(64, 64),
            ConvBlock(64, 64),
        )
        self.stage2 = nn.Sequential(
            ConvBlock(64, 128, stride=2),
            ConvBlock(128, 128),
        )
        self.stage3 = nn.Sequential(
            ConvBlock(128, out_dim, stride=2),
            ConvBlock(out_dim, out_dim),
        )
        self.feature_dims = {
            "stage1": 64,
            "stage2": 128,
            "stage3": out_dim,
        }

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.stem(x)
        stage1 = self.stage1(x)
        stage2 = self.stage2(stage1)
        stage3 = self.stage3(stage2)
        return {
            "stage1": stage1,
            "stage2": stage2,
            "stage3": stage3,
        }


if __name__ == "__main__":
    model = Backbone(out_dim=128)
    x = torch.randn(2, 3, 224, 224)
    outputs = model(x)
    for name, tensor in outputs.items():
        print(name, tensor.shape)
