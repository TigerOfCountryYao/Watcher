from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


class FPNBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.lateral = nn.Conv2d(in_dim, out_dim, kernel_size=1, bias=False)
        self.output = nn.Sequential(
            nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.lateral(feat)


class Backbone(nn.Module):
    """MobileNetV3-Small backbone with a lightweight FPN."""

    def __init__(self, out_dim: int = 128, pretrained: bool = False) -> None:
        super().__init__()
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        backbone = mobilenet_v3_small(weights=weights)
        self.features = backbone.features

        # MobileNetV3-Small stages:
        # idx 3 -> stride 8, idx 8 -> stride 16, idx 12 -> stride 32
        self.return_layers = {
            "stage1": 3,
            "stage2": 8,
            "stage3": 12,
        }
        self.in_dims = {
            "stage1": 24,
            "stage2": 48,
            "stage3": 576,
        }

        self.fpn_blocks = nn.ModuleDict(
            {
                name: FPNBlock(in_dim=in_dim, out_dim=out_dim)
                for name, in_dim in self.in_dims.items()
            }
        )
        self.post_fpn = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(out_dim),
                    nn.ReLU(inplace=True),
                )
                for name in self.return_layers
            }
        )
        self.feature_dims = {
            "stage1": out_dim,
            "stage2": out_dim,
            "stage3": out_dim,
        }

    def _forward_backbone(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        outputs: dict[str, torch.Tensor] = {}
        for idx, layer in enumerate(self.features):
            x = layer(x)
            for name, layer_idx in self.return_layers.items():
                if idx == layer_idx:
                    outputs[name] = x
        return outputs

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        feats = self._forward_backbone(x)

        c3 = self.fpn_blocks["stage1"](feats["stage1"])
        c4 = self.fpn_blocks["stage2"](feats["stage2"])
        c5 = self.fpn_blocks["stage3"](feats["stage3"])

        p5 = self.post_fpn["stage3"](c5)
        p4 = self.post_fpn["stage2"](c4 + F.interpolate(p5, size=c4.shape[-2:], mode="nearest"))
        p3 = self.post_fpn["stage1"](c3 + F.interpolate(p4, size=c3.shape[-2:], mode="nearest"))

        return {
            "stage1": p3,
            "stage2": p4,
            "stage3": p5,
        }


if __name__ == "__main__":
    model = Backbone(out_dim=128, pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    outputs = model(x)
    for name, tensor in outputs.items():
        print(name, tensor.shape)
