import torch
import torch.nn as nn


# =========================
# Basic Conv Block
# =========================
class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()

        self.conv = nn.Conv2d(
            in_c,
            out_c,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )

        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


# =========================
# Backbone (Light CNN)
# =========================
class Backbone(nn.Module):
    """
    输出：
        feature map: [B, C, H/16, W/16]
    """

    def __init__(self, out_dim=128):
        super().__init__()

        self.stem = nn.Sequential(
            ConvBlock(3, 32, stride=2),   # H/2
            ConvBlock(32, 64, stride=2),  # H/4
        )

        self.stage1 = nn.Sequential(
            ConvBlock(64, 64),
            ConvBlock(64, 64),
        )

        self.stage2 = nn.Sequential(
            ConvBlock(64, 128, stride=2), # H/8
            ConvBlock(128, 128),
        )

        self.stage3 = nn.Sequential(
            ConvBlock(128, out_dim, stride=2), # H/16
            ConvBlock(out_dim, out_dim),
        )

        self.out_dim = out_dim

    def forward(self, x):
        """
        x: [B, 3, H, W]
        """

        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)

        return x  # [B, C, H/16, W/16]


# =========================
# Optional: quick test
# =========================
if __name__ == "__main__":
    model = Backbone(out_dim=128)

    x = torch.randn(2, 3, 224, 224)
    y = model(x)

    print("output:", y.shape)