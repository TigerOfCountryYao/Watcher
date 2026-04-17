import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================
# 1. Backbone (轻量CNN)
# =========================
class SimpleCNN(nn.Module):
    def __init__(self, out_dim=128):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),

            nn.Conv2d(128, out_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.features(x)  # [B, C, H/16, W/16]


# =========================
# 2. Instruction Encoder
# =========================
class InstructionEncoder(nn.Module):
    """
    简化版本：用 embedding id 表示 instruction
    后面你可以换成 text encoder (CLIP / BERT)
    """
    def __init__(self, vocab_size=50, emb_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)

    def forward(self, inst):
        """
        inst: [B] int tensor
        """
        return self.embedding(inst)  # [B, D]


# =========================
# 3. FiLM (核心模块)
# =========================
class FiLM(nn.Module):
    def __init__(self, feat_dim=128, cond_dim=128):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(cond_dim, feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, feat_dim * 2)  # gamma + beta
        )

    def forward(self, x, cond):
        """
        x: [B, C, H, W]
        cond: [B, D]
        """

        gb = self.mlp(cond)  # [B, 2C]
        gamma, beta = gb.chunk(2, dim=-1)

        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)

        return gamma * x + beta


# =========================
# 4. Excitement Head
# =========================
class ExciteHead(nn.Module):
    def __init__(self, dim=128):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, delta_z):
        return self.net(delta_z)


# =========================
# 5. 主模型
# =========================
class ExcitementModel(nn.Module):
    """
    核心逻辑：

    frame + instruction
        ↓
    CNN
        ↓
    FiLM (task conditioning)
        ↓
    pooling → z_t
        ↓
    Δz
        ↓
    excite score
    """

    def __init__(self,
                 vocab_size=50,
                 emb_dim=128,
                 feat_dim=128):

        super().__init__()

        self.encoder = SimpleCNN(out_dim=feat_dim)

        self.inst_encoder = InstructionEncoder(
            vocab_size=vocab_size,
            emb_dim=emb_dim
        )

        self.film = FiLM(feat_dim=feat_dim, cond_dim=emb_dim)

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.head = ExciteHead(dim=feat_dim)

    # -------------------------
    # forward
    # -------------------------
    def forward(self, frame, instruction, prev_z=None):
        """
        frame: [B, 3, H, W]
        instruction: [B]
        prev_z: [B, C] or None
        """

        # 1. visual feature
        feat = self.encoder(frame)  # [B, C, H, W]

        # 2. instruction embedding
        cond = self.inst_encoder(instruction)  # [B, C]

        # 3. FiLM conditioning (关键)
        feat = self.film(feat, cond)

        # 4. global pooling → z_t
        z = self.pool(feat).squeeze(-1).squeeze(-1)  # [B, C]

        # 5. first frame
        if prev_z is None:
            return z, None

        # 6. temporal difference
        delta = z - prev_z

        # 7. excitement prediction
        excite = self.head(delta)

        return z, excite


# =========================
# 6. quick test
# =========================
if __name__ == "__main__":
    model = ExcitementModel()

    frame = torch.randn(2, 3, 224, 224)
    inst = torch.tensor([1, 2])

    z, e = model(frame, inst, None)
    print("z:", z.shape)
    print("excite:", e)