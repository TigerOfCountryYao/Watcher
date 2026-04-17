import torch
import torch.nn as nn

from model.backbone import Backbone
from model.film import FiLM
from model.excite_head import ExciteHead


class InstructionEncoder(nn.Module):
    def __init__(self, vocab_size: int = 50, emb_dim: int = 128) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)

    def forward(self, inst: torch.Tensor) -> torch.Tensor:
        return self.embedding(inst)


class ExcitementModel(nn.Module):
    def __init__(
        self,
        vocab_size: int = 50,
        emb_dim: int = 128,
        feat_dim: int = 128,
    ) -> None:
        super().__init__()
        self.encoder = Backbone(out_dim=feat_dim)
        self.inst_encoder = InstructionEncoder(vocab_size=vocab_size, emb_dim=emb_dim)
        self.film = FiLM(feat_dim=feat_dim, cond_dim=emb_dim)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = ExciteHead(dim=feat_dim)

    def forward(
        self,
        frame: torch.Tensor,
        instruction: torch.Tensor,
        prev_z: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        feat = self.encoder(frame)
        cond = self.inst_encoder(instruction)
        feat = self.film(feat, cond)

        z = self.pool(feat).flatten(1)
        if prev_z is None:
            return z, None

        delta = z - prev_z
        excite = self.head(delta)
        return z, excite


if __name__ == "__main__":
    model = ExcitementModel()
    frame = torch.randn(2, 3, 224, 224)
    inst = torch.tensor([1, 2], dtype=torch.long)

    z, excite = model(frame, inst, None)
    print("z:", z.shape)
    print("excite:", excite)
