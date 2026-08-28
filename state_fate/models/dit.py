from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from state_fate.models.mlp import SinusoidalTimeEmbedding


def _modulate(
    x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class VectorDiTBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, int(hidden_dim * mlp_ratio)),
            nn.GELU(approximate="tanh"),
            nn.Linear(int(hidden_dim * mlp_ratio), hidden_dim),
        )
        self.ada_ln = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, 6 * hidden_dim))
        nn.init.zeros_(self.ada_ln[-1].weight)
        nn.init.zeros_(self.ada_ln[-1].bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = self.ada_ln(cond).chunk(
            6, dim=-1
        )
        attn_in = _modulate(self.norm1(x), shift_a, scale_a)
        attn, _ = self.attn(attn_in, attn_in, attn_in, need_weights=False)
        x = x + gate_a.unsqueeze(1) * attn
        mlp_in = _modulate(self.norm2(x), shift_m, scale_m)
        return x + gate_m.unsqueeze(1) * self.mlp(mlp_in)


class StateFateDiT(nn.Module):
    def __init__(
        self,
        *,
        x_dim: int,
        num_classes: int,
        hidden_dim: int = 512,
        num_blocks: int = 8,
        time_dim: int = 128,
        class_dim: int = 128,
        token_dim: int = 8,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        if token_dim <= 0:
            raise ValueError("token_dim must be positive")
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.x_dim = int(x_dim)
        self.num_classes = max(1, int(num_classes))
        self.token_dim = int(token_dim)
        self.num_tokens = math.ceil(self.x_dim / self.token_dim)
        self.padded_dim = self.num_tokens * self.token_dim

        self.time_emb = SinusoidalTimeEmbedding(time_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.class_emb = nn.Embedding(self.num_classes, class_dim)
        self.reverse_emb = nn.Embedding(2, class_dim)
        self.cond_proj = nn.Sequential(
            nn.Linear(self.x_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.cond_mlp = nn.Sequential(
            nn.Linear(hidden_dim + class_dim + class_dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.token_in = nn.Linear(self.token_dim, hidden_dim)
        self.pos = nn.Parameter(torch.zeros(1, self.num_tokens, hidden_dim))
        self.blocks = nn.ModuleList(
            [
                VectorDiTBlock(hidden_dim, num_heads, mlp_ratio)
                for _ in range(num_blocks)
            ]
        )
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.token_out = nn.Linear(hidden_dim, self.token_dim)
        nn.init.zeros_(self.token_out.weight)
        nn.init.zeros_(self.token_out.bias)

    def _tokens(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.x_dim:
            raise ValueError(f"expected x_dim={self.x_dim}, got {x.shape[-1]}")
        if self.padded_dim != self.x_dim:
            x = F.pad(x, (0, self.padded_dim - self.x_dim))
        return x.view(x.shape[0], self.num_tokens, self.token_dim)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
        x_cond: torch.Tensor,
        reverse: bool = False,
        cond_mask: torch.Tensor | None = None,
        return_repa: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if x.shape != x_cond.shape:
            raise ValueError(
                f"x and x_cond must have the same shape, got {x.shape} and {x_cond.shape}"
            )
        if cond_mask is None:
            cond_mask = torch.ones((x.shape[0],), dtype=torch.bool, device=x.device)
        x_cond = x_cond * cond_mask.to(dtype=x.dtype).view(-1, 1)

        rev_idx = torch.full(
            (x.shape[0],),
            int(bool(reverse)),
            device=x.device,
            dtype=torch.long,
        )
        cond = self.cond_mlp(
            torch.cat(
                [
                    self.time_mlp(self.time_emb(t)),
                    self.class_emb(y.long().clamp(min=0, max=self.num_classes - 1)),
                    self.reverse_emb(rev_idx),
                    self.cond_proj(x_cond),
                ],
                dim=-1,
            )
        )

        h = self.token_in(self._tokens(x)) + self.pos
        for block in self.blocks:
            h = block(h, cond)
        out = self.token_out(F.silu(self.out_norm(h))).flatten(1)
        out = out[:, : self.x_dim]
        if return_repa:
            return out, {}
        return out

    def forward_with_cfg(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
        x_cond: torch.Tensor,
        reverse: bool = False,
        cfg_scale: float = 0.0,
    ) -> torch.Tensor:
        cond = self.forward(x=x, t=t, y=y, x_cond=x_cond, reverse=reverse)
        uncond = self.forward(
            x=x,
            t=t,
            y=y,
            x_cond=x_cond,
            reverse=reverse,
            cond_mask=torch.zeros((x.shape[0],), dtype=torch.bool, device=x.device),
        )
        return cond + cfg_scale * (cond - uncond)
