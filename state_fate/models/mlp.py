from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int, max_period: float = 10000.0) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("time embedding dim must be even")
        self.dim = dim
        self.max_period = max_period

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if torch.any(t < 0) or torch.any(t > 1):
            raise ValueError("t must be in [0, 1]")
        scaled_t = 1000.0 * t
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(half, device=t.device, dtype=torch.float32)
            / half
        )
        args = scaled_t.float().unsqueeze(-1) * freqs.unsqueeze(0)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class FiLMResBlock(nn.Module):
    def __init__(self, hidden_dim: int, cond_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.film = nn.Linear(cond_dim, 2 * hidden_dim)

    def forward(self, h: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        scale, shift = self.film(F.silu(cond)).chunk(2, dim=-1)
        z = self.norm1(h) * (1 + scale) + shift
        z = self.fc1(F.silu(z))
        z = self.fc2(F.silu(self.norm2(z)))
        return h + z


class StateFateScoreMLP(nn.Module):
    """Vector score model for bidirectional endpoint bridges.

    The forward signature matches ``sde_utils.loss.dsm_loss`` and
    ``SDE.simulate``:

    - forward direction conditions on ``x_0``
    - reverse direction conditions on ``x_1``
    - ``cond_mask`` implements classifier-free endpoint dropout
    """

    def __init__(
        self,
        *,
        x_dim: int,
        num_classes: int,
        hidden_dim: int = 512,
        num_blocks: int = 6,
        time_dim: int = 128,
        class_dim: int = 128,
    ) -> None:
        super().__init__()
        self.x_dim = int(x_dim)
        self.num_classes = max(1, int(num_classes))

        self.time_emb = SinusoidalTimeEmbedding(time_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.class_emb = nn.Embedding(self.num_classes, class_dim)
        self.reverse_emb = nn.Embedding(2, class_dim)
        self.cond_endpoint_emb = nn.Embedding(2, class_dim)

        self.cond_proj = nn.Sequential(
            nn.Linear(self.x_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        cond_dim = time_dim + class_dim + class_dim + class_dim + hidden_dim
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.x_in = nn.Linear(self.x_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [FiLMResBlock(hidden_dim, hidden_dim) for _ in range(num_blocks)]
        )
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.x_out = nn.Linear(hidden_dim, self.x_dim)

        nn.init.zeros_(self.x_out.weight)
        nn.init.zeros_(self.x_out.bias)

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
        if x.shape[-1] != self.x_dim:
            raise ValueError(f"expected x_dim={self.x_dim}, got {x.shape[-1]}")

        if cond_mask is None:
            cond_mask = torch.ones((x.shape[0],), dtype=torch.bool, device=x.device)
        cond = x_cond * cond_mask.to(dtype=x.dtype).view(-1, 1)

        t_e = self.time_mlp(self.time_emb(t))
        y_e = self.class_emb(y.long().clamp(min=0, max=self.num_classes - 1))
        rev_idx = torch.full(
            (x.shape[0],), int(bool(reverse)), device=x.device, dtype=torch.long
        )
        r_e = self.reverse_emb(rev_idx)
        endpoint_e = self.cond_endpoint_emb(rev_idx)
        c_e = self.cond_proj(cond)
        c = self.cond_mlp(torch.cat([t_e, y_e, r_e, endpoint_e, c_e], dim=-1))

        h = self.x_in(x)
        for block in self.blocks:
            h = block(h, c)
        out = self.x_out(F.silu(self.out_norm(h)))
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
        cond = self.forward(
            x=x,
            t=t,
            y=y,
            x_cond=x_cond,
            reverse=reverse,
            cond_mask=None,
        )
        uncond = self.forward(
            x=x,
            t=t,
            y=y,
            x_cond=x_cond,
            reverse=reverse,
            cond_mask=torch.zeros((x.shape[0],), dtype=torch.bool, device=x.device),
        )
        return cond + cfg_scale * (cond - uncond)
