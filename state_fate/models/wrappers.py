from __future__ import annotations

import torch
import torch.nn as nn


class DirectionalBridgeModel(nn.Module):
    def __init__(self, forward_model: nn.Module, reverse_model: nn.Module) -> None:
        super().__init__()
        self.forward_model = forward_model
        self.reverse_model = reverse_model

    def _select(self, reverse: bool) -> nn.Module:
        return self.reverse_model if reverse else self.forward_model

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
        return self._select(reverse)(
            x=x,
            t=t,
            y=y,
            x_cond=x_cond,
            reverse=reverse,
            cond_mask=cond_mask,
            return_repa=return_repa,
        )

    def forward_with_cfg(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
        x_cond: torch.Tensor,
        reverse: bool = False,
        cfg_scale: float = 0.0,
    ) -> torch.Tensor:
        return self._select(reverse).forward_with_cfg(
            x=x,
            t=t,
            y=y,
            x_cond=x_cond,
            reverse=reverse,
            cfg_scale=cfg_scale,
        )
