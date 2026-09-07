from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from sde_utils.loss import _expand_dims
from sde_utils.sde import SDE


class CosineDecayVolatilitySDE(SDE):
    def __init__(self, K: float, eps: float, score_network) -> None:
        super().__init__(A=0, score_network=score_network)
        if K <= 0:
            raise ValueError("K must be positive")
        if eps < 0:
            raise ValueError("eps must be non-negative")
        self.K = float(K)
        self.eps = float(eps)

    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        return self.eps + 0.5 * (self.K - self.eps) * (1.0 + torch.cos(math.pi * t))

    def phi(self, start: torch.Tensor, end: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(start)

    def C(
        self, start: torch.Tensor, t_a: torch.Tensor, t_b: torch.Tensor
    ) -> torch.Tensor:
        upper = torch.minimum(t_a, t_b)
        c = 0.5 * (self.K + self.eps)
        d = 0.5 * (self.K - self.eps)

        def integral(t: torch.Tensor) -> torch.Tensor:
            return (
                (c * c + 0.5 * d * d) * t
                + (2.0 * c * d / math.pi) * torch.sin(math.pi * t)
                + (d * d / (4.0 * math.pi)) * torch.sin(2.0 * math.pi * t)
            )

        return integral(upper) - integral(start)


def flow_matching_loss(
    *,
    model: torch.nn.Module,
    x_0: torch.Tensor,
    x_1: torch.Tensor,
    t: torch.Tensor,
    y: torch.Tensor,
    reverse: bool,
    cond_mask: torch.Tensor | None,
) -> torch.Tensor:
    left, right = (x_1, x_0) if reverse else (x_0, x_1)
    t_view = t.view(-1, *([1] * (left.dim() - 1)))
    x_t = (1.0 - t_view) * left + t_view * right
    target = right - left
    pred = model(
        x=x_t,
        t=t,
        y=y,
        x_cond=left,
        reverse=reverse,
        cond_mask=cond_mask,
    )
    return F.mse_loss(pred, target)


def endpoint_regression_loss(
    *,
    model: torch.nn.Module,
    x_0: torch.Tensor,
    x_1: torch.Tensor,
    y: torch.Tensor,
    reverse: bool,
    cond_mask: torch.Tensor | None,
) -> torch.Tensor:
    source, target = (x_1, x_0) if reverse else (x_0, x_1)
    t = torch.zeros(source.shape[0], device=source.device, dtype=source.dtype)
    pred = model(
        x=source,
        t=t,
        y=y,
        x_cond=source,
        reverse=reverse,
        cond_mask=cond_mask,
    )
    return F.mse_loss(pred, target)


def noise_to_data_coeffs(t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Variance-preserving noising coefficients for a conditional DDPM baseline.

    Here t=0 is clean data and t=1 is nearly pure Gaussian noise. We cap the
    numerical endpoint so deterministic sampling never divides by zero.
    """

    sigma = t.float().clamp(0.0, 0.999)
    alpha = torch.sqrt((1.0 - sigma.square()).clamp_min(1e-6))
    return alpha, sigma


def noise_to_data_loss(
    *,
    model: torch.nn.Module,
    x_0: torch.Tensor,
    x_1: torch.Tensor,
    t: torch.Tensor,
    y: torch.Tensor,
    reverse: bool,
    cond_mask: torch.Tensor | None,
) -> torch.Tensor:
    source, target = (x_1, x_0) if reverse else (x_0, x_1)
    alpha, sigma = noise_to_data_coeffs(t)
    eps = torch.randn_like(target)
    x_t = _expand_dims(alpha, target) * target + _expand_dims(sigma, target) * eps
    pred_eps = model(
        x=x_t,
        t=t,
        y=y,
        x_cond=source,
        reverse=reverse,
        cond_mask=cond_mask,
    )
    return F.mse_loss(pred_eps, eps)
