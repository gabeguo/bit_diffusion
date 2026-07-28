"""EDM-style preconditioning for the bidirectional bridge (Appendix E).

Endpoint statistics are per-coordinate SECOND MOMENTS (no mean subtraction):
sigma0_sq = E[x_0^2], sigma1_sq = E[x_1^2], sigma01 = E[x_0 * x_1]. They are
logged during training as data/x_0_sq_mean, data/x_1_sq_mean, data/sigma01.
"""

import torch

from .loss import _expand_dims


class EDMPrecond:
    """Computes (c_in, c_skip, c_out) and the score-conversion denominator."""

    def __init__(self, sde, sigma0_sq=1.0, sigma1_sq=1.0, sigma01=0.0):
        assert sde.A == 0, "EDM preconditioning is derived for the driftless base process"
        self.sde = sde
        self.sigma0_sq = sigma0_sq
        self.sigma1_sq = sigma1_sq
        self.sigma01 = sigma01

    def coeffs(self, t, reverse):
        """Per-sample (c_in, c_skip, c_out, score_denom), each shaped like ``t``."""
        zeros, ones = torch.zeros_like(t), torch.ones_like(t)
        C_0t = self.sde.C(start=zeros, t_a=t, t_b=t)
        C_t1 = self.sde.C(start=t, t_a=ones, t_b=ones)
        C_01 = C_0t + C_t1
        a, b, c = C_0t / C_01, C_t1 / C_01, C_0t * C_t1 / C_01
        s0, s1, s01 = self.sigma0_sq, self.sigma1_sq, self.sigma01
        var_t = b**2 * s0 + a**2 * s1 + 2 * a * b * s01 + c
        c_in = var_t**-0.5
        if reverse:  # predict x_0
            cov = b * s0 + a * s01
            c_out = torch.sqrt(a**2 * (s0 * s1 - s01**2) + s0 * c) * c_in
            score_denom = C_0t
        else:  # predict x_1
            cov = a * s1 + b * s01
            c_out = torch.sqrt(b**2 * (s0 * s1 - s01**2) + s1 * c) * c_in
            score_denom = C_t1
        c_skip = cov / var_t
        return c_in, c_skip, c_out, score_denom


class EDMScoreWrapper:
    """Presents an x-prediction net through the score interface expected by
    ``SDE.dX_t`` via the Lemma E.1 substitution:
        score = ((c_skip - 1) * x_t + c_out * F) / C(t,1)   [forward; C(0,t) reverse]
    """

    def __init__(self, net, precond):
        self.net = net
        self.precond = precond

    def _score(self, x, t, reverse, net_fn, **net_kwargs):
        c_in, c_skip, c_out, denom = self.precond.coeffs(t, reverse=reverse)
        e = lambda s: _expand_dims(s, x)
        F = net_fn(x=e(c_in) * x, t=t, reverse=reverse, **net_kwargs)
        return (e(c_skip - 1) * x + e(c_out) * F) / e(denom)

    def __call__(self, x, t, y, x_cond, reverse=False, cond_mask=None):
        return self._score(x, t, reverse, self.net,
                           y=y, x_cond=x_cond, cond_mask=cond_mask)

    def forward_with_cfg(self, x, t, y, x_cond, reverse=False, cfg_scale=0.0):
        # Guiding F is equivalent to guiding the score: the coefficients are
        # shared between the conditional and unconditional branches, so the
        # skip term is guidance-invariant (Appendix E).
        return self._score(x, t, reverse, self.net.forward_with_cfg,
                           y=y, x_cond=x_cond, cfg_scale=cfg_scale)
