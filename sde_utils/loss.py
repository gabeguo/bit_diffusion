import torch
import torch.nn.functional as F


def _expand_dims(s: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """View a per-sample tensor ``s`` of shape ``(N,)`` as ``(N, 1, 1, ..., 1)``
    so it broadcasts against ``ref`` of arbitrary trailing shape.

    Works for any data dimensionality: 2-D points (N, L), latents (N, C, H, W),
    videos (N, T, C, H, W), etc.
    """
    return s.view(-1, *([1] * (ref.dim() - 1)))


# This is imported from https://github.com/gabeguo/abc_diffusion
def sample_p_base_x_t_cond_x_prev_x_next(
    sde,
    x_prev,
    x_next,
    t,
    t_prev,
    t_next,
):
    assert x_prev.shape == x_next.shape
    # calc mu
    mu_prior = _expand_dims(sde.phi(start=t_prev, end=t), x_prev) * x_prev
    mu_gain = sde.C(start=t_prev, t_a=t, t_b=t_next) / sde.C(start=t_prev, t_a=t_next, t_b=t_next)
    mu_innovation = x_next - _expand_dims(sde.phi(start=t_prev, end=t_next), x_prev) * x_prev
    mu = mu_prior + _expand_dims(mu_gain, x_prev) * mu_innovation

    # calc sigma
    sigma_sq_first_term = sde.C(start=t_prev, t_a=t, t_b=t)
    sigma_sq_second_term = sde.C(start=t_prev, t_a=t, t_b=t_next)**2 / sde.C(start=t_prev, t_a=t_next, t_b=t_next)
    sigma_sq = _expand_dims(sigma_sq_first_term - sigma_sq_second_term, x_prev)

    # sample
    return mu + torch.sqrt(sigma_sq) * torch.randn_like(x_prev)

# Specific, since we fix x_0 and x_1
def sample_p_base_x_t_cond_x_0_x_1(
    sde,
    x_0,
    x_1,
    t,
):
    return sample_p_base_x_t_cond_x_prev_x_next(
        sde=sde,
        x_prev=x_0,
        x_next=x_1,
        t=t,
        t_prev=torch.zeros_like(t),
        t_next=torch.ones_like(t),
    )


def sample_flow_matching_x_t(
    x_0: torch.Tensor,
    x_1: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    """Sample the deterministic rectified-flow interpolant."""
    assert x_0.shape == x_1.shape
    t_expanded = _expand_dims(t, x_0)
    return (1.0 - t_expanded) * x_0 + t_expanded * x_1

# This is imported from https://github.com/gabeguo/abc_diffusion
def grad_wrt_x_t_log_p_base_x_next_cond_x_t(
    sde,
    x_t,
    t,
    x_next,
    t_next,
):
    assert x_t.shape == x_next.shape

    shrink = sde.phi(start=t, end=t_next)
    first_term = shrink / sde.C(start=t, t_a=t_next, t_b=t_next)
    second_term = x_next - _expand_dims(shrink, x_t) * x_t
    return _expand_dims(first_term, x_t) * second_term

# Specific, since we fix x_0 and x_1
def grad_wrt_x_t_log_p_base_x_1_cond_x_t(
    sde,
    x_t,
    t,
    x_1,
):
    return grad_wrt_x_t_log_p_base_x_next_cond_x_t(
        sde=sde,
        x_t=x_t,
        t=t,
        x_next=x_1,
        t_next=torch.ones_like(t),
    )

# Also need to implement reverse target
def grad_wrt_x_t_log_p_base_x_t_cond_x_0(
    sde,
    x_t,
    t,
    x_0,
):
    t_start = torch.zeros_like(t)
    left_term = -1 / sde.C(start=t_start, t_a=t, t_b=t)
    right_term = x_t - _expand_dims(sde.phi(start=t_start, end=t), x_t) * x_0
    return _expand_dims(left_term, x_t) * right_term

def repa_phase_weights(t, phase):
    """Per-sample (w_text, w_image) multipliers that split REPA strength over
    the timestep. ``t`` is (N,) in [0,1] with t=0 the text endpoint and t=1 the
    image endpoint.

      equal    : (0.5, 0.5)              -- constant 50/50 split
      image_up : (1 - t, t)             -- image strength grows toward t=1
      text_up  : (t, 1 - t)             -- text strength grows toward t=1
    """
    if phase == "equal":
        half = torch.full_like(t, 0.5)
        return half, half
    if phase == "image_up":
        return 1.0 - t, t
    if phase == "text_up":
        return t, 1.0 - t
    raise ValueError(f"unknown repa phase {phase!r}")


def repa_text_loss(
    repa_feat,   # (N, D) raw projection from the model's text REPA head
    text_emb,    # (N, F) global text embedding (F >= D); first D dims used
    weight,      # (N,) per-sample phase weight
):
    """Per-sample-weighted 1 - cosine between the model's pooled REPA
    projection and the (MRL-truncated, re-normalized) global text embedding.
    Both sides in fp32 for bf16-autocast stability. The global text embedding
    is present for every row, so the head is always used in the graph and the
    weight sum is essentially never 0 (no DDP unused-param guard needed)."""
    d = repa_feat.shape[-1]
    assert text_emb.shape[-1] >= d, (
        f"text_emb dim {text_emb.shape[-1]} < repa_dim {d}"
    )
    assert len(repa_feat.shape) == len(text_emb.shape) == 2
    assert text_emb.shape[0] == repa_feat.shape[0]
    target = F.normalize(text_emb[..., :d].float(), dim=-1)
    pred = F.normalize(repa_feat.float(), dim=-1)
    cos = (pred * target).sum(dim=-1)  # (N,)
    return (weight * (1.0 - cos)).sum() / weight.sum().clamp_min(1e-8)


def repa_image_loss(
    repa_feat,   # (N, T, D) per-token projection from the model's image head
    dino_emb,    # (N, T, D) DINOv2 patch tokens
    present,     # (N,) bool: which rows have a real DINO target
    weight,      # (N,) per-sample phase weight
):
    """Per-token cosine alignment to DINOv2 patch tokens, averaged over tokens,
    masked to rows with a real target and weighted per-sample. Only a subset of
    rows carry a target, so the ``0 * sum`` term keeps the head in the autograd
    graph even when a local batch has zero present rows (DDP
    find_unused_parameters=False safety). Returns (loss, n_present)."""
    assert repa_feat.shape == dino_emb.shape, (repa_feat.shape, dino_emb.shape)
    pred = F.normalize(repa_feat.float(), dim=-1)
    target = F.normalize(dino_emb.float(), dim=-1)
    cos = (pred * target).sum(dim=-1)          # (N, T)
    per_sample = cos.mean(dim=1)               # (N,)
    w = weight * present.float()
    loss = (w * (1.0 - per_sample)).sum() / w.sum().clamp_min(1e-8)
    loss = loss + 0.0 * repa_feat.float().sum()
    return loss, present.float().sum()


def dsm_loss(
    model,
    sde,
    x_t,   # (N, ...)
    x_1,   # (N, ...)
    x_0,   # (N, ...)
    t,     # (N,)
    y,     # (N,) class labels
    reverse=False,  # bool
    cond_mask=None, # (N,) bool tensor
    x_cond_0=None,
    x_cond_1=None,
    return_repa=False,
):
    t_prev = torch.zeros_like(t)
    t_next = torch.ones_like(t)

    if reverse:
        target = grad_wrt_x_t_log_p_base_x_t_cond_x_0(
            sde=sde,
            x_t=x_t,
            t=t,
            x_0=x_0,
        )
    else:
        target = grad_wrt_x_t_log_p_base_x_1_cond_x_t(
            sde=sde,
            x_t=x_t,
            t=t,
            x_1=x_1,
        )

    assert (x_cond_0 is None) == reverse
    model_out = model(
        x=x_t,
        t=t,
        y=y,
        x_cond=x_cond_1 if reverse else x_cond_0,
        reverse=reverse,
        cond_mask=cond_mask,
        return_repa=return_repa,
    )
    pred, repa_feat = model_out if return_repa else (model_out, None)
    if reverse:
        weighting = sde.C(start=t_prev, t_a=t, t_b=t)
    else:
        weighting = sde.C(start=t, t_a=t_next, t_b=t_next) / sde.phi(start=t, end=t_next)
    assert t.dtype == t_next.dtype == t_prev.dtype == weighting.dtype == torch.float32
    loss = (pred - target) ** 2 * _expand_dims(weighting, pred)

    if return_repa:
        return loss.mean(), repa_feat
    return loss.mean()


def flow_matching_loss(
    model,
    x_t,   # (N, ...)
    x_1,   # (N, ...)
    x_0,   # (N, ...)
    t,     # (N,)
    y,     # (N,) class labels
    reverse=False,
    cond_mask=None,
    x_cond_0=None,
    x_cond_1=None,
    return_repa=False,
):
    """Rectified-flow velocity regression on the linear endpoint coupling.

    Both directions predict the forward-time velocity ``x_1 - x_0``. Reverse
    sampling uses a negative signed timestep in ``FlowMatchingODE.dX_t``.
    """
    assert x_t.shape == x_0.shape == x_1.shape
    target = x_1 - x_0
    model_out = model(
        x=x_t,
        t=t,
        y=y,
        x_cond=x_cond_1 if reverse else x_cond_0,
        reverse=reverse,
        cond_mask=cond_mask,
        return_repa=return_repa,
    )
    pred, repa_feat = model_out if return_repa else (model_out, None)
    loss = ((pred - target) ** 2).mean()
    if return_repa:
        return loss, repa_feat
    return loss


def edm_dsm_loss(
    model,
    precond,  # EDMPrecond
    x_t,   # (N, ...)
    x_1,   # (N, ...)
    x_0,   # (N, ...)
    t,     # (N,)
    y,     # (N,) class labels
    reverse=False,  # bool
    cond_mask=None, # (N,) bool tensor
    x_cond_0=None,
    x_cond_1=None,
    return_repa=False,
):
    """x-prediction loss with EDM-style preconditioning (Appendix E).

    Regresses the raw network against the unit-variance effective target
    (target - c_skip * x_t) / c_out. The lambda(t) = 1/c_out^2 weighting
    cancels c_out^2 exactly, so no explicit weighting is applied; at init
    (zeroed output layer) the loss is ~1 at every t.
    """
    c_in, c_skip, c_out, _ = precond.coeffs(t, reverse=reverse)
    assert t.dtype == c_out.dtype == torch.float32
    target = x_0 if reverse else x_1
    eff_target = (target - _expand_dims(c_skip, x_t) * x_t) / _expand_dims(c_out, x_t)

    assert (x_cond_0 is None) == reverse
    model_out = model(
        x=_expand_dims(c_in, x_t) * x_t,
        t=t,
        y=y,
        x_cond=x_cond_1 if reverse else x_cond_0,
        reverse=reverse,
        cond_mask=cond_mask,
        return_repa=return_repa,
    )
    pred, repa_feat = model_out if return_repa else (model_out, None)
    loss = ((pred - eff_target) ** 2).mean()
    if return_repa:
        return loss, repa_feat
    return loss
