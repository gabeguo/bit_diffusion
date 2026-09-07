from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F

from state_fate.baselines import noise_to_data_coeffs
from state_fate.plotting import save_latent_eval_figure, save_trajectory_figure


def _rbf_mmd(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    xy = torch.cat([x, y], dim=0)
    with torch.no_grad():
        distances = torch.cdist(xy, xy).flatten()
        distances = distances[distances > 0]
        sigma = (
            distances.median().clamp_min(1e-6)
            if len(distances)
            else torch.tensor(1.0, device=x.device)
        )
    gamma = 1.0 / (2.0 * sigma**2)
    k_xx = torch.exp(-gamma * torch.cdist(x, x).pow(2)).mean()
    k_yy = torch.exp(-gamma * torch.cdist(y, y).pow(2)).mean()
    k_xy = torch.exp(-gamma * torch.cdist(x, y).pow(2)).mean()
    return k_xx + k_yy - 2.0 * k_xy


def _knn_label_accuracy(
    generated: torch.Tensor,
    reference: torch.Tensor,
    query_labels: torch.Tensor,
    reference_labels: torch.Tensor,
) -> float:
    nn_idx = torch.cdist(generated, reference).argmin(dim=1)
    return (reference_labels[nn_idx] == query_labels).float().mean().item()


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def _model_call(
    *,
    model,
    x: torch.Tensor,
    t: torch.Tensor,
    y: torch.Tensor,
    x_cond: torch.Tensor,
    reverse: bool,
    cfg_scale: float,
) -> torch.Tensor:
    raw_model = _unwrap_model(model)
    if cfg_scale > 0 and hasattr(raw_model, "forward_with_cfg"):
        return raw_model.forward_with_cfg(
            x=x,
            t=t,
            y=y,
            x_cond=x_cond,
            reverse=reverse,
            cfg_scale=cfg_scale,
        )
    return model(x=x, t=t, y=y, x_cond=x_cond, reverse=reverse)


def _simulate_score(
    *,
    sde,
    x_start: torch.Tensor,
    x_cond: torch.Tensor,
    y: torch.Tensor,
    num_steps: int,
    reverse: bool,
    cfg_scale: float,
    sampler: str,
) -> torch.Tensor:
    return sde.simulate(
        x_start,
        num_steps=num_steps,
        reverse=reverse,
        x_cond=x_cond,
        y=y,
        cfg_scale=cfg_scale,
        ode=sampler == "ode",
    )


def _simulate_flow(
    *,
    model,
    x_start: torch.Tensor,
    x_cond: torch.Tensor,
    y: torch.Tensor,
    num_steps: int,
    reverse: bool,
    cfg_scale: float,
) -> torch.Tensor:
    x = x_start
    dt = 1.0 / max(1, num_steps)
    for i in range(num_steps):
        t = torch.full((x.shape[0],), i * dt, device=x.device, dtype=x.dtype)
        v = _model_call(
            model=model,
            x=x,
            t=t,
            y=y,
            x_cond=x_cond,
            reverse=reverse,
            cfg_scale=cfg_scale,
        )
        x = x + dt * v
    return x


def _predict_endpoint(
    *,
    model,
    x_start: torch.Tensor,
    x_cond: torch.Tensor,
    y: torch.Tensor,
    reverse: bool,
    cfg_scale: float,
) -> torch.Tensor:
    t = torch.zeros((x_start.shape[0],), device=x_start.device, dtype=x_start.dtype)
    return _model_call(
        model=model,
        x=x_start,
        t=t,
        y=y,
        x_cond=x_cond,
        reverse=reverse,
        cfg_scale=cfg_scale,
    )


def _simulate_noise_to_data(
    *,
    model,
    x_cond: torch.Tensor,
    y: torch.Tensor,
    num_steps: int,
    reverse: bool,
    cfg_scale: float,
) -> torch.Tensor:
    x = torch.randn_like(x_cond)
    all_t = torch.linspace(
        0.99, 0.0, num_steps + 1, device=x.device, dtype=torch.float32
    )
    for i in range(num_steps):
        t = all_t[i].expand(x.shape[0])
        t_next = all_t[i + 1].expand(x.shape[0])
        eps = _model_call(
            model=model,
            x=x,
            t=t,
            y=y,
            x_cond=x_cond,
            reverse=reverse,
            cfg_scale=cfg_scale,
        )
        alpha, sigma = noise_to_data_coeffs(t)
        alpha_next, sigma_next = noise_to_data_coeffs(t_next)
        x0_pred = (x - sigma.view(-1, 1) * eps) / alpha.view(-1, 1).clamp_min(1e-3)
        x = alpha_next.view(-1, 1) * x0_pred + sigma_next.view(-1, 1) * eps
    return x


@torch.inference_mode()
def evaluate_batch(
    *,
    model=None,
    sde=None,
    batch: dict[str, torch.Tensor],
    device: torch.device,
    num_steps: int,
    cfg_scale: float = 0.0,
    objective: str = "score",
    sampler: str = "sde",
) -> tuple[dict[str, float], dict[str, torch.Tensor]]:
    if objective == "score":
        if sde is None:
            raise ValueError("score evaluation requires sde")
        model = sde.score_network
    if model is None:
        raise ValueError("model is required")
    was_training = model.training
    model.eval()

    x_0 = batch["x_0"].to(device)
    x_1 = batch["x_1"].to(device)
    y = batch["y"].to(device)
    fate = batch["fate_label"].to(device)
    clone = batch["clone_id"].to(device)

    def simulate(x: torch.Tensor, reverse: bool) -> torch.Tensor:
        if objective == "score":
            return _simulate_score(
                sde=sde,
                x_start=x,
                x_cond=x,
                y=y,
                num_steps=num_steps,
                reverse=reverse,
                cfg_scale=cfg_scale,
                sampler=sampler,
            )
        if objective == "flow":
            return _simulate_flow(
                model=model,
                x_start=x,
                x_cond=x,
                y=y,
                num_steps=num_steps,
                reverse=reverse,
                cfg_scale=cfg_scale,
            )
        if objective == "endpoint":
            return _predict_endpoint(
                model=model,
                x_start=x,
                x_cond=x,
                y=y,
                reverse=reverse,
                cfg_scale=cfg_scale,
            )
        if objective == "noise":
            return _simulate_noise_to_data(
                model=model,
                x_cond=x,
                y=y,
                num_steps=num_steps,
                reverse=reverse,
                cfg_scale=cfg_scale,
            )
        raise ValueError(f"unknown objective: {objective}")

    x_1_fwd = simulate(x_0, False)
    x_0_rev = simulate(x_1, True)
    x_0_cycle = simulate(x_1_fwd, True)
    x_1_cycle = simulate(x_0_rev, False)

    metrics = {
        "forward_mse": F.mse_loss(x_1_fwd, x_1).item(),
        "reverse_mse": F.mse_loss(x_0_rev, x_0).item(),
        "cycle_x0_mse": F.mse_loss(x_0_cycle, x_0).item(),
        "cycle_x1_mse": F.mse_loss(x_1_cycle, x_1).item(),
        "forward_mmd": _rbf_mmd(x_1_fwd, x_1).item(),
        "reverse_mmd": _rbf_mmd(x_0_rev, x_0).item(),
        "forward_fate_knn_acc": _knn_label_accuracy(x_1_fwd, x_1, fate, fate),
        "forward_clone_knn_acc": _knn_label_accuracy(x_1_fwd, x_1, clone, clone),
        "reverse_clone_knn_acc": _knn_label_accuracy(x_0_rev, x_0, clone, clone),
    }
    tensors = {
        "x_0": x_0.detach().cpu(),
        "x_1": x_1.detach().cpu(),
        "x_1_fwd": x_1_fwd.detach().cpu(),
        "x_0_rev": x_0_rev.detach().cpu(),
        "x_0_cycle": x_0_cycle.detach().cpu(),
        "x_1_cycle": x_1_cycle.detach().cpu(),
        "context_label": y.detach().cpu(),
        "fate_label": fate.detach().cpu(),
        "clone_id": clone.detach().cpu(),
    }

    if was_training:
        model.train()
    return metrics, tensors


def save_eval_outputs(
    *,
    out_dir: str | Path,
    step: int,
    metrics: dict[str, float],
    tensors: dict[str, torch.Tensor],
    title: str,
    embedding: str = "pca",
    latent_embedding: str | None = None,
) -> None:
    latent_embedding = embedding if latent_embedding is None else latent_embedding
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / f"metrics_{step:07d}.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    tensor_path = out_dir / f"eval_tensors_{step:07d}.pt"
    torch.save(tensors, tensor_path)
    save_trajectory_figure(
        out_dir=out_dir,
        step=step,
        metrics=metrics,
        tensors=tensors,
        title=title,
        embedding=embedding,
    )
    save_latent_eval_figure(
        out_dir=out_dir,
        step=step,
        tensors=tensors,
        title=title,
        embedding=latent_embedding,
    )


def load_cached_eval_outputs(
    out_dir: str | Path,
    step: int | None = None,
) -> tuple[int, dict[str, float], dict[str, torch.Tensor]]:
    out_dir = Path(out_dir)
    if step is None:
        tensor_paths = sorted(out_dir.glob("eval_tensors_*.pt"))
        if not tensor_paths:
            raise FileNotFoundError(f"No cached eval tensors found in {out_dir}")
        tensor_path = tensor_paths[-1]
        step = int(tensor_path.stem.split("_")[-1])
    tensor_path = out_dir / f"eval_tensors_{step:07d}.pt"
    metrics_path = out_dir / f"metrics_{step:07d}.json"
    if not tensor_path.exists():
        raise FileNotFoundError(f"Missing cached eval tensors: {tensor_path}")
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing cached metrics: {metrics_path}")
    tensors = torch.load(tensor_path, map_location="cpu", weights_only=False)
    metrics = json.loads(metrics_path.read_text())
    return step, metrics, tensors
