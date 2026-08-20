"""
Training script for the bidirectional text-to-image DiT bridge.

Trains a single ``DiTWithCrossAttention`` model that bridges two endpoints
expressed in the SD-1.x VAE latent shape (4, 32, 32):
    - ``x_0`` = Qwen3 text embedding (4096,) viewed as (4, 32, 32)
    - ``x_1`` = SD VAE image latent (4, 32, 32)

A single Brownian-bridge SDE is fit in both directions:
    forward (text -> image)  --  ``reverse=False``
    reverse (image -> text)  --  ``reverse=True``

Launch (8x A100 80GB):
    cd BiB/text_to_image
    PYTHONPATH=.. torchrun --nproc-per-node=8 train.py \\
        --data-root /path/to/cc_by_sa_latents
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
from dataclasses import replace
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path
from time import time
from typing import Iterator

import torch
import torch.distributed as dist
import torch.optim as optim
from torch.amp import autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.data.distributed import DistributedSampler

import wandb

from data_utils.latent_dataset import (
    CommonCatalogLatentDataset,
    random_split_indices,
)
from eval_metrics import compute_fid_distributed, compute_text_decode_distributed
from eval_plot import eval_and_log_visuals
from time_sampling import logit_normal_timestep_sample, uniform_timestep_sample
from models.dit import DiT_models
from models.token_decoder import SharedTokenDecoder
from sde_utils.loss import (
    dsm_loss,
    edm_dsm_loss,
    flow_matching_loss,
    repa_image_loss,
    repa_phase_weights,
    repa_text_loss,
    sample_flow_matching_x_t,
    sample_p_base_x_t_cond_x_0_x_1,
)
from sde_utils.precond import EDMPrecond, EDMScoreWrapper
from sde_utils.sde import (
    CosineDecayingVolatilitySDE,
    FlowMatchingODE,
    PeriodicVolatilitySDE,
    UniformVolatilitySDE,
)
from token_bridge import (
    BRIDGE_RUNTIME_PRESETS,
    PROMPT_NUM_CLASSES,
    TOKEN_LAYOUTS,
    bridge_config_from_manifest,
    bridge_to_token_flat,
    prepare_bridge_batch,
)


# ---------------------------------------------------------------------------
# DDP helpers
# ---------------------------------------------------------------------------

def setup_ddp() -> tuple[int, int, int, dist.ProcessGroup]:
    """Init NCCL and return ``(rank, world_size, local_rank, eval_pg)``.

    The default group uses a short 5 min timeout so a stalled training
    collective fails fast (and the batch-script retry loop requeues). ``eval_pg``
    is a separate long-timeout group for eval/checkpoint barriers, where rank 0
    legitimately does minutes of solo work (sampling/decode/save) while the
    other ranks wait at the barrier (up to an hour, dominated by evaluation
    visualization).
    """
    # Resolve + set the GPU BEFORE init_process_group so we can pass an explicit
    # device_id. Without it, NCCL "guesses" the device from the global rank,
    # which both emits a warning and can hang under heterogeneous rank->GPU
    # mappings (multi-node). Under torchrun LOCAL_RANK is always set.
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        "nccl",
        timeout=timedelta(minutes=5),
        device_id=torch.device(f"cuda:{local_rank}"),
    )
    eval_pg = dist.new_group(timeout=timedelta(minutes=60))
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    return rank, world_size, local_rank, eval_pg


def _find_latest_checkpoint(out_dir: str) -> str | None:
    """Newest ``*/checkpoints/step_*.pt`` under ``out_dir`` (errors on ties)."""
    ckpts = list(Path(out_dir).glob("*/checkpoints/step_*.pt"))
    if not ckpts:
        return None
    step_of = lambda p: int(p.stem.split("_")[1])
    max_step = max(step_of(p) for p in ckpts)
    latest = [p for p in ckpts if step_of(p) == max_step]
    if len(latest) > 1:
        raise RuntimeError(
            f"--auto-resume: multiple checkpoints at max step {max_step}: {latest}"
        )
    return str(latest[0])


_EVAL_CHECKPOINT_ARGS = (
    "model", "bridge_preset", "use_token_text_bridge", "token_layout",
    "x0_cond_source", "text_token_scale", "forward_cond_scale",
    "no_reverse", "no_forward", "text_as_noise", "image_as_noise",
    "num_classes", "force_unconditional", "prompt_kind_dropout",
    "token_decoder_hidden_dim", "repa_text", "repa_text_dim",
    "repa_text_layer", "repa_image", "repa_image_layer", "sde",
    "periodic_sde_alpha", "periodic_sde_k", "periodic_sde_eps", "K",
    "edm_precond", "sigma0_sq", "sigma1_sq", "sigma01", "vae_ckpt",
)


def _validate_eval_checkpoint_args(args, ckpt, repa_image_dim) -> None:
    """Fail before eval if the current model semantics differ from the checkpoint."""
    saved = ckpt["args"]
    mismatches = []
    for key in _EVAL_CHECKPOINT_ARGS:
        if key == "force_unconditional" and key not in saved:
            continue  # Backward compatibility with pre-flow-matching checkpoints.
        current, expected = getattr(args, key), saved.get(key, "<missing>")
        if current != expected:
            mismatches.append((key, current, expected))

    saved_repa_dim = next((
        value.shape[0] for key, value in ckpt["ema"].items()
        if key.endswith("repa_image_head_forward.4.weight")
    ), None)
    if repa_image_dim != saved_repa_dim:
        mismatches.append(("repa_image_dim", repa_image_dim, saved_repa_dim))

    saved_root = saved.get("data_root")
    current_cfg = Path(args.data_root) / "token_embed_config.json"
    saved_cfg = Path(saved_root) / "token_embed_config.json" if saved_root else None
    if current_cfg.is_file() and saved_cfg is not None and saved_cfg.is_file():
        current = json.loads(current_cfg.read_text())["config"]["text_model"]
        expected = json.loads(saved_cfg.read_text())["config"]["text_model"]
        if current != expected:
            mismatches.append(("tokenizer_text_model", current, expected))

    if mismatches:
        detail = "\n".join(
            f"  {key}: current={current!r}, checkpoint={expected!r}"
            for key, current, expected in mismatches
        )
        raise ValueError(f"--eval-only arguments do not match the checkpoint:\n{detail}")


def cleanup_ddp() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------

@torch.no_grad()
def update_ema(ema_model: torch.nn.Module, model: torch.nn.Module, decay: float) -> None:
    """Step the EMA model towards the current model parameters."""
    ema_params = dict(ema_model.named_parameters())
    for name, param in model.named_parameters():
        if not param.requires_grad: # NOTE: avoids small numerical change of pos_embed
            continue
        ema_params[name].data.mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model: torch.nn.Module, flag: bool = True) -> None:
    for p in model.parameters():
        p.requires_grad = flag


# ---------------------------------------------------------------------------
# Train step
# ---------------------------------------------------------------------------




def token_decoder_loss(
    *,
    decoder: SharedTokenDecoder,
    sde,
    x_0: torch.Tensor,
    token_ids: torch.Tensor,
    token_mask: torch.Tensor,
    noise_t_max: float,
    pad_zero_prob: float,
    token_layout: str,
    bridge_config,
) -> tuple[torch.Tensor, dict[str, float]]:
    bsz = x_0.shape[0]
    if noise_t_max > 0:
        if isinstance(sde, FlowMatchingODE):
            x_dec = x_0 + noise_t_max * torch.randn_like(x_0)
        else:
            t_noise = torch.rand(bsz, device=x_0.device) * noise_t_max
            sigma = torch.sqrt(sde.C(
                start=torch.zeros_like(t_noise),
                t_a=t_noise,
                t_b=t_noise,
            )).view(-1, 1, 1, 1)
            x_dec = x_0 + sigma * torch.randn_like(x_0)
    else:
        x_dec = x_0

    token_flat = bridge_to_token_flat(x_dec, layout=token_layout, config=bridge_config)
    logits = decoder(token_flat)
    valid = token_mask.to(device=x_0.device, dtype=torch.bool)
    if pad_zero_prob > 0:
        pad_train = (~valid) & (
            torch.rand(valid.shape, device=x_0.device) < pad_zero_prob
        )
        loss_mask = valid | pad_train
    else:
        loss_mask = valid
    if not torch.any(loss_mask):
        return logits.sum() * 0.0, {"token_decoder/acc": 0.0, "token_decoder/n": 0.0}

    targets = token_ids.to(device=x_0.device, dtype=torch.long)
    loss = torch.nn.functional.cross_entropy(logits[loss_mask], targets[loss_mask])
    with torch.no_grad():
        pred = logits.argmax(dim=-1)
        valid_n = valid.sum().clamp_min(1)
        acc = ((pred == targets) & valid).sum().float() / valid_n

        valid_n_including_loss_mask = loss_mask.sum().clamp_min(1)
        acc_including_loss_mask = ((pred == targets) & loss_mask).sum().float() / valid_n_including_loss_mask
    return loss, {
        "token_decoder/acc": float(acc.item()),
        "token_decoder/n": float(valid.sum().item()),
        "token_decoder/acc_including_loss_mask": float(acc_including_loss_mask.item()),
        "token_decoder/n_including_loss_mask": float(valid_n_including_loss_mask.item()),
    }


def train_step(
    *,
    model: torch.nn.Module,             # DDP-wrapped
    ema: torch.nn.Module,                # un-wrapped EMA model
    alt_emas: dict[float, torch.nn.Module] | None = None,  # sanity-only EMAs
    sde,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.LRScheduler,
    batch: dict,
    device: torch.device,
    train_forward: bool,
    train_reverse: bool,
    eps: float,
    grad_clip: float,
    ema_decay: float,
    unconditional_percent: float,
    use_token_text_bridge: bool,
    token_layout: str,
    x0_cond_source: str,
    token_decoder: SharedTokenDecoder | None,
    token_decoder_optimizer: optim.Optimizer | None,
    token_decoder_noise_t_max: float,
    token_decoder_pad_zero_prob: float,
    bridge_config,
    precond: EDMPrecond | None = None,
    use_repa_text: bool = False,
    repa_text_lambda: float = 0.0,
    use_repa_image: bool = False,
    repa_image_lambda: float = 0.0,
    repa_phase: str = "equal",
    time_sampler: str = "uniform",
    time_sampler_logit_normal_mean: float = 0.0,
    time_sampler_logit_normal_std: float = 1.0,
) -> dict[str, float]:
    model.train()
    if token_decoder is not None:
        token_decoder.train()
    x_0, x_1, y, x_cond_0, x_cond_1 = prepare_bridge_batch(
        batch,
        device,
        use_token_text_bridge=use_token_text_bridge,
        token_layout=token_layout,
        x0_cond_source=x0_cond_source,
        config=bridge_config,
    )
    if time_sampler == "uniform":
        t = uniform_timestep_sample(batch_size=x_0.shape[0], device=device, eps=eps)
    elif time_sampler == "logit_normal":
        t = logit_normal_timestep_sample(
            P_mean=time_sampler_logit_normal_mean,
            P_std=time_sampler_logit_normal_std,
            num_samples=x_0.shape[0],
            device=device,
            eps=eps,
        )
    else:
        raise ValueError(f"unknown time sampler: {time_sampler}")
    # interpolates between PHYSICAL x_0 and x_1 based on t (conditioning signal may be different)
    is_flow_matching = isinstance(sde, FlowMatchingODE)
    if is_flow_matching:
        x_t = sample_flow_matching_x_t(x_0=x_0, x_1=x_1, t=t)
    else:
        x_t = sample_p_base_x_t_cond_x_0_x_1(
            sde=sde, x_0=x_0, x_1=x_1, t=t
        )
    assert not torch.isnan(x_t).any()

    # REPA targets. Text: flat global text embedding (truncated + normalized
    # inside repa_text_loss). Image: DINOv2 patch tokens + presence mask (only
    # a subset of rows were re-encoded). Both available regardless of bridge
    # mode / x0_cond_source.
    use_repa = use_repa_text or use_repa_image
    repa_text_target = (
        batch["text_emb"].to(device, non_blocking=True).float()
        if use_repa_text else None
    )
    # Per-sample presence: rows the sidecar's --percent prefix didn't cover get
    # weight 0 so their zero-filled targets don't pollute the loss.
    repa_text_present = (
        batch["text_emb_present"].to(device, non_blocking=True).float()
        if use_repa_text else None
    )
    if use_repa_image:
        dino_target = batch["dino_emb"].to(device, non_blocking=True).float()
        dino_present = batch["dino_present"].to(device, non_blocking=True).bool()

    # Compute the forward (text->image) and reverse (image->text) losses in a
    # SINGLE graph and do ONE backward / optimizer step. This (a) exercises
    # both conditioning encoders in every backward -> no unused parameters
    # (so DDP runs with find_unused_parameters=False), and (b) collapses the
    # two per-step all-reduces into one, halving the cross-rank communication
    # volume that stragglers amplify on multi-node.
    logs: dict[str, float] = {}
    for tag, tensor in (("x_0", x_0), ("x_1", x_1), ("x_cond_0", x_cond_0)):
        if tensor is None:
            continue
        logs[f"data/{tag}_norm"] = tensor.flatten(1).norm(dim=1).mean().item()
        logs[f"data/{tag}_mean"] = tensor.mean().item()
        logs[f"data/{tag}_sq_mean"] = tensor.pow(2).mean().item()
    logs["data/sigma01"] = (x_0 * x_1).mean().item()
    directions = []
    if train_forward:
        directions.append(False)
    if train_reverse:
        directions.append(True)
    if not directions:
        raise ValueError("At least one training direction must be enabled.")
    per_dir_loss: dict[str, torch.Tensor] = {}
    optimizer.zero_grad(set_to_none=True)
    if token_decoder_optimizer is not None:
        token_decoder_optimizer.zero_grad(set_to_none=True)
    with autocast(device_type="cuda", dtype=torch.bfloat16):
        total_loss = x_t.new_zeros(())
        for reverse in directions:
            cond_mask = torch.rand(
                x_0.shape[0], device=device,
            ) > unconditional_percent
            if is_flow_matching:
                out = flow_matching_loss(
                    model=model,
                    x_t=x_t, x_1=x_1, x_0=x_0,
                    t=t, y=y, reverse=reverse,
                    cond_mask=cond_mask,
                    x_cond_0=x_cond_0 if (not reverse) else None,
                    x_cond_1=x_cond_1 if reverse else None,
                    return_repa=use_repa,
                )
            elif precond is not None:
                raise NotImplementedError("EDM-style preconditioning is not supported yet.")
                out = edm_dsm_loss(
                    model=model, precond=precond,
                    x_t=x_t, x_1=x_1, x_0=x_0,
                    t=t, y=y, reverse=reverse,
                    cond_mask=cond_mask,
                    x_cond_0=x_cond_0 if (not reverse) else None,
                    x_cond_1=x_cond_1 if (reverse) else None,
                    return_repa=use_repa,
                )
            else:
                out = dsm_loss(
                    model=model, sde=sde,
                    x_t=x_t, x_1=x_1, x_0=x_0,
                    t=t, y=y, reverse=reverse,
                    cond_mask=cond_mask,
                    x_cond_0=x_cond_0 if (not reverse) else None,
                    x_cond_1=x_cond_1 if (reverse) else None,
                    return_repa=use_repa,
                )
            dir_tag = "reverse" if reverse else "forward"
            loss, repa = out if use_repa else (out, None)
            if use_repa:
                # Phase split only matters when both flavours are on; otherwise
                # each kind keeps its full per-sample weight (=1).
                if use_repa_text and use_repa_image:
                    w_text, w_image = repa_phase_weights(t, repa_phase)
                else:
                    w_text = w_image = torch.ones_like(t)
                if use_repa_text:
                    r_loss = repa_text_loss(repa["text"], repa_text_target, w_text * repa_text_present)
                    total_loss = total_loss + repa_text_lambda * r_loss
                    logs[f"loss/repa_text_{dir_tag}"] = r_loss.item()
                    logs["repa/lambda_text"] = repa_text_lambda
                if use_repa_image:
                    r_loss, n = repa_image_loss(
                        repa["image"], dino_target, dino_present, w_image
                    )
                    total_loss = total_loss + repa_image_lambda * r_loss
                    logs[f"loss/repa_image_{dir_tag}"] = r_loss.item()
                    logs[f"repa/n_image_{dir_tag}"] = float(n.item())
                    logs["repa/lambda_image"] = repa_image_lambda
            # Now handle logging and accumulation of the main loss
            per_dir_loss[dir_tag] = loss
            total_loss = total_loss + loss
        if use_token_text_bridge and not bridge_config.text_as_noise:
            assert token_decoder is not None
            dec_loss, dec_logs = token_decoder_loss(
                decoder=token_decoder,
                sde=sde,
                x_0=x_0,
                token_ids=batch["text_token_ids"],
                token_mask=batch["text_token_mask"],
                noise_t_max=token_decoder_noise_t_max,
                pad_zero_prob=token_decoder_pad_zero_prob,
                token_layout=token_layout,
                bridge_config=bridge_config,
            )
            logs["loss/token_decoder"] = dec_loss.item()
            logs.update(dec_logs)

    total_loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
    optimizer.step()

    # Step the LR scheduler once per training iteration so --warmup-steps
    # lines up with --steps.
    scheduler.step()

    if token_decoder is not None:
        dec_loss.backward()
        token_decoder_grad_norm = torch.nn.utils.clip_grad_norm_(
            token_decoder.parameters(), max_norm=grad_clip
        )
        logs["token_decoder/grad_norm"] = token_decoder_grad_norm.item()
        token_decoder_optimizer.step()
        # NOTE: token decoder optimizer has no scheduler, so we don't need to step it

    inner = model.module if isinstance(model, DDP) else model
    update_ema(ema, inner, decay=ema_decay)
    if alt_emas:
        for alt_decay, alt_ema in alt_emas.items():
            update_ema(alt_ema, inner, decay=alt_decay)

    for tag, loss in per_dir_loss.items():
        logs[f"loss/{tag}"] = loss.item()
    logs["grad_norm"] = grad_norm.item()
    return logs


# ---------------------------------------------------------------------------
# DataLoader cycling
# ---------------------------------------------------------------------------

def _worker_init_fn(_worker_id):
    info = torch.utils.data.get_worker_info()
    seed = int(info.seed)
    def _reset(ds):
        if isinstance(ds, ConcatDataset):
            for c in ds.datasets:
                _reset(c)
        elif hasattr(ds, "_reset_worker_state"):
            ds._reset_worker_state(seed)
        else:
            raise ValueError(f"Unknown dataset type: {type(ds)}")
            if hasattr(ds, "_handles"):
                ds._handles = {}
            if hasattr(ds, "_captions_cache"):
                ds._captions_cache = {}
    _reset(info.dataset)


def cycle_loader(loader: DataLoader, sampler: DistributedSampler) -> Iterator[dict]:
    """Yield batches forever, calling ``sampler.set_epoch`` each epoch."""
    epoch = 0
    while True:
        sampler.set_epoch(epoch)
        for batch in loader:
            yield batch
        epoch += 1


# ---------------------------------------------------------------------------
# Cross-rank scalar averaging
# ---------------------------------------------------------------------------

def _avg_across_ranks(value: float, device: torch.device) -> float:
    t = torch.tensor([value], device=device, dtype=torch.float32)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return (t.item() / dist.get_world_size())


def _gather_across_ranks(value: float, device: torch.device) -> list[float]:
    """All-gather a scalar from every rank; returns a list indexed by rank.

    Used for straggler diagnostics: a single slow node shows up as one entry
    far above the others (max >> mean).
    """
    t = torch.tensor([value], device=device, dtype=torch.float32)
    gathered = [torch.zeros_like(t) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, t)
    return [float(x.item()) for x in gathered]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()

    # Model
    parser.add_argument("--model", type=str, default="DiTXA-B/2",
                        choices=list(DiT_models.keys()))

    # Data
    parser.add_argument("--data-root", type=str, required=True,
                        help="Root containing latents_*.memmap, text_*.memmap, config.json.")
    parser.add_argument(
        "--bridge-preset", type=str, default="auto",
        choices=["auto", *BRIDGE_RUNTIME_PRESETS.keys()],
        help="Bridge geometry preset. auto reads dataset metadata; old data defaults to sd.",
    )
    parser.add_argument("--val-fraction", type=float, default=0.005)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument(
        "--use-token-text-bridge", action="store_true",
        help="Use token-by-token text sidecars as the t=0 endpoint. "
             "Token sidecars are read from each dataset's own root.",
    )
    parser.add_argument(
        "--token-layout", type=str, default="row_major", choices=list(TOKEN_LAYOUTS),
        help="How token embeddings are packed into DiT patches.",
    )
    parser.add_argument(
        "--x0-cond-source", type=str, default="x0",
        choices=["global_text", "x0"],
        help="Forward conditioning source: global text embedding, or the actual x_0 endpoint.",
    )
    parser.add_argument(
        "--text-token-scale", type=float, default=None,
        help="Override TEXT_TOKEN_SCALE (default sqrt(TOKEN_EMB_DIM)).",
    )
    parser.add_argument(
        "--forward-cond-scale", type=float, default=1.0,
        help="Multiplier applied to x_cond in the forward direction inside the "
             "model (e.g. 64 = sqrt(4096) to bring the unit-norm global text "
             "embedding to unit per-coordinate scale).",
    )

    # REPA (REPresentation Alignment). Two independent flavours, each with its
    # own enable flag + hyperparameters + projection heads:
    #   text : pooled intermediate feature -> (MRL-truncated) global text emb.
    #   image: per-token intermediate feature -> DINOv2 patch tokens (sidecar).
    # When both are on, --repa-phase splits their per-sample strength over t.
    parser.add_argument("--repa-text", action="store_true",
                        help="Enable text REPA (align pooled feature to global text embedding).")
    parser.add_argument("--repa-text-lambda", type=float, default=0.5,
                        help="Weight on the text REPA loss (per direction).")
    parser.add_argument("--repa-text-warmup-steps", type=int, default=0,
                        help="Linearly ramp repa-text-lambda from 0 over this many steps.")
    parser.add_argument("--repa-text-dim", type=int, default=1024,
                        help="Projection / truncation dimension for the text REPA target.")
    parser.add_argument("--repa-text-layer", type=int, default=None,
                        help="Block index to tap for text REPA (default: depth // 3).")
    parser.add_argument("--repa-image", action="store_true",
                        help="Enable image REPA (align per-token feature to DINOv2 patch tokens). Requires --dino-dir.")
    parser.add_argument("--repa-image-lambda", type=float, default=0.5,
                        help="Weight on the image REPA loss (per direction).")
    parser.add_argument("--repa-image-warmup-steps", type=int, default=0,
                        help="Linearly ramp repa-image-lambda from 0 over this many steps.")
    parser.add_argument("--repa-image-layer", type=int, default=None,
                        help="Block index to tap for image REPA (default: depth // 3).")
    parser.add_argument("--repa-phase", type=str, default="equal",
                        choices=["equal", "image_up", "text_up"],
                        help="When both REPA flavours are on, how to split strength over t.")
    parser.add_argument("--dino-dir", type=str, default=None,
                        help="DINOv2 sidecar dir (encode_dino_features.py output) for image REPA.")
    parser.add_argument("--gtext-dir", type=str, default=None,
                        help="Global text-embedding sidecar dir (encode_global_text.py output) for text REPA.")

    # Train
    parser.add_argument("--steps", type=int, default=1_000_000)
    parser.add_argument("--global-batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=10_000)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.999)

    parser.add_argument("--eps", type=float, default=9.9e-4,
                        help="Margin for sampling t in (eps, 1-eps).")
    parser.add_argument("--time-sampler", type=str, default="uniform", choices=["uniform", "logit_normal"],
                        help="Time sampler to use for sampling t.")
    parser.add_argument("--time-sampler-logit-normal-mean", type=float, default=0.0,
                        help="Mean for logit_normal time sampler.")
    parser.add_argument("--time-sampler-logit-normal-std", type=float, default=1.0,
                        help="Std for logit_normal time sampler.")

    ### ABLATION OPTIONS
    parser.add_argument("--no-reverse", action="store_true",
                        help="Train forward direction only.")
    parser.add_argument("--no-forward", action="store_true",
                        help="Train reverse direction only.")
    parser.add_argument("--text-as-noise", action="store_true",
                        help="Train with text end of bridge replaced with noise (like traditional conditional diffusion models).")
    parser.add_argument("--image-as-noise", action="store_true",
                        help="Train with image end of bridge replaced with noise (like traditional conditional diffusion models).")

    parser.add_argument("--num-classes", type=int, default=1)
    parser.add_argument("--unconditional-percent", type=float, default=0.1,
                        help="Percentage of loss calculations that are unconditional.")
    parser.add_argument(
        "--force-unconditional", action="store_true",
        help="Train and sample one unconditional forward FlowMatchingODE field.",
    )
    parser.add_argument("--prompt-kind-dropout", type=float, default=0.2,
                        help="Classifier-free dropout probability for prompt-kind labels.")
    parser.add_argument("--token-decoder-lr", type=float, default=5e-3)
    parser.add_argument("--token-decoder-weight-decay", type=float, default=0.0)
    parser.add_argument("--token-decoder-hidden-dim", type=int, default=128)
    parser.add_argument("--token-decoder-noise-t-max", type=float, default=0.01)
    parser.add_argument("--token-decoder-pad-zero-prob", type=float, default=1e-5)
    parser.add_argument("--eval-cfg-scales", type=float, nargs="+", default=[0.0, 1.0, 2.0],
                        help="CFG scale for evaluation.")
    # EMA
    parser.add_argument("--ema-decay", type=float, default=0.9995)
    parser.add_argument(
        "--alt-ema-decays", type=float, nargs="+", default=[],
        help="Additional EMA decay rates to maintain alongside the primary "
             "--ema-decay. These alternate EMAs are updated every step and "
             "checkpointed, but are NOT used during evaluation (kept for "
             "sanity / later inspection only).",
    )

    # SDE
    parser.add_argument(
        "--sde", type=str, default="uniform",
        choices=["uniform", "periodic", "cosine_decay", "flow_matching"],
    )
    parser.add_argument("--periodic_sde_alpha", type=float, default=1.0,
                        help="Alpha for PeriodicVolatilitySDE.")
    parser.add_argument("--periodic_sde_k", type=float, default=1.0,
                        help="K for PeriodicVolatilitySDE.")
    parser.add_argument("--periodic_sde_eps", type=float, default=0.05,
                        help="Eps for PeriodicVolatilitySDE.")
    parser.add_argument("--K", type=float, default=1.0,
                        help="Diffusion magnitude for UniformVolatilitySDE.")

    # EDM-style preconditioning (Appendix E). Stats are per-coordinate SECOND
    # MOMENTS of the endpoints (no mean subtraction); read them off the
    # data/x_0_sq_mean, data/x_1_sq_mean, data/sigma01 logs of a prior run.
    parser.add_argument("--edm-precond", action="store_true",
                        help="Use EDM-style x-prediction preconditioning instead of the score loss.")
    parser.add_argument("--sigma0-sq", type=float, default=1.0,
                        help="Per-coordinate second moment of x_0 (text endpoint).")
    parser.add_argument("--sigma1-sq", type=float, default=1.0,
                        help="Per-coordinate second moment of x_1 (image endpoint).")
    parser.add_argument("--sigma01", type=float, default=0.0,
                        help="Per-coordinate cross moment E[x_0 * x_1].")

    # Logging / checkpointing
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--ckpt-every", type=int, default=20_000)
    parser.add_argument("--eval-every", type=int, default=20_000)
    parser.add_argument("--min-throughput", type=float, default=0.0,
                        help="Throughput watchdog (0 disables). If train steps_per_sec "
                             "stays below this for --min-throughput-windows consecutive "
                             "log windows (eval/ckpt windows excluded), rank 0 exits 1 so "
                             "the launch retry loop requeues onto healthy nodes.")
    parser.add_argument("--min-throughput-windows", type=int, default=3,
                        help="Consecutive sub-threshold log windows before the watchdog fires.")
    parser.add_argument("--out-dir", type=str, default="/data/text_to_image")
    parser.add_argument("--wandb-project", type=str, default="bib-text2image")
    parser.add_argument("--wandb-name", type=str, default=None)
    parser.add_argument("--wandb-mode", type=str, default="online",
                        choices=["online", "offline", "disabled"])

    # Eval
    parser.add_argument("--eval-batch-size", type=int, default=64,
                        help="Val-loss batch size; the first --eval-n-decode samples are visualized.")
    parser.add_argument("--eval-n-decode", type=int, default=4)
    parser.add_argument("--eval-grid-nrow", type=int, default=2,
                        help="Video/PNG grid columns; 0 chooses ceil(sqrt(eval-n-decode)).")
    parser.add_argument("--eval-num-sde-steps", type=int, default=500)
    parser.add_argument("--eval-ode", action="store_true",
                        help="Use ODE for evaluation.")
    parser.add_argument("--eval-num-ode-steps", type=int, default=100)
    parser.add_argument("--eval-decode-every-k", type=int, default=20)
    parser.add_argument("--eval-fps", type=int, default=10)
    parser.add_argument("--vae-ckpt", type=str, default=None)

    # FID (text -> image). DDP-sharded; each rank generates a slice of the
    # samples, ``torchmetrics`` all-reduces the running stats inside compute.
    parser.add_argument("--eval-fid-num-samples", type=int, default=2048,
                        help="Total number of generated samples used for FID "
                             "per eval call (sharded across ranks). Set to 0 "
                             "to skip FID.")
    parser.add_argument("--eval-fid-batch-size", type=int, default=16,
                        help="Per-rank batch size used when generating FID "
                             "samples. Lower if you OOM on the VAE decode.")
    parser.add_argument("--eval-fid-feature", type=int, default=2048,
                        choices=[64, 192, 768, 2048],
                        help="Inception feature dim used by torchmetrics FID.")

    parser.add_argument("--eval-text-decode-num-samples", "--eval-i2t-num-samples",
                        dest="eval_text_decode_num_samples", type=int, default=512,
                        help="Total number of samples for decoded text eval. Set to 0 to skip.")
    parser.add_argument("--eval-text-decode-batch-size", "--eval-i2t-batch-size",
                        dest="eval_text_decode_batch_size", type=int, default=16)
    parser.add_argument("--no-eval-cider", action="store_true",
                        help="Skip CIDEr even if pycocoevalcap is installed.")
    parser.add_argument("--no-eval-clipscore", action="store_true",
                        help="Skip CLIPScore (both i2t and t2i directions) even if installed.")
    parser.add_argument("--no-eval-genppl", action="store_true",
                        help="Skip generative perplexity of decoded captions (Qwen3 oracle).")
    parser.add_argument("--eval-genppl-model", type=str, default="Qwen/Qwen3-1.7B",
                        help="Oracle causal LM for generative perplexity (must be in the HF cache on offline nodes).")
    parser.add_argument("--eval-genppl-batch-size", type=int, default=16,
                        help="Batch size for the generative-perplexity oracle forward passes.")
    parser.add_argument("--eval-text-decode-include-padding-in-accuracy", action="store_true",
                        help="Include padding in text decode accuracy calculation.")
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip training: resume a checkpoint, run one eval pass, then exit.")

    # Resume
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--auto-resume", action="store_true",
                        help="If --resume is unset, resume from the newest "
                             "step_*.pt under --out-dir (for SLURM requeue).")
    parser.add_argument("--seed", type=int, default=0)

    # Cache
    parser.add_argument("--cache-dir", type=str, default="/pscratch/sd/g/gabeguo/cache_sub")

    args = parser.parse_args()
    if args.no_forward and args.no_reverse:
        parser.error("At least one direction must be enabled (--no-forward/--no-reverse).")
    if not 0.0 <= args.unconditional_percent <= 1.0:
        parser.error("--unconditional-percent must be between 0 and 1.")
    if args.force_unconditional and args.sde != "flow_matching":
        parser.error("--force-unconditional is only supported with --sde flow_matching.")
    if args.sde == "flow_matching":
        assert args.force_unconditional, "I realized that the conditional path is degenerate and unnecessary, since flow matching is deterministic."
        if args.text_as_noise or args.image_as_noise:
            parser.error("Flow matching currently supports only data-to-data training.")
        if args.edm_precond:
            parser.error("--edm-precond is incompatible with --sde flow_matching.")
        if (
            abs(args.unconditional_percent - 1.0) <= 1e-3
            and not args.force_unconditional
        ):
            parser.error(
                "--unconditional-percent within 1e-3 of 1 requires "
                "--force-unconditional."
            )
        if args.force_unconditional:
            if args.no_forward:
                parser.error(
                    "--force-unconditional flow matching requires forward training."
                )
            print("forcing --no-reverse=True and --unconditional-percent=1.0 because --force-unconditional is True")
            args.no_reverse = True
            args.unconditional_percent = 1.0
    can_infer_reverse = (not args.no_reverse) or (args.force_unconditional and args.sde == "flow_matching")
    print(f"can_infer_reverse: {can_infer_reverse}")
    if args.x0_cond_source != "x0":
        raise ValueError("global_text conditioning is deprecated; use --x0-cond-source x0.")
    if not args.use_token_text_bridge:
        raise ValueError("--use-token-text-bridge is now required.")
    if args.forward_cond_scale != 1.0 and args.x0_cond_source != "global_text":
        raise ValueError(
            "--forward-cond-scale != 1 is only meant for --x0-cond-source "
            "global_text (the x0 token grid is already unit-scale via "
            "TEXT_TOKEN_SCALE)."
        )
    bridge_runtime = bridge_config_from_manifest(args.data_root, preset=args.bridge_preset)
    print(f"bridge_runtime: {bridge_runtime}")
    bridge_config = bridge_runtime.bridge
    
    # This is for the ablation against noise-to-data: sets one end of the bridge to noise, but still keeps the conditioning signal.
    if args.text_as_noise:
        assert args.no_reverse, "Inverse is poorly defined when text is noise."
        bridge_config = replace(bridge_config, text_as_noise=True)
    if args.image_as_noise:
        assert args.no_forward, "Inverse is poorly defined when image is noise."
        bridge_config = replace(bridge_config, image_as_noise=True)
    assert not (args.text_as_noise and args.image_as_noise), "--text-as-noise and --image-as-noise cannot be used together."
    # Wire the changes back through, BEFORE this gets passed anywhere
    bridge_runtime = replace(bridge_runtime, bridge=bridge_config)

    if args.repa_text:
        assert args.use_token_text_bridge, "--repa-text requires --use-token-text-bridge (x_0 token initial condition)."
        assert args.x0_cond_source == "x0", "--repa-text requires --x0-cond-source x0 (x_0 token initial condition). Otherwise, it can just cheat for REPA."
        if args.gtext_dir is None:
            raise ValueError("--repa-text requires --gtext-dir (encode_global_text.py output).")
        gcfg = json.loads((Path(args.gtext_dir) / "gtext_config.json").read_text())["config"]
        global_text_dim = int(gcfg["flat_dim"])
        assert 0 < args.repa_text_dim <= global_text_dim, f"--repa-text-dim must be in (0, {global_text_dim}], got {args.repa_text_dim}."
    # Image REPA shape comes from the DINO sidecar config (the model's image
    # head output dim must equal the stored token_dim).
    repa_image_dim = None
    dino_shape = None
    if args.repa_image:
        if args.dino_dir is None:
            raise ValueError("--repa-image requires --dino-dir.")
        dcfg = json.loads((Path(args.dino_dir) / "dino_config.json").read_text())["config"]
        repa_image_dim = int(dcfg["token_dim"])
        dino_shape = (int(dcfg["num_tokens"]), int(dcfg["token_dim"]))

    # Resolve the token scale once and thread it through the dataset + eval so
    # the apply (dataset) and unapply (eval stop-detection) sides stay in sync.
    token_scale = math.sqrt(bridge_config.token_emb_dim) if args.text_token_scale is None else args.text_token_scale
    assert token_scale == bridge_runtime.token_scale, f"token_scale {token_scale} does not match bridge_runtime.token_scale {bridge_runtime.token_scale}"

    # A100-friendly TF32
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # ---- DDP setup
    rank, world_size, local_rank, eval_pg = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")
    assert args.global_batch_size % world_size == 0, \
        f"global-batch-size {args.global_batch_size} not divisible by world_size {world_size}"
    local_batch = args.global_batch_size // world_size
    torch.manual_seed(args.seed * world_size + rank)
    print(f"rank: {rank} out of {world_size} world_size")

    # ---- Output dir + wandb (rank 0 only)
    if rank == 0:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path(args.out_dir) / timestamp
        out_dir.mkdir(parents=True, exist_ok=True)
        ckpt_dir = out_dir / "checkpoints"; ckpt_dir.mkdir(exist_ok=True)
        eval_dir = out_dir / "eval"; eval_dir.mkdir(exist_ok=True)
        # Stable id (SLURM job id) so the launch-script retry loop resumes the
        # same run instead of fragmenting across one run per attempt.
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_name or timestamp,
            id=os.environ.get("SLURM_JOB_ID") or timestamp,
            resume="allow",
            config=vars(args),
            dir=str(out_dir),
            mode=args.wandb_mode,
        )
        print(f"[rank 0] writing to {out_dir}")
    else:
        out_dir = None
        ckpt_dir = None
        eval_dir = None

    # Every rank needs the same score-output path because score collection is a
    # distributed collective, even though only rank 0 writes the resulting file.
    eval_dir_holder = [str(eval_dir) if rank == 0 else None]
    dist.broadcast_object_list(eval_dir_holder, src=0)
    clip_scores_dir = Path(eval_dir_holder[0])
    dist.barrier()

    # ---- Model + EMA
    assert "XA" in args.model, "Cross-attention required for text<->image bridge"
    model = DiT_models[args.model](
        input_size=bridge_config.height,
        in_channels=bridge_config.channels,
        num_classes=PROMPT_NUM_CLASSES if args.use_token_text_bridge else args.num_classes,
        class_dropout_prob=args.prompt_kind_dropout if args.use_token_text_bridge else 0.0,
        forward_cond_scale=args.forward_cond_scale,
        use_repa_text=args.repa_text,
        repa_text_dim=args.repa_text_dim,
        repa_text_layer=args.repa_text_layer,
        use_repa_image=args.repa_image,
        repa_image_dim=repa_image_dim,
        repa_image_layer=args.repa_image_layer,
        repa_train_reverse=not args.no_reverse,
    ).to(device)
    # Keep DDP's find_unused_parameters=False valid when a direction is disabled.
    if args.no_forward:
        requires_grad(model.cond_embedder_forward, False)
        if hasattr(model, "repa_text_head_forward"):
            requires_grad(model.repa_text_head_forward, False)
        if hasattr(model, "repa_image_head_forward"):
            requires_grad(model.repa_image_head_forward, False)
    if args.no_reverse:
        requires_grad(model.cond_embedder_reverse, False)
        if hasattr(model, "repa_text_head_reverse"):
            requires_grad(model.repa_text_head_reverse, False)
        if hasattr(model, "repa_image_head_reverse"):
            requires_grad(model.repa_image_head_reverse, False)
    assert bridge_config.patch_size == model.patch_size, f"bridge_config.patch_size {bridge_config.patch_size} does not match model.patch_size {model.patch_size}"

    # Deepcopy BEFORE DDP wrapping so the EMA has the same buffer/parameter
    # tree shape as the un-wrapped module (no .module prefix).
    ema = deepcopy(model).to(device)
    requires_grad(ema, False)
    ema.eval()

    # Optional alternate-decay EMAs. Kept purely for sanity / later inspection:
    # they are updated every step and saved into every checkpoint, but are never
    # swapped in during the eval loop. Keyed by their decay rate.
    alt_emas: dict[float, torch.nn.Module] = {}
    for alt_decay in args.alt_ema_decays:
        if alt_decay == args.ema_decay:
            continue  # already covered by the primary EMA
        alt = deepcopy(model).to(device)
        requires_grad(alt, False)
        alt.eval()
        alt_emas[alt_decay] = alt
    if rank == 0 and alt_emas:
        print(f"Maintaining alternate EMAs (sanity only) at decays: "
              f"{sorted(alt_emas)}")

    # Both conditioning encoders are now exercised in every backward (the
    # forward + reverse losses are summed into a single backward in
    # train_step), so there are no unused parameters and we can disable the
    # expensive find_unused_parameters autograd-graph traversal.
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
    if rank == 0:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {n_params:,}")

    # ---- Optimizer + LR schedule (linear warmup, no decay afterwards)
    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr,
        betas=(args.adam_beta1, args.adam_beta2), weight_decay=0.0,
    )
    scheduler = optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1e-3,
        end_factor=1.0,
        total_iters=max(1, args.warmup_steps),
    )
    token_decoder = None
    token_decoder_optimizer = None
    token_tokenizer = None
    token_model_id = None
    token_pad_id = None
    global_text_model_id = None
    if args.use_token_text_bridge and not bridge_config.text_as_noise:
        # token decoder not needed when text is noise
        from transformers import AutoTokenizer
        token_cfg_path = Path(args.data_root) / "token_embed_config.json"
        token_cfg = json.loads(token_cfg_path.read_text())["config"]
        token_model_id = token_cfg["text_model"]
        source_cfg_path = Path(args.data_root) / "config.json"
        source_cfg = json.loads(source_cfg_path.read_text())["config"]
        global_text_model_id = source_cfg["text_model"]
        token_tokenizer = AutoTokenizer.from_pretrained(token_model_id)
        token_pad_id = token_tokenizer.pad_token_id
        if token_pad_id is None:
            print("token_pad_id is None, using eos_token_id")
            token_pad_id = token_tokenizer.eos_token_id
        assert token_pad_id is not None, (
            "tokenizer exposes neither pad_token_id nor eos_token_id; cannot "
            "remap padding token ids."
        )
        token_pad_id = int(token_pad_id)
        assert 0 <= token_pad_id < len(token_tokenizer)
        if rank == 0:
            pad_tok = token_tokenizer.convert_ids_to_tokens(token_pad_id)
            print(
                f"[token bridge] masked token-id padding -> token_pad_id="
                f"{token_pad_id} ({pad_tok!r})"
            )
        token_decoder = SharedTokenDecoder(
            vocab_size=len(token_tokenizer),
            hidden_dim=args.token_decoder_hidden_dim,
            token_seq_len=bridge_config.token_seq_len,
            token_emb_dim=bridge_config.token_emb_dim,
        ).to(device)
        token_decoder_optimizer = optim.AdamW(
            token_decoder.parameters(),
            lr=args.token_decoder_lr,
            weight_decay=args.token_decoder_weight_decay,
        )

    # ---- SDE (score network is the DDP-wrapped train model; we swap to EMA inside eval)
    if args.sde == "uniform":
        sde = UniformVolatilitySDE(A=0, K=args.K, score_network=model)
    elif args.sde == "periodic":
        sde = PeriodicVolatilitySDE(alpha=args.periodic_sde_alpha, k=args.periodic_sde_k, eps=args.periodic_sde_eps, score_network=model)
    elif args.sde == "cosine_decay":
        sde = CosineDecayingVolatilitySDE(alpha=args.periodic_sde_alpha, eps=args.periodic_sde_eps, score_network=model)
    elif args.sde == "flow_matching":
        sde = FlowMatchingODE(
            score_network=model,
            force_unconditional=args.force_unconditional,
        )
    else:
        raise ValueError(f"unknown sde: {args.sde}")

    precond = None
    if args.edm_precond:
        precond = EDMPrecond(
            sde, sigma0_sq=args.sigma0_sq, sigma1_sq=args.sigma1_sq, sigma01=args.sigma01,
        )
        # Sampling consumes scores; wrap the x-prediction net so SDE.dX_t is unchanged.
        sde.score_network = EDMScoreWrapper(model, precond)

    # ---- Dataset / loaders
    # Shared dataset kwargs reused across train / val / extra datasets so they
    # all see consistent configuration.
    ds_kwargs = {
        "token_pad_id": token_pad_id,
        "token_scale": token_scale,
        "config": bridge_config,
        "latent_scale": bridge_runtime.latent_scale,
        "latent_shift": bridge_runtime.latent_shift,
    }
    # Every row now emits an identical set of keys, so PyTorch's default
    # collate handles batching.
    collate_fn = None

    # Probe full dataset size for the train/val index split. The probe object
    # is discarded immediately; the per-shard memmaps it lazy-opened in this
    # process get garbage-collected.
    probe = CommonCatalogLatentDataset(
        args.data_root, cast_dtype=torch.float32, **ds_kwargs,
    )
    n_total = len(probe)
    if rank == 0:
        print(f"Dataset size: {n_total:,}")
    train_idx, val_idx = random_split_indices(
        n_total, val_fraction=args.val_fraction, seed=args.seed,
    )
    del probe

    train_ds = CommonCatalogLatentDataset(
        args.data_root, cast_dtype=torch.float32, indices=train_idx,
        dino_dir=args.dino_dir if args.repa_image else None,
        gtext_dir=args.gtext_dir if args.repa_text else None,
        **ds_kwargs,
    )
    val_ds = CommonCatalogLatentDataset(
        args.data_root, cast_dtype=torch.float32, indices=val_idx,
        return_caption=True, **ds_kwargs,
    )

    if rank == 0:
        print(f"Train: {len(train_ds):,}  Val: {len(val_ds):,}")

    # Fixed, identical-across-ranks index lists for the distributed metrics.
    # Taking the first N positions into val_ds keeps the eval set stable
    # across training runs (val_idx is already a deterministic shuffle of
    # the full dataset under args.seed).
    fid_eval_indices = list(range(min(args.eval_fid_num_samples, len(val_ds))))
    text_decode_eval_indices = list(range(min(args.eval_text_decode_num_samples, len(val_ds))))
    if rank == 0:
        if args.eval_fid_num_samples > 0:
            print(f"FID eval samples: {len(fid_eval_indices)} "
                  f"(requested {args.eval_fid_num_samples})")
        if args.eval_text_decode_num_samples > 0:
            print(f"Text decode eval samples: {len(text_decode_eval_indices)} "
                  f"(requested {args.eval_text_decode_num_samples})")

    train_sampler = DistributedSampler(
        train_ds, num_replicas=world_size, rank=rank,
        shuffle=True, seed=args.seed, drop_last=True,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=local_batch,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
        worker_init_fn=_worker_init_fn,
        collate_fn=collate_fn,
        # Deepen the per-worker prefetch buffer so an occasional slow Lustre
        # read is hidden behind already-fetched batches. The per-step max over
        # ranks means even rare read stalls hurt much more on multi-node, so a
        # larger buffer directly reduces stall probability.
        prefetch_factor=4 if args.num_workers > 0 else None,
    )

    # Val batch: rank 0 only. Built once with num_workers=0 so we don't leave
    # persistent worker processes hanging on a loader we only consume once.
    if rank == 0:
        assert args.eval_n_decode <= args.eval_batch_size, \
            "--eval-n-decode must be <= --eval-batch-size"
        _vloader = DataLoader(
            val_ds,
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            drop_last=False,
            collate_fn=collate_fn,
        )
        fixed_val_batch = next(iter(_vloader))
        del _vloader
    else:
        fixed_val_batch = None

    # ---- VAE: loaded on every rank now that FID is sharded across ranks
    #      (each rank decodes its own slice of generated/GT latents to RGB).
    #      The rank-0-only visualization in eval_plot still uses the same VAE.
    from diffusers.models import AutoencoderKL
    vae_kwargs = {}
    if bridge_runtime.vae_subfolder is not None and args.vae_ckpt is None:
        vae_kwargs["subfolder"] = bridge_runtime.vae_subfolder
    vae = AutoencoderKL.from_pretrained(
        args.vae_ckpt or bridge_runtime.vae_model,
        **vae_kwargs,
    ).to(device)
    vae.to(dtype=torch.bfloat16)
    vae.eval()
    requires_grad(vae, False)
    dist.barrier()

    # ---- Resume
    train_steps = 0
    if args.auto_resume:
        assert args.resume is None
        args.resume = _find_latest_checkpoint(args.out_dir)
        if rank == 0:
            print(f"[auto-resume] latest checkpoint: {args.resume}")
    if args.resume is not None:
        from checkpoint_utils import _resolve_ckpt
        if rank == 0:
            print(f"Resuming from checkpoint: {args.resume}")
            _resolve_ckpt(args.resume) # to populate the cache
        dist.barrier()
        args.resume = _resolve_ckpt(args.resume)
        # Load onto CPU (not the GPU) so we don't briefly double the on-device
        # footprint against the already-instantiated model/EMA/optimizer.
        # load_state_dict below remaps each tensor to the param's device.
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        if args.eval_only:
            _validate_eval_checkpoint_args(
                args, ckpt, repa_image_dim,
            )
        model.module.load_state_dict(ckpt["model"])
        ema.load_state_dict(ckpt["ema"])
        ckpt_alt_emas = ckpt.get("alt_emas", {})
        for alt_decay, alt_ema in alt_emas.items():
            key = f"{alt_decay}"
            if key in ckpt_alt_emas:
                alt_ema.load_state_dict(ckpt_alt_emas[key])
            else:
                # Not present in this checkpoint (e.g. newly added decay): seed
                # it from the just-loaded model weights.
                if rank == 0:
                    print(f"[resume] alt EMA decay {alt_decay} missing from "
                          f"checkpoint; initializing from model weights.")
                update_ema(alt_ema, model.module, decay=0.0)
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        if args.use_token_text_bridge and token_decoder is not None:
            token_decoder.load_state_dict(ckpt["token_decoder"])
            token_decoder_optimizer.load_state_dict(ckpt["token_decoder_optimizer"])
        train_steps = int(ckpt.get("step", 0))
        # Drop the checkpoint dict now that everything is loaded; otherwise it
        # lingers in scope for the whole run as a full duplicate of the model /
        # EMAs / optimizer state and can OOM a resumed run on the first step.
        del ckpt, ckpt_alt_emas
        gc.collect()
        torch.cuda.empty_cache()
        if rank == 0:
            print(f"Resumed at step {train_steps}")
    else:
        if args.eval_only:
            raise ValueError("--eval-only requires --resume or --auto-resume.")
        # First-time init: sync EMA(s) with model exactly.
        update_ema(ema, model.module, decay=0.0)
        for alt_ema in alt_emas.values():
            update_ema(alt_ema, model.module, decay=0.0)

    # ---- Train loop
    slow_windows = 0  # consecutive sub-threshold log windows (throughput watchdog)
    running: dict[str, float] = {}
    counts: dict[str, int] = {}
    train_iter = cycle_loader(train_loader, train_sampler)
    log_window_start = time()
    log_window_steps = 0
    # Straggler diagnostics: time spent blocked on the dataloader vs. in the
    # compute/comm step, accumulated per rank over the logging window.
    data_wait_accum = 0.0
    compute_accum = 0.0
    # FLOP accounting (rank 0 only). Baseline is the first measured per-rank
    # fwd+bwd count; later measurements are a sanity check against it. The
    # extrapolated cumulative total is per_step_global * train_steps (constant
    # per-step, and resume-safe since train_steps is restored from checkpoint).
    _flop_baseline = None
    _flop_per_step_global = 0.0
    _params_logged = False

    while train_steps < args.steps or args.eval_only:
        if not args.eval_only:
            _t_data0 = time()
            batch = next(train_iter)
            _t_data1 = time()

            # FLOP measurement (rank 0 only). Wraps the real train_step, so it
            # captures fwd+bwd for the actual config (both directions, token
            # decoder, REPA). Fires on this process's first step (resume-safe)
            # and every eval_every steps as a "did the graph change?" check.
            _flop_ctx = None
            if rank == 0 and (_flop_baseline is None or train_steps % args.eval_every == 0):
                try:
                    from torch.utils.flop_counter import FlopCounterMode
                    _flop_ctx = FlopCounterMode(display=False)
                    _flop_ctx.__enter__()
                except Exception as _e:
                    print(f"[flops] counter unavailable: {_e}", flush=True)
                    _flop_ctx = None

            logs = train_step(
                model=model, ema=ema, alt_emas=alt_emas, sde=sde,
                optimizer=optimizer, scheduler=scheduler,
                batch=batch, device=device,
                train_forward=not args.no_forward,
                train_reverse=not args.no_reverse,
                eps=args.eps,
                grad_clip=args.grad_clip,
                ema_decay=args.ema_decay,
                unconditional_percent=args.unconditional_percent,
                use_token_text_bridge=args.use_token_text_bridge,
                token_layout=args.token_layout,
                x0_cond_source=args.x0_cond_source,
                token_decoder=token_decoder,
                token_decoder_optimizer=token_decoder_optimizer,
                token_decoder_noise_t_max=args.token_decoder_noise_t_max,
                token_decoder_pad_zero_prob=args.token_decoder_pad_zero_prob,
                bridge_config=bridge_config,
                precond=precond,
                use_repa_text=args.repa_text,
                repa_text_lambda=args.repa_text_lambda * min(
                    1.0, train_steps / max(1, args.repa_text_warmup_steps)
                ) if args.repa_text_warmup_steps > 0 else args.repa_text_lambda,
                use_repa_image=args.repa_image,
                repa_image_lambda=args.repa_image_lambda * min(
                    1.0, train_steps / max(1, args.repa_image_warmup_steps)
                ) if args.repa_image_warmup_steps > 0 else args.repa_image_lambda,
                repa_phase=args.repa_phase,
                time_sampler=args.time_sampler,
                time_sampler_logit_normal_mean=args.time_sampler_logit_normal_mean,
                time_sampler_logit_normal_std=args.time_sampler_logit_normal_std,
            )
            if _flop_ctx is not None:
                _flop_ctx.__exit__(None, None, None)
                _sf = _flop_ctx.get_total_flops()
                _world = dist.get_world_size()
                _flop_per_step_global = float(_sf) * _world
                # Sanity check: counts are exact integers, so any change means the
                # per-step graph changed (unexpected for this deterministic config).
                if _flop_baseline is not None and _sf != _flop_baseline:
                    print(f"[flops][WARN] step {train_steps}: per-step FLOPs changed "
                          f"{_sf:.3e} != baseline {_flop_baseline:.3e}", flush=True)
                _flop_baseline = _sf # use most recent value
                wandb.log({
                    "flops/train_step_fwd_bwd_per_rank": float(_sf),
                    "flops/train_step_fwd_bwd_global": _flop_per_step_global,
                }, step=train_steps)
                print(f"[flops] train_step (fwd+bwd): "
                      f"{_sf:.3e} FLOPs/rank, {_flop_per_step_global:.3e} global", flush=True)
                # Accurate parameter counts, once, after the first real backward.
                # `p.grad is not None` = params actually exercised by the enabled
                # direction(s); frozen/disabled branches (--no-forward/--no-reverse)
                # never get grads, so this excludes them (unlike a raw numel sum).
                if not _params_logged:
                    _inner = model.module if isinstance(model, DDP) else model
                    _p_total = sum(p.numel() for p in _inner.parameters())
                    _p_train = sum(p.numel() for p in _inner.parameters() if p.requires_grad)
                    _p_used = sum(p.numel() for p in _inner.parameters() if p.grad is not None)
                    wandb.log({
                        "params/total": float(_p_total),
                        "params/trainable": float(_p_train),
                        "params/used": float(_p_used),
                    }, step=train_steps)
                    print(f"[params] total={_p_total:,} trainable={_p_train:,} "
                          f"used={_p_used:,}", flush=True)
                    _params_logged = True
            # train_step ends with .item() calls that synchronize the device, so
            # this wall-time reflects the real per-rank compute+comm cost.
            compute_accum += time() - _t_data1
            data_wait_accum += _t_data1 - _t_data0
            for k, v in logs.items():
                running[k] = running.get(k, 0.0) + v
                counts[k] = counts.get(k, 0) + 1
            log_window_steps += 1
            train_steps += 1

            # ---- Periodic logging
            if train_steps % args.log_every == 0:
                torch.cuda.synchronize()
                elapsed = time() - log_window_start
                steps_per_sec = log_window_steps / max(elapsed, 1e-6)
                # Cross-rank average of the local running means.
                log_dict: dict[str, float] = {}
                for k in sorted(running):
                    local_avg = running[k] / max(1, counts[k])
                    log_dict[f"train/{k}"] = _avg_across_ranks(local_avg, device)
                log_dict["train/steps_per_sec"] = steps_per_sec
                log_dict["train/lr"] = scheduler.get_last_lr()[0]

                # ---- Per-rank straggler diagnostics. Gather each rank's mean
                #      data-wait and compute time over this window. A healthy job
                #      has max ~= mean; a slow node/link shows up as max >> mean,
                #      and compute_straggler_rank names the worst rank.
                local_data = data_wait_accum / max(1, log_window_steps)
                local_compute = compute_accum / max(1, log_window_steps)
                data_all = _gather_across_ranks(local_data, device)
                compute_all = _gather_across_ranks(local_compute, device)
                if rank == 0:
                    log_dict["train/data_wait_s_mean"] = sum(data_all) / len(data_all)
                    log_dict["train/data_wait_s_max"] = max(data_all)
                    log_dict["train/compute_s_mean"] = sum(compute_all) / len(compute_all)
                    log_dict["train/compute_s_max"] = max(compute_all)
                    log_dict["train/compute_straggler_rank"] = float(
                        max(range(len(compute_all)), key=lambda i: compute_all[i])
                    )
                    # Extrapolated cumulative training FLOPs = constant per-step
                    # global cost * steps taken (resume-safe; no per-step logging).
                    log_dict["flops/total_global_extrapolated"] = (
                        _flop_per_step_global * train_steps
                    )

                if rank == 0:
                    wandb.log(log_dict, step=train_steps)
                    msg = " | ".join(f"{k}={v:.4f}" for k, v in log_dict.items())
                    print(f"step {train_steps:>7d} | {msg}")

                # ---- Throughput watchdog: a degraded-but-not-hung node keeps every
                # collective under the NCCL timeout, so it would otherwise crawl
                # forever undetected. Rank 0 (whose steps_per_sec reflects the global
                # lockstep rate) counts consecutive slow windows and hard-exits; the
                # other ranks then fall out via the short PG timeout and the launch
                # retry loop requeues. Placed after all log-window collectives so the
                # exit can't desync a partially-completed collective.
                if rank == 0 and args.min_throughput > 0:
                    if steps_per_sec < args.min_throughput:
                        slow_windows += 1
                        print(f"[throughput-watchdog] steps_per_sec={steps_per_sec:.4f} < "
                              f"{args.min_throughput} ({slow_windows}/{args.min_throughput_windows})",
                              flush=True)
                    else:
                        slow_windows = 0
                    if slow_windows >= args.min_throughput_windows:
                        print(f"[throughput-watchdog] throughput below {args.min_throughput} it/s "
                              f"for {slow_windows} consecutive windows; exiting rank 0 (code 1).",
                              flush=True)
                        os._exit(1)

                running.clear(); counts.clear()
                log_window_start = time()
                log_window_steps = 0
                data_wait_accum = 0.0
                compute_accum = 0.0

            # ---- Eval
            ran_eval = train_steps % args.eval_every == 0
            ran_ckpt = train_steps % args.ckpt_every == 0
        else:
            # --eval-only: run a single eval pass on the resumed checkpoint, no ckpt.
            ran_eval, ran_ckpt = True, False

        # ---- Checkpoint
        if ran_ckpt:
            if rank == 0:
                ckpt_path = ckpt_dir / f"step_{train_steps:07d}.pt"
                torch.save({
                    "step": train_steps,
                    "model": model.module.state_dict(),
                    "ema": ema.state_dict(),
                    **({
                        "alt_emas": {
                            f"{alt_decay}": alt_ema.state_dict()
                            for alt_decay, alt_ema in alt_emas.items()
                        },
                    } if alt_emas else {}),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    **({
                        "token_decoder": token_decoder.state_dict(),
                        "token_decoder_optimizer": token_decoder_optimizer.state_dict(),
                    } if args.use_token_text_bridge and token_decoder is not None else {}),
                    "args": vars(args),
                }, ckpt_path)
                print(f"[rank 0] saved checkpoint -> {ckpt_path}")
            dist.barrier(group=eval_pg)

        if ran_eval and rank == 0:
            print(f"[mem] pre-eval      step {train_steps} "
                  f"alloc={torch.cuda.memory_allocated()/1e9:.2f}GB "
                  f"reserved={torch.cuda.memory_reserved()/1e9:.2f}GB "
                  f"max_reserved={torch.cuda.max_memory_reserved()/1e9:.2f}GB")
        if ran_eval:
            # Rank-0-only visualizations + paired val loss (existing).
            if isinstance(sde, FlowMatchingODE):
                ode_eval_flags = [True]
                eval_cfg_scales = (
                    [0.0] if args.force_unconditional else args.eval_cfg_scales
                )
            else:
                ode_eval_flags = [False, True] if args.eval_ode else [False]
                eval_cfg_scales = args.eval_cfg_scales
            if rank == 0:
              for cfg_scale in eval_cfg_scales:
                for ode in ode_eval_flags:
                  cycle_text_model = None
                  cycle_text_tokenizer = None
                  if args.use_token_text_bridge and args.x0_cond_source == "global_text":
                    from data_utils.encode_common_catalog import load_text_encoder
                    cycle_text_tokenizer, cycle_text_model = load_text_encoder(
                        global_text_model_id,
                        device,
                        dtype=torch.bfloat16,
                        cache_dir=args.cache_dir,
                    )
                  eval_and_log_visuals(
                    eval_model=ema,
                    sde=sde,
                    vae=vae,
                    val_batch=fixed_val_batch,
                    step=train_steps,
                    device=device,
                    out_dir=eval_dir,
                    num_steps=args.eval_num_ode_steps if ode else args.eval_num_sde_steps,
                    decode_every_k=args.eval_decode_every_k,
                    n_decode=args.eval_n_decode,
                    grid_nrow=args.eval_grid_nrow or None,
                    fps=args.eval_fps,
                    wandb_logger=wandb,
                    autocast_dtype=torch.bfloat16,
                    cfg_scale=cfg_scale,
                    ode=ode,
                    use_token_text_bridge=args.use_token_text_bridge,
                    token_layout=args.token_layout,
                    x0_cond_source=args.x0_cond_source,
                    runtime_config=bridge_runtime,
                    token_decoder=token_decoder,
                    token_tokenizer=token_tokenizer,
                    cycle_text_tokenizer=cycle_text_tokenizer,
                    cycle_text_model=cycle_text_model,
                    cycle_text_max_length=128,
                    no_reverse=not can_infer_reverse,
                    no_forward=args.no_forward,
                  )
                  del cycle_text_model, cycle_text_tokenizer
                  torch.cuda.empty_cache()
            dist.barrier(group=eval_pg)

            # ---- Distributed metrics: every rank participates so we shard
            #      generation across all GPUs. Returned dicts are non-empty
            #      on rank 0 (FID) / every rank (i2t cosine, all-reduced).
            metric_logs: dict[str, float] = {}
            if args.eval_fid_num_samples > 0 and len(fid_eval_indices) > 0 \
            and (not args.no_forward) and (not bridge_config.image_as_noise):
              for cfg_scale in eval_cfg_scales:
                for ode in ode_eval_flags:
                  score_tag = f"step_{train_steps:07d}_cfg_{cfg_scale}_{'ode' if ode else 'sde'}"
                  fid_logs = compute_fid_distributed(
                    eval_model=ema, sde=sde, vae=vae,
                    val_ds=val_ds, eval_indices=fid_eval_indices,
                    rank=rank, world_size=world_size, device=device,
                    num_steps=args.eval_num_ode_steps if ode else args.eval_num_sde_steps,
                    batch_size=args.eval_fid_batch_size,
                    collate_fn=collate_fn,
                    autocast_dtype=torch.bfloat16,
                    feature=args.eval_fid_feature,
                    cfg_scale=cfg_scale,
                    ode=ode,
                    use_token_text_bridge=args.use_token_text_bridge,
                    token_layout=args.token_layout,
                    x0_cond_source=args.x0_cond_source,
                    runtime_config=bridge_runtime,
                    compute_clipscore=not args.no_eval_clipscore,
                    clip_scores_path=(
                        clip_scores_dir / f"clip_scores_t2i_{score_tag}.json"
                        if not args.no_eval_clipscore
                        else None
                    ),
                    eval_pg=eval_pg,
                  )
                  metric_logs.update(fid_logs)

            if (
                args.use_token_text_bridge
                and token_decoder is not None
                and args.eval_text_decode_num_samples > 0
                and len(text_decode_eval_indices) > 0
                and can_infer_reverse
                and (not bridge_config.text_as_noise)
            ):
              for cfg_scale in eval_cfg_scales:
                for ode in ode_eval_flags:
                  score_tag = f"step_{train_steps:07d}_cfg_{cfg_scale}_{'ode' if ode else 'sde'}"
                  text_logs = compute_text_decode_distributed(
                    eval_model=ema, sde=sde,
                    token_decoder=token_decoder, tokenizer=token_tokenizer,
                    val_ds=val_ds, eval_indices=text_decode_eval_indices,
                    rank=rank, world_size=world_size, device=device,
                    num_steps=args.eval_num_ode_steps if ode else args.eval_num_sde_steps,
                    batch_size=args.eval_text_decode_batch_size,
                    collate_fn=collate_fn,
                    autocast_dtype=torch.bfloat16,
                    cfg_scale=cfg_scale,
                    ode=ode,
                    token_layout=args.token_layout,
                    runtime_config=bridge_runtime,
                    compute_cider=not args.no_eval_cider,
                    compute_clipscore=not args.no_eval_clipscore,
                    compute_genppl=not args.no_eval_genppl,
                    genppl_model=args.eval_genppl_model,
                    genppl_batch_size=args.eval_genppl_batch_size,
                    include_padding_in_accuracy=args.eval_text_decode_include_padding_in_accuracy,
                    vae=vae,
                    clip_scores_path=(
                        clip_scores_dir / f"clip_scores_i2t_{score_tag}.json"
                        if not args.no_eval_clipscore
                        else None
                    ),
                    eval_pg=eval_pg,
                  )
                  metric_logs.update(text_logs)

            if rank == 0 and metric_logs:
                # commit=True finalizes this step now so a crash before the next
                # training-step log can't drop the eval row.
                wandb.log(metric_logs, step=train_steps, commit=True)
                msg = " | ".join(
                    f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                    for k, v in metric_logs.items()
                )
                print(f"eval step {train_steps:>7d} | {msg}")
            dist.barrier(group=eval_pg)

        # ---- Post eval/ckpt housekeeping
        # Eval and checkpointing do a large amount of non-training GPU work.
        # Reclaim the transient allocations they leave behind (VAE decode
        # buffers, sampling activations, ...) so they don't fragment the
        # caching allocator and slow down subsequent training steps, then
        # restart the throughput window so the eval/ckpt wall-time is not
        # charged to the next steps_per_sec measurement.
        if ran_eval or ran_ckpt:
            gc.collect()
            torch.cuda.empty_cache()
            if rank == 0:
                print(f"[mem] post-cleanup step {train_steps} "
                      f"alloc={torch.cuda.memory_allocated()/1e9:.2f}GB "
                      f"reserved={torch.cuda.memory_reserved()/1e9:.2f}GB "
                      f"max_reserved={torch.cuda.max_memory_reserved()/1e9:.2f}GB")
            torch.cuda.reset_peak_memory_stats()
            slow_windows = 0  # don't let eval/post-eval (cold-cache) windows trip the watchdog
            log_window_start = time()
            log_window_steps = 0
            data_wait_accum = 0.0
            compute_accum = 0.0

        if args.eval_only:
            break

    if rank == 0:
        wandb.finish()
        print("Done.")
    cleanup_ddp()


if __name__ == "__main__":
    main()

# APPROVED