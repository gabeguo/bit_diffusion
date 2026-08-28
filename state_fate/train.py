from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime
from pathlib import Path
from time import time

import torch
import torch.distributed as dist
import torch.optim as optim
from torch.amp import autocast
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from sde_utils.loss import dsm_loss, sample_p_base_x_t_cond_x_0_x_1
from sde_utils.sde import PeriodicVolatilitySDE, UniformVolatilitySDE
from state_fate.baselines import (
    CosineDecayVolatilitySDE,
    endpoint_regression_loss,
    flow_matching_loss,
    noise_to_data_loss,
)
from state_fate.dataset import StateFatePairDataset
from state_fate.eval_utils import evaluate_batch, save_eval_outputs
from state_fate.models import DirectionalBridgeModel, StateFateDiT, StateFateScoreMLP


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the bit_diffusion LARRY state-fate benchmark on clone-paired endpoints."
    )
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default="state_fate/runs/larry")
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Per-process batch size; multiply by world size for the global batch size.",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument(
        "--precision", type=str, default="bf16", choices=["bf16", "fp32"]
    )
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--num-blocks", type=int, default=6)
    parser.add_argument("--time-dim", type=int, default=128)
    parser.add_argument("--class-dim", type=int, default=128)
    parser.add_argument("--arch", type=str, default="mlp", choices=["mlp", "dit"])
    parser.add_argument(
        "--model-sharing", type=str, default="shared", choices=["shared", "separate"]
    )
    parser.add_argument(
        "--shared-param-mult",
        type=float,
        default=1.0,
        help="Approximate parameter multiplier for shared models; use 2 for shared-vs-two-model runs.",
    )
    parser.add_argument("--dit-token-dim", type=int, default=8)
    parser.add_argument("--dit-num-heads", type=int, default=8)
    parser.add_argument("--dit-mlp-ratio", type=float, default=4.0)
    parser.add_argument(
        "--objective",
        type=str,
        default="score",
        choices=["score", "flow", "endpoint", "noise"],
    )
    parser.add_argument("--unconditional-percent", type=float, default=0.0)
    parser.add_argument(
        "--no-reverse", action="store_true", help="Train forward direction only."
    )
    parser.add_argument("--eps", type=float, default=1e-4)
    parser.add_argument(
        "--sde",
        type=str,
        default="uniform",
        choices=["uniform", "periodic", "cosine_decay"],
    )
    parser.add_argument("--K", type=float, default=0.5)
    parser.add_argument("--cosine-sde-eps", type=float, default=0.03)
    parser.add_argument("--periodic-sde-alpha", type=float, default=0.85)
    parser.add_argument("--periodic-sde-k", type=float, default=1.0)
    parser.add_argument("--periodic-sde-eps", type=float, default=0.05)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--ckpt-every", type=int, default=5000)
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=512,
        help="Number of deterministic test pairs evaluated at each interval.",
    )
    parser.add_argument("--eval-num-steps", type=int, default=250)
    parser.add_argument("--eval-cfg-scale", type=float, default=0.0)
    parser.add_argument(
        "--eval-sampler", type=str, default="sde", choices=["sde", "ode"]
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--local-rank", type=int, default=-1)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def setup_distributed(args: argparse.Namespace) -> tuple[torch.device, int, int, bool]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    if distributed:
        local_rank = int(
            os.environ.get("LOCAL_RANK", args.local_rank if args.local_rank >= 0 else 0)
        )
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
            backend = "nccl"
        else:
            device = torch.device("cpu")
            backend = "gloo"
        dist.init_process_group(backend=backend)
        return device, dist.get_rank(), world_size, True
    return torch.device(args.device), 0, 1, False


def cleanup_distributed(distributed: bool) -> None:
    if distributed and dist.is_initialized():
        dist.destroy_process_group()


def is_main(rank: int) -> bool:
    return rank == 0


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def build_model(
    args: argparse.Namespace,
    *,
    x_dim: int,
    num_context_classes: int,
) -> torch.nn.Module:
    hidden_dim = getattr(args, "hidden_dim", 512)
    model_sharing = getattr(args, "model_sharing", "shared")
    shared_param_mult = getattr(args, "shared_param_mult", 1.0)
    if model_sharing == "shared" and shared_param_mult != 1.0:
        hidden_dim = max(1, int(round(hidden_dim * math.sqrt(shared_param_mult))))
        if getattr(args, "arch", "mlp") == "dit":
            num_heads = getattr(args, "dit_num_heads", 8)
            hidden_dim = max(num_heads, int(round(hidden_dim / num_heads)) * num_heads)

    def make_one() -> torch.nn.Module:
        common = {
            "x_dim": x_dim,
            "num_classes": num_context_classes,
            "hidden_dim": hidden_dim,
            "num_blocks": getattr(args, "num_blocks", 6),
            "time_dim": getattr(args, "time_dim", 128),
            "class_dim": getattr(args, "class_dim", 128),
        }
        arch = getattr(args, "arch", "mlp")
        if arch == "mlp":
            return StateFateScoreMLP(**common)
        if arch == "dit":
            return StateFateDiT(
                **common,
                token_dim=getattr(args, "dit_token_dim", 8),
                num_heads=getattr(args, "dit_num_heads", 8),
                mlp_ratio=getattr(args, "dit_mlp_ratio", 4.0),
            )
        raise ValueError(f"unknown arch: {arch}")

    if model_sharing == "separate":
        return DirectionalBridgeModel(make_one(), make_one())
    return make_one()


def build_sde(args: argparse.Namespace, model: torch.nn.Module):
    sde_name = getattr(args, "sde", "uniform")
    if sde_name == "uniform":
        return UniformVolatilitySDE(A=0, K=getattr(args, "K", 0.5), score_network=model)
    if sde_name == "periodic":
        return PeriodicVolatilitySDE(
            alpha=getattr(args, "periodic_sde_alpha", 0.85),
            k=getattr(args, "periodic_sde_k", 1.0),
            eps=getattr(args, "periodic_sde_eps", 0.05),
            score_network=model,
        )
    if sde_name == "cosine_decay":
        return CosineDecayVolatilitySDE(
            K=getattr(args, "K", 0.5),
            eps=getattr(args, "cosine_sde_eps", 0.03),
            score_network=model,
        )
    raise ValueError(f"unknown sde: {sde_name}")


def sample_t(batch_size: int, device: torch.device, eps: float) -> torch.Tensor:
    return torch.rand(batch_size, device=device) * (1 - 2 * eps) + eps


def cycle_loader(loader: DataLoader, sampler: DistributedSampler | None = None):
    epoch = 0
    while True:
        if sampler is not None:
            sampler.set_epoch(epoch)
        for batch in loader:
            yield batch
        epoch += 1


def make_cond_mask(
    batch_size: int,
    *,
    device: torch.device,
    unconditional_percent: float,
) -> torch.Tensor:
    if unconditional_percent > 0:
        return torch.rand(batch_size, device=device) > unconditional_percent
    return torch.ones(batch_size, dtype=torch.bool, device=device)


def train_step(
    *,
    model: torch.nn.Module,
    sde,
    optimizer: optim.Optimizer,
    batch: dict[str, torch.Tensor],
    device: torch.device,
    train_reverse: bool,
    eps: float,
    grad_clip: float,
    unconditional_percent: float,
    objective: str,
    precision: str,
) -> dict[str, float]:
    model.train()
    x_0 = batch["x_0"].to(device)
    x_1 = batch["x_1"].to(device)
    y = batch["y"].to(device)

    optimizer.zero_grad(set_to_none=True)
    total = x_0.new_zeros(())
    logs: dict[str, float] = {}
    use_bf16 = precision == "bf16" and device.type == "cuda"
    with autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
        for reverse in [False, True] if train_reverse else [False]:
            cond_mask = make_cond_mask(
                x_0.shape[0],
                device=device,
                unconditional_percent=unconditional_percent,
            )
            if objective == "score":
                t = sample_t(x_0.shape[0], device, eps=eps)
                x_t = sample_p_base_x_t_cond_x_0_x_1(sde=sde, x_0=x_0, x_1=x_1, t=t)
                loss = dsm_loss(
                    model=model,
                    sde=sde,
                    x_t=x_t,
                    x_1=x_1,
                    x_0=x_0,
                    t=t,
                    y=y,
                    reverse=reverse,
                    cond_mask=cond_mask,
                    x_cond_0=x_0 if not reverse else None,
                    x_cond_1=x_1 if reverse else None,
                )
            elif objective == "flow":
                t = sample_t(x_0.shape[0], device, eps=eps)
                loss = flow_matching_loss(
                    model=model,
                    x_0=x_0,
                    x_1=x_1,
                    t=t,
                    y=y,
                    reverse=reverse,
                    cond_mask=cond_mask,
                )
            elif objective == "endpoint":
                loss = endpoint_regression_loss(
                    model=model,
                    x_0=x_0,
                    x_1=x_1,
                    y=y,
                    reverse=reverse,
                    cond_mask=cond_mask,
                )
            elif objective == "noise":
                loss = noise_to_data_loss(
                    model=model,
                    x_0=x_0,
                    x_1=x_1,
                    t=sample_t(x_0.shape[0], device, eps=eps),
                    y=y,
                    reverse=reverse,
                    cond_mask=cond_mask,
                )
            else:
                raise ValueError(f"unknown objective: {objective}")
            total = total + loss
            logs[f"loss/{'reverse' if reverse else 'forward'}"] = loss.item()
    total.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
    optimizer.step()
    logs["grad_norm"] = float(grad_norm.item())
    return logs


def main() -> None:
    args = parse_args()
    if not 0 <= args.unconditional_percent < 1:
        raise ValueError("--unconditional-percent must be in [0, 1)")
    if args.shared_param_mult <= 0:
        raise ValueError("--shared-param-mult must be positive")
    device, rank, world_size, distributed = setup_distributed(args)
    torch.manual_seed(args.seed + rank)

    train_ds = StateFatePairDataset(args.data_root, split="train")
    test_ds = StateFatePairDataset(args.data_root, split="test")

    train_sampler = (
        DistributedSampler(
            train_ds,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=False,
        )
        if distributed
        else None
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.test_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )
    test_batch = next(iter(test_loader))

    raw_model = build_model(
        args,
        x_dim=train_ds.x_dim,
        num_context_classes=train_ds.num_context_classes,
    ).to(device)
    sde = build_sde(args, raw_model)
    model = raw_model
    if distributed:
        ddp_kwargs = {"device_ids": [device.index]} if device.type == "cuda" else {}
        # Both subnetworks participate unless a separate model is forward-only.
        ddp_kwargs["find_unused_parameters"] = (
            args.model_sharing == "separate" and args.no_reverse
        )
        model = DDP(model, **ddp_kwargs)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) / timestamp
    ckpt_dir = out_dir / "checkpoints"
    test_dir = out_dir / "test"
    if is_main(rank):
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        test_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "args.json").write_text(json.dumps(vars(args), indent=2) + "\n")
    if distributed:
        dist.barrier()

    if is_main(rank):
        n_params = sum(p.numel() for p in raw_model.parameters())
        print(f"data_root={args.data_root}")
        print(
            f"train pairs={len(train_ds):,}; test pairs={len(test_ds):,}; "
            f"x_dim={train_ds.x_dim}"
        )
        print(
            f"per-process batch={args.batch_size:,}; "
            f"global batch={args.batch_size * world_size:,}"
        )
        print(
            f"context classes={train_ds.num_context_classes}; fate classes={train_ds.num_fate_classes}"
        )
        print(
            f"arch={args.arch}; sharing={args.model_sharing}; "
            f"objective={args.objective}; sde={args.sde}; "
            f"precision={args.precision}; world_size={world_size}"
        )
        print(f"params={n_params:,}")
        print(f"writing to {out_dir}")

    train_iter = cycle_loader(train_loader, train_sampler)
    running: dict[str, float] = {}
    counts: dict[str, int] = {}
    window_start = time()

    try:
        for step in range(1, args.steps + 1):
            batch = next(train_iter)
            logs = train_step(
                model=model,
                sde=sde,
                optimizer=optimizer,
                batch=batch,
                device=device,
                train_reverse=not args.no_reverse,
                eps=args.eps,
                grad_clip=args.grad_clip,
                unconditional_percent=args.unconditional_percent,
                objective=args.objective,
                precision=args.precision,
            )
            if is_main(rank):
                for key, value in logs.items():
                    running[key] = running.get(key, 0.0) + value
                    counts[key] = counts.get(key, 0) + 1

            if is_main(rank) and step % args.log_every == 0:
                elapsed = time() - window_start
                avg = {k: running[k] / max(1, counts[k]) for k in sorted(running)}
                avg["steps_per_sec"] = args.log_every / max(elapsed, 1e-6)
                msg = " | ".join(f"{k}={v:.4f}" for k, v in avg.items())
                print(f"step {step:>7d} | {msg}")
                running.clear()
                counts.clear()
                window_start = time()

            if step % args.eval_every == 0 or step == args.steps:
                if distributed:
                    dist.barrier()
                if is_main(rank):
                    raw_model.eval()
                    metrics, tensors = evaluate_batch(
                        model=raw_model,
                        sde=sde,
                        batch=test_batch,
                        device=device,
                        num_steps=args.eval_num_steps,
                        cfg_scale=args.eval_cfg_scale,
                        objective=args.objective,
                        sampler=args.eval_sampler,
                    )
                    msg = " | ".join(f"{k}={v:.4f}" for k, v in sorted(metrics.items()))
                    print(f"test {step:>7d} | {msg}")
                    save_eval_outputs(
                        out_dir=test_dir,
                        step=step,
                        metrics=metrics,
                        tensors=tensors,
                        title=f"LARRY {args.objective} step {step}",
                    )
                    raw_model.train()
                if distributed:
                    dist.barrier()

            if step % args.ckpt_every == 0 or step == args.steps:
                if is_main(rank):
                    ckpt_path = ckpt_dir / f"step_{step:07d}.pt"
                    torch.save(
                        {
                            "step": step,
                            "model": unwrap_model(model).state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "args": vars(args),
                            "x_dim": train_ds.x_dim,
                            "num_context_classes": train_ds.num_context_classes,
                        },
                        ckpt_path,
                    )
                    print(f"saved checkpoint -> {ckpt_path}")
                if distributed:
                    dist.barrier()

        if is_main(rank):
            print(f"done. outputs in {out_dir}")
    finally:
        cleanup_distributed(distributed)


if __name__ == "__main__":
    main()
