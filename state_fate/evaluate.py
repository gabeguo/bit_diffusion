from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from state_fate.dataset import StateFatePairDataset
from state_fate.eval_utils import (
    evaluate_batch,
    load_cached_eval_outputs,
    save_eval_outputs,
)
from state_fate.plotting import (
    save_day6_overlay_figure,
    save_latent_eval_figure,
    save_trajectory_figure,
)
from state_fate.train import build_model, build_sde


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a bit_diffusion LARRY state-fate checkpoint."
    )
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Processed data root. Defaults to checkpoint training args.",
    )
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-steps", type=int, default=250)
    parser.add_argument("--cfg-scale", type=float, default=0.0)
    parser.add_argument("--sampler", type=str, default="sde", choices=["sde", "ode"])
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument(
        "--replot",
        action="store_true",
        help="Regenerate figures from cached eval_tensors/metrics in --out-dir.",
    )
    parser.add_argument(
        "--step", type=int, default=None, help="Cached eval step to replot."
    )
    parser.add_argument(
        "--embedding",
        type=str,
        default="pca",
        choices=["pca", "tsne", "umap", "raw"],
        help="2D embedding used for the trajectory figure.",
    )
    parser.add_argument(
        "--latent-embedding",
        type=str,
        default="pca",
        choices=["pca", "tsne", "umap", "raw"],
        help="2D embedding used for the six-panel latent diagnostic.",
    )
    parser.add_argument(
        "--replot-figures",
        type=str,
        default="both",
        choices=["both", "trajectory", "latent", "overlay"],
        help="Which cached figures to regenerate when --replot is set.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Optional title for saved figures.",
    )
    parser.add_argument(
        "--max-arrows",
        type=int,
        default=28,
        help="Maximum transport arrows shown in trajectory replots.",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=220,
        help="Maximum visible scatter points per trajectory panel.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.replot:
        if args.out_dir is None:
            raise ValueError(
                "--replot requires --out-dir pointing at an eval output directory"
            )
        step, metrics, tensors = load_cached_eval_outputs(args.out_dir, step=args.step)
        title = (
            args.title
            if args.title is not None
            else f"LARRY cached evaluation step {step}"
        )
        if args.replot_figures in {"both", "trajectory"}:
            save_trajectory_figure(
                out_dir=args.out_dir,
                step=step,
                metrics=metrics,
                tensors=tensors,
                title=title,
                max_arrows=args.max_arrows,
                max_points=args.max_points,
                embedding=args.embedding,
            )
        if args.replot_figures in {"both", "latent"}:
            save_latent_eval_figure(
                out_dir=args.out_dir,
                step=step,
                tensors=tensors,
                title=title,
                embedding=args.latent_embedding,
            )
        if args.replot_figures in {"both", "overlay"}:
            save_day6_overlay_figure(
                out_dir=args.out_dir,
                step=step,
                tensors=tensors,
                title=title,
                embedding=args.embedding,
            )
        print(
            f"replotted {args.replot_figures} figure(s) for step {step} "
            f"with trajectory_embedding={args.embedding}, "
            f"latent_embedding={args.latent_embedding} in {args.out_dir}"
        )
        return
    if args.checkpoint is None:
        raise ValueError("--checkpoint is required unless --replot is set")

    device = torch.device(args.device)
    ckpt_path = Path(args.checkpoint)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    train_args = argparse.Namespace(**ckpt["args"])
    data_root = args.data_root or train_args.data_root

    dataset = StateFatePairDataset(data_root, split=args.split)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    batch = next(iter(loader))

    model = build_model(
        train_args,
        x_dim=int(ckpt.get("x_dim", dataset.x_dim)),
        num_context_classes=int(
            ckpt.get("num_context_classes", dataset.num_context_classes)
        ),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    sde = build_sde(train_args, model)

    metrics, tensors = evaluate_batch(
        model=model,
        sde=sde,
        batch=batch,
        device=device,
        num_steps=args.num_steps,
        cfg_scale=args.cfg_scale,
        objective=getattr(train_args, "objective", "score"),
        sampler=args.sampler,
    )
    step = int(ckpt.get("step", 0))
    out_dir = (
        Path(args.out_dir)
        if args.out_dir is not None
        else ckpt_path.parent.parent / "test_standalone"
    )
    save_eval_outputs(
        out_dir=out_dir,
        step=step,
        metrics=metrics,
        tensors=tensors,
        title=(
            args.title if args.title is not None else f"LARRY evaluation step {step}"
        ),
        embedding=args.embedding,
        latent_embedding=args.latent_embedding,
    )
    for key, value in sorted(metrics.items()):
        print(f"{key}: {value:.6f}")
    print(f"wrote eval outputs to {out_dir}")


if __name__ == "__main__":
    main()
