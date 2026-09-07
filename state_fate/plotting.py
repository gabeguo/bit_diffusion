from __future__ import annotations

import os
import tempfile
import warnings
from pathlib import Path

import torch
import torch.nn.functional as F


def _setup_matplotlib():
    if "MPLCONFIGDIR" not in os.environ:
        mpl_dir = Path(tempfile.gettempdir()) / "bit_diffusion_state_fate_mpl"
        mpl_dir.mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(mpl_dir)
    if "XDG_CACHE_HOME" not in os.environ:
        cache_dir = Path(tempfile.gettempdir()) / "bit_diffusion_state_fate_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ["XDG_CACHE_HOME"] = str(cache_dir)
    import matplotlib.pyplot as plt

    try:
        import seaborn as sns

        sns.set_theme(style="whitegrid", context="talk")
    except ImportError:
        pass

    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 320,
            "font.size": 16,
            "axes.titlesize": 21,
            "axes.labelsize": 19,
            "axes.linewidth": 2.0,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "xtick.major.width": 1.7,
            "ytick.major.width": 1.7,
            "xtick.major.size": 6,
            "ytick.major.size": 6,
            "grid.linewidth": 1.1,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    return plt


def _style_axis(ax) -> None:
    ax.grid(alpha=0.18, linewidth=1.1)
    for spine in ax.spines.values():
        spine.set_linewidth(2.0)
        spine.set_color("#1f2933")
    ax.tick_params(width=1.7, length=6, colors="#1f2933")
    ax.xaxis.label.set_fontweight("bold")
    ax.yaxis.label.set_fontweight("bold")
    ax.title.set_fontweight("bold")


def _fate_scatter_style(labels):
    import matplotlib.colors as mcolors

    n_colors = max(1, int(labels.max()) + 1 if len(labels) else 1)
    try:
        import seaborn as sns

        colors = sns.color_palette("tab20", n_colors=n_colors)
    except ImportError:
        cmap = "tab20"
        norm = None
        ticks = None
        return cmap, norm, ticks
    cmap = mcolors.ListedColormap(colors)
    norm = mcolors.BoundaryNorm([i - 0.5 for i in range(n_colors + 1)], cmap.N)
    ticks = list(range(n_colors))
    return cmap, norm, ticks


def _draw_density(ax, arr) -> None:
    if len(arr) < 12:
        return
    try:
        import seaborn as sns
    except ImportError:
        return
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        sns.kdeplot(
            x=arr[:, 0],
            y=arr[:, 1],
            ax=ax,
            levels=7,
            fill=True,
            thresh=0.04,
            cmap="Greys",
            alpha=0.20,
            zorder=0,
        )
        sns.kdeplot(
            x=arr[:, 0],
            y=arr[:, 1],
            ax=ax,
            levels=5,
            fill=False,
            thresh=0.06,
            color="#111827",
            linewidths=1.15,
            alpha=0.18,
            zorder=1,
        )


def _split_projection(xy: torch.Tensor, *xs: torch.Tensor) -> list[torch.Tensor]:
    out = []
    offset = 0
    for x in xs:
        out.append(xy[offset : offset + len(x)])
        offset += len(x)
    return out


def _raw_projection(*xs: torch.Tensor) -> list[torch.Tensor]:
    out = []
    for x in xs:
        xy = x.float()
        if xy.shape[1] < 2:
            xy = F.pad(xy, (0, 2 - xy.shape[1]))
        out.append(xy[:, :2])
    return out


def _pca_projection(*xs: torch.Tensor) -> list[torch.Tensor]:
    all_x = torch.cat(xs, dim=0).float()
    if all_x.shape[1] < 2:
        all_x = F.pad(all_x, (0, 2 - all_x.shape[1]))
    centered = all_x - all_x.mean(dim=0, keepdim=True)
    try:
        _, _, vh = torch.linalg.svd(centered, full_matrices=False)
        basis = vh[:2].T
        xy = centered @ basis
    except RuntimeError:
        xy = centered[:, :2]
    return _split_projection(xy, *xs)


def _nonlinear_projection(
    method: str,
    *xs: torch.Tensor,
    seed: int = 0,
) -> list[torch.Tensor]:
    all_x = torch.cat(xs, dim=0).float().numpy()
    if method == "tsne":
        from sklearn.manifold import TSNE

        n = len(all_x)
        perplexity = max(5, min(30, (n - 1) // 3))
        xy = TSNE(
            n_components=2,
            init="pca",
            learning_rate="auto",
            method="exact",
            perplexity=perplexity,
            random_state=seed,
        ).fit_transform(all_x)
    elif method == "umap":
        try:
            import umap
        except ImportError as exc:
            raise ImportError(
                "UMAP plotting requires `pip install umap-learn`; use "
                "`--embedding tsne` if you want no new dependency."
            ) from exc
        xy = umap.UMAP(
            n_components=2,
            n_neighbors=30,
            min_dist=0.12,
            metric="euclidean",
            random_state=seed,
        ).fit_transform(all_x)
    else:
        raise ValueError(f"unknown nonlinear embedding: {method}")
    return _split_projection(torch.from_numpy(xy).float(), *xs)


def _projection(
    *xs: torch.Tensor,
    embedding: str,
    seed: int = 0,
) -> list[torch.Tensor]:
    if embedding == "raw":
        return _raw_projection(*xs)
    if embedding == "pca":
        return _pca_projection(*xs)
    if embedding in {"tsne", "umap"}:
        return _nonlinear_projection(embedding, *xs, seed=seed)
    raise ValueError(f"unknown embedding: {embedding}")


def _axis_labels(embedding: str) -> tuple[str, str]:
    if embedding == "raw":
        return "latent dim 1", "latent dim 2"
    if embedding == "pca":
        return "PC 1", "PC 2"
    if embedding == "tsne":
        return "t-SNE 1", "t-SNE 2"
    if embedding == "umap":
        return "UMAP 1", "UMAP 2"
    raise ValueError(f"unknown embedding: {embedding}")


def _plot_limits(
    xy: torch.Tensor,
    *,
    pad_frac: float = 0.08,
    robust: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    if robust and len(xy) >= 20:
        lo = torch.quantile(xy, 0.02, dim=0)
        hi = torch.quantile(xy, 0.98, dim=0)
    else:
        lo = xy.min(dim=0).values
        hi = xy.max(dim=0).values
    pad = pad_frac * (hi - lo).clamp_min(1e-6)
    return lo - pad, hi + pad


def _subsample_indices(n: int, max_count: int, *, seed: int = 0) -> torch.Tensor:
    if n <= max_count:
        return torch.arange(n)
    if max_count <= 0:
        return torch.empty(0, dtype=torch.long)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return torch.randperm(n, generator=generator)[:max_count].sort().values


def save_trajectory_figure(
    *,
    out_dir: str | Path,
    step: int,
    metrics: dict[str, float],
    tensors: dict[str, torch.Tensor],
    title: str,
    max_arrows: int = 28,
    max_points: int = 220,
    embedding: str = "pca",
    seed: int = 0,
) -> None:
    try:
        plt = _setup_matplotlib()
    except ImportError:
        print("matplotlib is not installed; skipping trajectory plot")
        return

    x0, xf, x1 = _projection(
        tensors["x_0"],
        tensors["x_1_fwd"],
        tensors["x_1"],
        embedding=embedding,
        seed=seed,
    )
    x_label, y_label = _axis_labels(embedding)
    labels = tensors["fate_label"].numpy()
    cmap, norm, cbar_ticks = _fate_scatter_style(labels)
    panels = [
        ("Day 2 Progenitors", x0),
        ("Generated Day 6", xf),
        ("Real Day 6 Descendants", x1),
    ]
    all_xy = torch.cat([x0, xf, x1], dim=0)
    lo = all_xy.min(dim=0).values
    hi = all_xy.max(dim=0).values
    pad = 0.06 * (hi - lo).clamp_min(1e-6)
    lo = lo - pad
    hi = hi + pad

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(18.6, 6.2),
        sharex=True,
        sharey=True,
    )
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0.055, right=0.925, top=0.835, bottom=0.125, wspace=0.09)
    scatter = None
    for panel_idx, (ax, (name, xy)) in enumerate(zip(axes, panels)):
        arr = xy.numpy()
        _draw_density(ax, arr)
        point_idx = _subsample_indices(len(arr), max_points, seed=seed + panel_idx)
        point_arr = arr[point_idx.numpy()]
        point_labels = labels[point_idx.numpy()]
        scatter = ax.scatter(
            point_arr[:, 0],
            point_arr[:, 1],
            c=point_labels,
            s=34,
            alpha=0.84,
            linewidths=0.45,
            edgecolors="white",
            cmap=cmap,
            norm=norm,
            rasterized=True,
            zorder=4,
        )
        ax.set_title(name)
        ax.set_xlim(float(lo[0]), float(hi[0]))
        ax.set_ylim(float(lo[1]), float(hi[1]))
        ax.set_xlabel(x_label)
        _style_axis(ax)
    axes[0].set_ylabel(y_label)

    n = len(x0)
    if n and max_arrows > 0:
        idx = _subsample_indices(n, max_arrows, seed=seed + 99)
        start = x0[idx].numpy()
        end = xf[idx].numpy()
        for xy_start, xy_end in zip(start, end):
            axes[1].annotate(
                "",
                xy=xy_end,
                xytext=xy_start,
                arrowprops={
                    "arrowstyle": "-|>",
                    "color": "#111827",
                    "linewidth": 1.75,
                    "alpha": 0.30,
                    "mutation_scale": 11,
                    "shrinkA": 0,
                    "shrinkB": 0,
                },
                zorder=2,
            )

    fig.suptitle(title, fontsize=25, fontweight="bold", x=0.47, y=0.955)
    if scatter is not None:
        cbar = fig.colorbar(
            scatter,
            ax=axes,
            fraction=0.026,
            pad=0.012,
            ticks=cbar_ticks,
        )
        cbar.set_label("Fate Label", fontweight="bold")
        if cbar_ticks is not None:
            cbar.set_ticks(cbar_ticks)
            cbar.set_ticklabels([str(tick) for tick in cbar_ticks])
        cbar.outline.set_linewidth(1.6)
        cbar.ax.tick_params(width=1.5, length=5)

    out_dir = Path(out_dir)
    fig.savefig(out_dir / f"trajectory_{step:07d}.png", bbox_inches="tight")
    fig.savefig(out_dir / f"trajectory_{step:07d}.pdf", bbox_inches="tight")
    plt.close(fig)


def save_day6_overlay_figure(
    *,
    out_dir: str | Path,
    step: int,
    tensors: dict[str, torch.Tensor],
    title: str,
    embedding: str = "pca",
    seed: int = 0,
) -> None:
    try:
        plt = _setup_matplotlib()
        import seaborn as sns
    except ImportError:
        print("seaborn/matplotlib is not installed; skipping day-6 overlay plot")
        return

    gen, real = _projection(
        tensors["x_1_fwd"],
        tensors["x_1"],
        embedding=embedding,
        seed=seed,
    )
    x_label, y_label = _axis_labels(embedding)
    gen_arr = gen.numpy()
    real_arr = real.numpy()
    all_xy = torch.cat([gen, real], dim=0)
    lo, hi = _plot_limits(all_xy, pad_frac=0.10, robust=True)

    fig, ax = plt.subplots(1, 1, figsize=(8.8, 7.2))
    fig.patch.set_facecolor("white")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        sns.kdeplot(
            x=real_arr[:, 0],
            y=real_arr[:, 1],
            ax=ax,
            levels=9,
            fill=True,
            thresh=0.04,
            cmap="Greys",
            alpha=0.58,
            zorder=0,
        )
        sns.kdeplot(
            x=gen_arr[:, 0],
            y=gen_arr[:, 1],
            ax=ax,
            levels=9,
            fill=True,
            thresh=0.04,
            cmap="mako",
            alpha=0.46,
            zorder=1,
        )
        sns.kdeplot(
            x=real_arr[:, 0],
            y=real_arr[:, 1],
            ax=ax,
            levels=6,
            fill=False,
            thresh=0.06,
            color="#111827",
            linewidths=2.0,
            alpha=0.55,
            zorder=3,
        )
        sns.kdeplot(
            x=gen_arr[:, 0],
            y=gen_arr[:, 1],
            ax=ax,
            levels=6,
            fill=False,
            thresh=0.06,
            color="#0891b2",
            linewidths=2.2,
            alpha=0.72,
            zorder=4,
        )

    real_center = real_arr.mean(axis=0)
    gen_center = gen_arr.mean(axis=0)
    ax.scatter(
        [real_center[0]],
        [real_center[1]],
        s=180,
        marker="o",
        facecolor="#111827",
        edgecolor="white",
        linewidth=1.8,
        label="real day 6",
        zorder=5,
    )
    ax.scatter(
        [gen_center[0]],
        [gen_center[1]],
        s=210,
        marker="D",
        facecolor="#0891b2",
        edgecolor="white",
        linewidth=1.8,
        label="generated day 6",
        zorder=6,
    )
    ax.set_title(title, fontsize=25, fontweight="bold", pad=16)
    ax.set_xlim(float(lo[0]), float(hi[0]))
    ax.set_ylim(float(lo[1]), float(hi[1]))
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    _style_axis(ax)
    legend = ax.legend(loc="upper right", fontsize=14, frameon=True)
    legend.get_frame().set_linewidth(0)
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_alpha(0.88)

    out_dir = Path(out_dir)
    fig.savefig(out_dir / f"day6_overlay_{step:07d}.png", bbox_inches="tight")
    fig.savefig(out_dir / f"day6_overlay_{step:07d}.pdf", bbox_inches="tight")
    plt.close(fig)


def save_latent_eval_figure(
    *,
    out_dir: str | Path,
    step: int,
    tensors: dict[str, torch.Tensor],
    title: str,
    embedding: str = "pca",
    seed: int = 0,
) -> None:
    try:
        plt = _setup_matplotlib()
    except ImportError:
        print("matplotlib is not installed; skipping eval plot")
        return

    color = tensors["fate_label"].numpy()
    cmap, norm, cbar_ticks = _fate_scatter_style(color)
    panels = [
        ("Real day 2", tensors["x_0"]),
        ("Generated day 6", tensors["x_1_fwd"]),
        ("Cycle to day 2", tensors["x_0_cycle"]),
        ("Real day 6", tensors["x_1"]),
        ("Generated day 2", tensors["x_0_rev"]),
        ("Cycle to day 6", tensors["x_1_cycle"]),
    ]
    projected = _projection(
        *(tensor for _, tensor in panels),
        embedding=embedding,
        seed=seed,
    )
    x_label, y_label = _axis_labels(embedding)
    all_xy = torch.cat(projected, dim=0)
    lo = all_xy.min(dim=0).values
    hi = all_xy.max(dim=0).values
    pad = 0.06 * (hi - lo).clamp_min(1e-6)
    lo = lo - pad
    hi = hi + pad

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(18.4, 10.8),
        sharex=True,
        sharey=True,
    )
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(
        left=0.06, right=0.925, top=0.89, bottom=0.08, wspace=0.10, hspace=0.26
    )
    scatter = None
    for ax, (name, _tensor), xy_tensor in zip(axes.flatten(), panels, projected):
        xy = xy_tensor.numpy()
        scatter = ax.scatter(
            xy[:, 0],
            xy[:, 1],
            c=color,
            s=22,
            alpha=0.86,
            linewidths=0.25,
            edgecolors="white",
            cmap=cmap,
            norm=norm,
            rasterized=True,
        )
        ax.set_title(name)
        ax.set_xlim(float(lo[0]), float(hi[0]))
        ax.set_ylim(float(lo[1]), float(hi[1]))
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        _style_axis(ax)

    fig.suptitle(title, fontsize=22, fontweight="semibold", y=0.965)
    if scatter is not None:
        cbar = fig.colorbar(
            scatter,
            ax=axes,
            fraction=0.026,
            pad=0.012,
            ticks=cbar_ticks,
        )
        cbar.set_label("fate label")
        if cbar_ticks is not None:
            cbar.set_ticks(cbar_ticks)
            cbar.set_ticklabels([str(tick) for tick in cbar_ticks])
        cbar.outline.set_linewidth(1.2)
        cbar.ax.tick_params(width=1.1, length=4)

    out_dir = Path(out_dir)
    fig.savefig(out_dir / f"latent_eval_{step:07d}.png", bbox_inches="tight")
    fig.savefig(out_dir / f"latent_eval_{step:07d}.pdf", bbox_inches="tight")
    plt.close(fig)
