"""
Plotting / aggregation for editing_experiment.py results.

Kept deliberately separate from the experiment logic: this module only reads
results.json dicts and draws curves. Each run is one JSON file; comparing models
(the "baseline") is just plotting several JSONs together.

  # single run (also callable from editing_experiment.py via --plot)
  python editing_plot.py --results run.json --out run.png
  # overlay several runs
  python editing_plot.py --results a.json b.json c.json --out compare.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

# Distinct marker per overlaid run, cycled by index.
_MARKERS = ["o", "s", "^", "D", "v", "P", "*", "X", "<", ">"]


def _series(result: dict):
    """(fractions, fidelity_mean, fidelity_std, diversity) sorted by fraction."""
    items = sorted(result["by_fraction"].items(), key=lambda kv: float(kv[0]))
    fr = [float(k) for k, _ in items]
    fmean = [v["fidelity_mean"] for _, v in items]
    fstd = [v["fidelity_std"] for _, v in items]
    div = [v["diversity_mean_pairwise"] for _, v in items]
    return fr, fmean, fstd, div


def _convert_mode_name(mode_name: str) -> str:
    if "data2data" in mode_name:
        return "BIT"
    elif "n2d" in mode_name:
        return "Noise-to-Data"
    return mode_name

def _label(result: dict, include_ckpt: bool = False) -> str:
    m = result["meta"]
    ck = m.get("forward_ckpt") or m.get("reverse_ckpt") or "?"
    # Ground-truth references x variations per reference. Fall back to total
    # fidelity samples (n = num_images * num_variations) if meta is missing.
    n_ref = m.get("num_images")
    n_var = m.get("num_variations")
    if n_ref and n_var:
        n_str = f"{n_ref} ref \u00d7 {n_var} var"
    else:
        ns = {v["n"] for v in result["by_fraction"].values() if v.get("n")}
        n_str = f"n={ns.pop()}" if len(ns) == 1 else (f"n~{min(ns)}" if ns else "")
    tail = f" ({n_str})" if n_str else ""
    mode_name = _convert_mode_name(m['mode'])
    if include_ckpt:
        return f"{mode_name} ({Path(ck).parent.parent.name or Path(ck).name}){tail}"
    else:
        return f"{mode_name}{tail}"


def _plot(results: list[dict], out_path):
    plt.rcParams.update({
        'font.size': 12,          # base size for everything
        'axes.titlesize': 16,     # subplot titles
        'axes.labelsize': 12.5,     # x/y labels
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 18,   # suptitle
    })
    modality = results[0]["meta"]["generate_modality"]
    sim = "DINOv2" if modality == "image" else "Qwen3"
    fig, (ax_fid, ax_div) = plt.subplots(1, 2, figsize=(12, 4.5))
    for i, r in enumerate(results):
        fr, fmean, fstd, div = _series(r)
        marker = _MARKERS[i % len(_MARKERS)]
        ax_fid.errorbar(fr, fmean, yerr=fstd, marker=marker, capsize=3,
                        alpha=0.7, label=_label(r))
        ax_div.plot(fr, div, marker=marker, alpha=0.7, label=_label(r))
    ax_fid.set(xlabel="corruption fraction", ylabel=f"{sim} cosine to original",
               title="Fidelity (mean +/- std)")
    ax_div.set(xlabel="corruption fraction", ylabel="mean pairwise cosine",
               title="Diversity (lower = more diverse)")
    for ax in (ax_fid, ax_div):
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle(f"Cross-Modal Round-Trip Stochastic Variation ({modality})")
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[editing_plot] wrote {out_path}")


def save_edit_image_panel(original, noised, restored, out_path, title=None):
    """One-row panel: original | noised | restored_1 .. restored_K.

    ``original`` / ``noised`` are (3, H, W) uint8 tensors; ``restored`` is a list
    of (3, H, W) uint8 tensors. All plotting-only; the experiment just hands over
    already-decoded RGB.
    """
    components = [("original", original), ("noised", noised)]
    components += [(f"restored_{i + 1}", r) for i, r in enumerate(restored)]
    component_dir = Path(out_path).parent / Path(out_path).stem
    component_dir.mkdir(parents=True, exist_ok=True)
    for name, img in components:
        Image.fromarray(img.permute(1, 2, 0).numpy()).save(
            component_dir / f"{name}.png"
        )

    cols = [(name.replace("_", " "), img) for name, img in components]
    fig, axes = plt.subplots(1, len(cols), figsize=(2.4 * len(cols), 2.8))
    axes = [axes] if len(cols) == 1 else axes
    for ax, (name, img) in zip(axes, cols):
        ax.imshow(img.permute(1, 2, 0).numpy())
        ax.set_title(name, fontsize=9)
        ax.axis("off")
    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_edit_text_panel(original, noised, restored, out_path, title=None):
    """Text analog of the image panel: dump original / noised / restored captions."""
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    components = [("original", original), ("noised", noised)]
    components += [(f"restored_{i + 1}", text) for i, text in enumerate(restored)]
    component_dir = Path(out_path).parent / Path(out_path).stem
    component_dir.mkdir(parents=True, exist_ok=True)
    for name, text in components:
        (component_dir / f"{name}.txt").write_text(text + "\n")

    lines = ([title] if title else []) + [
        f"{name.replace('_', ' '):<11}: {text}" for name, text in components
    ]
    Path(out_path).write_text("\n".join(lines) + "\n")


def plot_single(result: dict, out_path):
    _plot([result], out_path)


def plot_many(json_paths: list[str], out_path):
    _plot([json.loads(Path(p).read_text()) for p in json_paths], out_path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", nargs="+", required=True)
    ap.add_argument("--out", default="editing_results/comparison.png")
    a = ap.parse_args()
    plot_many(a.results, a.out)
