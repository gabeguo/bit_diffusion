"""
Plotting / aggregation for editing_experiment.py results.

Kept deliberately separate from the experiment logic: this module only reads
results.json dicts and draws curves. Each run is one JSON file; comparing models
(the "baseline") is just plotting several JSONs together.

  # single run (also callable from editing_experiment.py via --plot)
  python editing_plot.py --results run.json --out run.png
  # overlay several runs (one modality), e.g. Fig. 3 plus a guided BIT run
  python editing_plot.py --results bit_cfg0.json bit_cfg0.5.json flow.json n2d.json --out compare.png
  # explicit legend entries, one per file
  python editing_plot.py --results ... --labels "BIT (CFG=0)" "BIT (CFG=0.5)" "Flow Matching" "Noise-to-Data"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

# Fixed style per method, matching the paper's Fig. 3 (BIT blue circles,
# noise-to-data orange squares, flow green triangles); guided BIT is distinct.
_STYLE = {
    "bit": ("C0", "o"),
    "bit_cfg": ("C3", "D"),
    "n2d": ("C1", "s"),
    "flow": ("C2", "^"),
    "n2d_cond": ("C4", "v"),
}
_EXTRA = [("C5", "P"), ("C6", "*"), ("C7", "X"), ("C8", "<"), ("C9", ">")]
_NAMES = {"bit": "BIT", "flow": "Flow Matching", "n2d": "Noise-to-Data",
          "n2d_cond": "Noise-to-Data (inferred cond.)"}


def _series(result: dict):
    """(fractions, fidelity_mean, fidelity_std, diversity) sorted by fraction."""
    items = sorted(result["by_fraction"].items(), key=lambda kv: float(kv[0]))
    fr = [float(k) for k, _ in items]
    fmean = [v["fidelity_mean"] for _, v in items]
    fstd = [v["fidelity_std"] for _, v in items]
    div = [v["diversity_mean_pairwise"] for _, v in items]
    return fr, fmean, fstd, div


def _method(result: dict) -> str:
    """bit / flow / n2d / n2d_cond. A single-field flow round trip is stored as a
    data2data mode; results.json does not record the transport, so a flow run is
    recognized by an explicit meta["sde"] or by "flow" in its checkpoint path.
    Use --labels if that heuristic does not fit your paths."""
    m = result["meta"]
    mode = m["mode"]
    if mode == "n2d_two":
        return "n2d_cond"
    if "n2d" in mode:
        return "n2d"
    ckpts = " ".join(str(m.get(k) or "") for k in ("forward_ckpt", "reverse_ckpt")).lower()
    if m.get("sde") == "flow_matching" or (m.get("sde") is None and "flow" in ckpts):
        return "flow"
    return "bit"


def _cfg(result: dict) -> float:
    return float(result["meta"].get("cfg_scale") or 0.0)


def _counts(result: dict) -> str:
    m = result["meta"]
    n_ref, n_var = m.get("num_images"), m.get("num_variations")
    if n_ref and n_var:
        return f"{n_ref} ref × {n_var} var"
    ns = {v["n"] for v in result["by_fraction"].values() if v.get("n")}
    return f"n={ns.pop()}" if len(ns) == 1 else (f"n~{min(ns)}" if ns else "")


def _label(result: dict, show_cfg: bool = False, counts: bool = True) -> str:
    """e.g. "BIT (CFG=0.5; 500 ref x 8 var)". CFG is printed for BIT whenever
    ``show_cfg`` (any overlaid BIT run is guided) or the run itself is guided."""
    method = _method(result)
    parts = []
    if method == "bit" and (show_cfg or _cfg(result) > 0):
        parts.append(f"CFG={_cfg(result):g}")
    elif method == "n2d_cond" and _cfg(result) > 0:
        parts.append(f"CFG={_cfg(result):g}")
    if counts and _counts(result):
        parts.append(_counts(result))
    name = _NAMES[method]
    return f"{name} ({'; '.join(parts)})" if parts else name


def _styles(results: list[dict]) -> list[tuple[str, str]]:
    """Fixed color/marker per method; repeated methods get unused extras."""
    used, extra, out = set(), iter(_EXTRA), []
    for r in results:
        key = _method(r)
        if key == "bit" and _cfg(r) > 0:
            key = "bit_cfg"
        style = _STYLE[key] if key not in used else next(extra, ("k", "o"))
        used.add(key)
        out.append(style)
    return out


def _plot(results: list[dict], out_path, labels: list[str] | None = None, counts: bool = True):
    plt.rcParams.update({
        'font.size': 12,          # base size for everything
        'axes.titlesize': 16,     # subplot titles
        'axes.labelsize': 12.5,     # x/y labels
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 18,   # suptitle
    })
    modalities = {r["meta"]["generate_modality"] for r in results}
    if len(modalities) != 1:
        raise ValueError(f"cannot overlay different modalities: {sorted(modalities)}")
    if labels is not None and len(labels) != len(results):
        raise ValueError(f"got {len(labels)} labels for {len(results)} results")
    modality = modalities.pop()
    sim = "DINOv2" if modality == "image" else "Qwen3"
    show_cfg = any(_method(r) == "bit" and _cfg(r) > 0 for r in results)
    fig, (ax_fid, ax_div) = plt.subplots(1, 2, figsize=(12, 4.5))
    for i, (r, (color, marker)) in enumerate(zip(results, _styles(results))):
        fr, fmean, fstd, div = _series(r)
        label = labels[i] if labels is not None else _label(r, show_cfg=show_cfg, counts=counts)
        ax_fid.errorbar(fr, fmean, yerr=fstd, marker=marker, color=color, capsize=3,
                        alpha=0.7, label=label)
        ax_div.plot(fr, div, marker=marker, color=color, alpha=0.7, label=label)
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


def plot_many(json_paths: list[str], out_path, labels: list[str] | None = None, counts: bool = True):
    _plot([json.loads(Path(p).read_text()) for p in json_paths], out_path, labels=labels, counts=counts)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", nargs="+", required=True,
                    help="results.json files of ONE modality, plotted in this order.")
    ap.add_argument("--labels", nargs="+", default=None,
                    help="Legend entries, one per --results file (overrides automatic labels).")
    ap.add_argument("--no-counts", action="store_true",
                    help="Omit the '500 ref x 8 var' sample counts from automatic labels.")
    ap.add_argument("--out", default="editing_results/comparison.png")
    a = ap.parse_args()
    plot_many(a.results, a.out, labels=a.labels, counts=not a.no_counts)
