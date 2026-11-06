"""
Compose a Figure-5-style image-editing grid from ``perturbation_slerp_experiment.py``
outputs (``<run-dir>/rows/img####/{col*.png,source.png,meta.json}``).

Each figure row is one source image; each column is one spherical-interpolation
coordinate lambda (0 = text inferred from the image, 1 = LLM-edited text), with
the caption decoded at that coordinate printed above the image, as in the
paper's Figure 5. Only numpy / Pillow / matplotlib are needed.

  # rows = dataset indices from the panel_img####.png / rows/img#### names
  python editing_figure.py --run-dir RUN --images 3 17 22 40 --out fig5_cfg0.5.pdf
  # same, with the VAE-decoded source image as an extra leftmost column
  python editing_figure.py --run-dir RUN --images 3 17 22 40 --show-source --out fig.png
  # thumbnails of every completed row, to choose --images
  python editing_figure.py --run-dir RUN --contact-sheet contact.png
"""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

import numpy as np
from PIL import Image


def load_row(run_dir: Path, index: int) -> dict:
    row_dir = Path(run_dir) / "rows" / f"img{index:04d}"
    meta_path = row_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"{meta_path} not found; was image {index} part of this run?")
    meta = json.loads(meta_path.read_text())
    n = int(meta["num_columns"])
    columns = [np.asarray(Image.open(row_dir / f"col{j}.png").convert("RGB")) for j in range(n)]
    source_path = row_dir / "source.png"
    source = np.asarray(Image.open(source_path).convert("RGB")) if source_path.exists() else None
    return {"meta": meta, "columns": columns, "source": source}


def completed_indices(run_dir: Path) -> list[int]:
    manifest = Path(run_dir) / "manifest.json"
    if manifest.exists():
        data = json.loads(manifest.read_text())
        if data.get("complete"):
            return [int(i) for i in data["completed_indices"]]
    # Partial run: fall back to whatever rows were written.
    return sorted(int(p.name[3:]) for p in (Path(run_dir) / "rows").glob("img*") if (p / "meta.json").exists())


def _wrap(text: str, width: int) -> list[str]:
    return textwrap.wrap(text or "(empty)", width=width) or ["(empty)"]


def compose(rows: list[dict], *, show_source: bool = False, columns: list[int] | None = None,
            panel_in: float = 1.25, gap_in: float = 0.06, fontsize: float = 5.0,
            wrap: int = 30, title: str | None = None):
    """Build the figure. Panel geometry is fixed in inches so that the caption
    band of each row grows with its longest wrapped caption, as in Figure 5."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not rows:
        raise ValueError("no rows to plot")
    ncols_all = len(rows[0]["columns"])
    columns = list(range(ncols_all)) if columns is None else columns
    for row in rows:
        if len(row["columns"]) != ncols_all:
            raise ValueError("rows have different numbers of interpolation columns")
    line_in = fontsize * 1.25 / 72.0
    cells = []  # per row: list of (image, caption lines)
    for row in rows:
        caps = row["meta"]["column_captions"]
        cell = [(row["columns"][j], _wrap(caps[j], wrap)) for j in columns]
        if show_source:
            if row["source"] is None:
                raise ValueError("--show-source requested but source.png is missing")
            cell.insert(0, (row["source"], ["Source image"]))
        cells.append(cell)
    ncols = len(cells[0])
    band = [max(len(lines) for _, lines in cell) * line_in + 2 * gap_in for cell in cells]
    title_in = 0.3 if title else 0.0
    width = ncols * panel_in + (ncols + 1) * gap_in
    height = title_in + sum(b + panel_in + 2 * gap_in for b in band) + gap_in
    fig = plt.figure(figsize=(width, height))
    if title:
        fig.text(0.5, 1 - 0.5 * title_in / height, title, ha="center", va="center", fontsize=fontsize + 3)
    top = height - title_in - gap_in
    for cell, b in zip(cells, band):
        img_top = top - b
        for c, (image, lines) in enumerate(cell):
            left = gap_in + c * (panel_in + gap_in)
            ax = fig.add_axes([left / width, (img_top - panel_in) / height, panel_in / width, panel_in / height])
            ax.imshow(image)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.text((left + panel_in / 2) / width, (img_top + 0.02) / height, "\n".join(lines),
                     ha="center", va="bottom", fontsize=fontsize, linespacing=1.1)
        top = img_top - panel_in - 2 * gap_in
    return fig


def contact_sheet(run_dir: Path, out: Path, per_row: int = 8, thumb: int = 96) -> None:
    """Grid of (lambda=0 | lambda=1) thumbnail pairs labeled by dataset index."""
    from PIL import ImageDraw
    indices = completed_indices(run_dir)
    if not indices:
        raise ValueError(f"no completed rows under {run_dir}")
    pad, label = 4, 14
    cell_w, cell_h = 2 * thumb + pad, thumb + label
    nrows = (len(indices) + per_row - 1) // per_row
    sheet = Image.new("RGB", (per_row * (cell_w + pad) + pad, nrows * (cell_h + pad) + pad), "white")
    draw = ImageDraw.Draw(sheet)
    for k, index in enumerate(indices):
        row = load_row(run_dir, index)
        x = pad + (k % per_row) * (cell_w + pad)
        y = pad + (k // per_row) * (cell_h + pad)
        for m, image in enumerate((row["columns"][0], row["columns"][-1])):
            sheet.paste(Image.fromarray(image).resize((thumb, thumb)), (x + m * (thumb + pad), y + label))
        draw.text((x, y), f"{index}", fill="black")
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--images", type=int, nargs="+", default=None,
                   help="Dataset indices of the rows, top to bottom (default: first four completed).")
    p.add_argument("--num-rows", type=int, default=4, help="Rows when --images is omitted.")
    p.add_argument("--columns", type=int, nargs="+", default=None,
                   help="Interpolation columns to show (default: all).")
    p.add_argument("--show-source", action="store_true",
                   help="Prepend the VAE-decoded source image as a column (not in the paper's Figure 5).")
    p.add_argument("--out", type=Path, nargs="+", default=None, help="Output file(s), e.g. fig.pdf fig.png.")
    p.add_argument("--title", default=None)
    p.add_argument("--fontsize", type=float, default=5.0)
    p.add_argument("--wrap", type=int, default=30, help="Caption wrap width (characters).")
    p.add_argument("--panel-inches", type=float, default=1.25)
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--contact-sheet", type=Path, default=None,
                   help="Also write thumbnails of every completed row (helps choose --images).")
    args = p.parse_args(argv)

    if args.contact_sheet is not None:
        contact_sheet(args.run_dir, args.contact_sheet)
        print(f"[editing-figure] wrote {args.contact_sheet}")
    if args.out is None:
        if args.contact_sheet is None:
            p.error("pass --out and/or --contact-sheet")
        return
    indices = args.images or completed_indices(args.run_dir)[:args.num_rows]
    rows = [load_row(args.run_dir, i) for i in indices]
    fig = compose(rows, show_source=args.show_source, columns=args.columns,
                  panel_in=args.panel_inches, fontsize=args.fontsize, wrap=args.wrap, title=args.title)
    for out in args.out:
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=args.dpi)
        print(f"[editing-figure] wrote {out} (rows={indices})")


if __name__ == "__main__":
    main()
