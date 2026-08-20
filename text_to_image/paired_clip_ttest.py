#!/usr/bin/env python3
"""Run a paired t-test on two per-example CLIP score JSON files."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import ttest_rel


def load_scores(path: Path) -> dict[int, float]:
    data = json.loads(path.read_text())
    indices = data["indices"]
    scores = data["scores"]
    if len(indices) != len(scores):
        raise ValueError(f"{path}: indices and scores have different lengths")
    if len(indices) != len(set(indices)):
        raise ValueError(f"{path}: duplicate dataset indices")
    if not np.isfinite(scores).all():
        raise ValueError(f"{path}: scores contain non-finite values")
    return dict(zip(indices, scores))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method_a", type=Path)
    parser.add_argument("--method_b", type=Path)
    parser.add_argument(
        "--alternative",
        choices=("two-sided", "less", "greater"),
        default="two-sided",
        help="Alternative hypothesis for mean(method_a - method_b).",
    )
    args = parser.parse_args()

    scores_a = load_scores(args.method_a)
    scores_b = load_scores(args.method_b)
    if scores_a.keys() != scores_b.keys():
        only_a = len(scores_a.keys() - scores_b.keys())
        only_b = len(scores_b.keys() - scores_a.keys())
        raise ValueError(
            f"Dataset indices differ: {only_a} only in A, {only_b} only in B"
        )
    if len(scores_a) < 2:
        raise ValueError("A paired t-test requires at least two score pairs")

    indices = sorted(scores_a)
    a = np.asarray([scores_a[index] for index in indices], dtype=np.float64)
    b = np.asarray([scores_b[index] for index in indices], dtype=np.float64)
    result = ttest_rel(a, b, alternative=args.alternative)

    print(
        json.dumps(
            {
                "n": len(indices),
                "mean_a": a.mean(),
                "mean_b": b.mean(),
                "mean_difference_a_minus_b": (a - b).mean(),
                "std_difference_a_minus_b": (a - b).std(),
                "t_statistic": result.statistic,
                "p_value": result.pvalue,
                "alternative": args.alternative,
            },
            indent=2,
            default=float,
        )
    )


if __name__ == "__main__":
    main()
