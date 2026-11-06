"""CFG noise backsolving (Alg. 2 / Eq. 61) and editing-figure helpers.

Run from text_to_image with PYTHONPATH=.:.. ; needs torch, numpy, Pillow, matplotlib.
wandb / torchvision are stubbed when absent because only pure helpers are tested.
"""

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

for _name in ("wandb", "torchvision"):
    try:
        __import__(_name)
    except ImportError:
        sys.modules[_name] = types.ModuleType(_name)

from perturbation_slerp_experiment import (  # noqa: E402
    forward_score, infer_noise_samples, save_row_data, simulate_with_fixed_noise,
)
from sde_utils.sde import PeriodicVolatilitySDE  # noqa: E402


class _GuidableNet(torch.nn.Module):
    """Deterministic stand-in with the DiTXA calling convention."""

    def __init__(self):
        super().__init__()
        self.calls = {"plain": 0, "cfg": 0}

    def forward(self, x, t, y, x_cond, reverse=False, cond_mask=None):
        t = t.view(-1, 1, 1, 1)
        uncond = cond_mask is not None and not bool(cond_mask.any())
        context = torch.zeros_like(x_cond) if uncond else x_cond
        return torch.tanh(x) * (1 + t) - 0.3 * context + 0.1 * torch.sin(3 * x * t)

    def __call__(self, *args, **kwargs):
        self.calls["plain"] += 1
        return super().__call__(*args, **kwargs)

    def forward_with_cfg(self, x, t, y, x_cond, reverse=False, cfg_scale=0.0):
        self.calls["cfg"] += 1
        cond = self.forward(x, t, y, x_cond, reverse=reverse)
        uncond = self.forward(x, t, y, x_cond, reverse=reverse,
                              cond_mask=torch.zeros(x.shape[0], dtype=torch.bool))
        return cond + cfg_scale * (cond - uncond)


def _model():
    sde = PeriodicVolatilitySDE(alpha=0.95, k=1.0, eps=0.05, score_network=None)
    return SimpleNamespace(net=_GuidableNet(), sde=sde)


class CfgBacksolveTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.fwd = _model()
        self.x0 = torch.randn(3, 4, 8, 8, dtype=torch.float64)
        self.x1 = torch.randn(3, 4, 8, 8, dtype=torch.float64)
        self.y = torch.zeros(3, dtype=torch.long)

    def test_zero_scale_keeps_the_original_unguided_call(self):
        t = torch.full((3,), 0.5, dtype=torch.float64)
        forward_score(self.fwd, self.x0, t, self.y, self.x0, cfg_scale=0.0)
        self.assertEqual(self.fwd.net.calls, {"plain": 1, "cfg": 0})
        forward_score(self.fwd, self.x0, t, self.y, self.x0, cfg_scale=0.5)
        self.assertEqual(self.fwd.net.calls["cfg"], 1)

    def test_guided_replay_at_lambda_zero_recovers_the_source(self):
        for cfg in (0.0, 0.5):
            dB = infer_noise_samples(self.x0, self.x1, self.y, 40, self.fwd, "cpu", cfg_scale=cfg)
            out = simulate_with_fixed_noise(self.x0, dB, self.y, 40, self.fwd, "cpu", cfg_scale=cfg)
            torch.testing.assert_close(out, self.x1, rtol=1e-8, atol=1e-8)

    def test_backsolve_and_replay_must_share_the_guidance(self):
        dB = infer_noise_samples(self.x0, self.x1, self.y, 40, self.fwd, "cpu", cfg_scale=0.5)
        out = simulate_with_fixed_noise(self.x0, dB, self.y, 40, self.fwd, "cpu", cfg_scale=0.0)
        self.assertGreater(float((out - self.x1).abs().max()), 1e-3)


class RowDataTests(unittest.TestCase):
    def test_saved_rows_compose_into_a_figure(self):
        from editing_figure import completed_indices, compose, load_row
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            images = torch.randint(0, 255, (5, 3, 16, 16), dtype=torch.uint8)
            source = torch.randint(0, 255, (3, 16, 16), dtype=torch.uint8)
            for index in (7, 2):
                save_row_data(run / "rows" / f"img{index:04d}", images,
                              [f"caption {j}" for j in range(5)], source, meta={"column_captions": [f"caption {j}" for j in range(5)]})
            self.assertEqual(completed_indices(run), [2, 7])
            (run / "manifest.json").write_text(json.dumps({"complete": True, "completed_indices": [7, 2]}))
            self.assertEqual(completed_indices(run), [7, 2])
            row = load_row(run, 7)
            self.assertEqual(len(row["columns"]), 5)
            self.assertTrue((row["columns"][1] == images[1].permute(1, 2, 0).numpy()).all())
            fig = compose([load_row(run, 7), load_row(run, 2)], show_source=True)
            self.assertEqual(len(fig.axes), 12)


if __name__ == "__main__":
    unittest.main()
