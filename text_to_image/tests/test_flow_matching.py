"""CPU regression tests: PYTHONPATH=.:.. python -m unittest discover -s tests."""
import copy
from contextlib import nullcontext
import unittest
from unittest.mock import patch

import torch

from checkpoint_utils import build_sde, LoadedModel
from sde_utils.sde import FlowMatchingODE, SDE


class RecordingVelocity(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.value = torch.nn.Parameter(torch.tensor(0.25))
        self.calls = []

    def forward(self, x, t, y, x_cond, reverse=False, **kwargs):
        self.calls.append((x.detach().clone(), x_cond.detach().clone(), reverse, kwargs))
        return self.value.expand_as(x)


class FlowTests(unittest.TestCase):
    def test_zero_sigmas_preserve_original_values_and_rng(self):
        network = RecordingVelocity()
        flow = FlowMatchingODE(network, force_unconditional=True)
        x, y = torch.randn(4, 3), torch.zeros(4, dtype=torch.long)
        rng = torch.get_rng_state().clone()
        a, b = flow.perturb_endpoints(x, x)
        self.assertIs(a, x)
        self.assertIs(b, x)
        for reverse in (False, True):
            actual = flow.simulate(x, 5, reverse=reverse, y=y)
            expected = SDE.simulate(flow, x, 5, reverse=reverse, y=y)
            self.assertTrue(torch.equal(actual, expected))
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertTrue(all(not call[2] and not call[3]["cond_mask"].any() for call in network.calls))

    def test_stochastic_source_clean_condition_and_signed_velocity(self):
        for reverse, sigma in ((False, 0.01), (True, 0.02)):
            network = RecordingVelocity()
            flow = FlowMatchingODE(network, text_sigma=0.01, image_sigma=0.02)
            clean = torch.zeros(20000, 1)
            torch.manual_seed(123)
            with patch("torch.randn_like", wraps=torch.randn_like) as draw:
                out = flow.simulate(clean, 4, reverse=reverse, y=torch.zeros(len(clean), dtype=torch.long))
            self.assertEqual(draw.call_count, 1)
            residual = out - (-0.25 if reverse else 0.25)
            self.assertAlmostEqual(residual.std().item(), sigma, delta=sigma * 0.03)
            self.assertTrue(all(torch.equal(call[1], clean) and call[2] == reverse for call in network.calls))
            other = flow.simulate(clean, 4, reverse=reverse, y=torch.zeros(len(clean), dtype=torch.long))
            self.assertFalse(torch.equal(out, other))

    def test_checkpoint_defaults_and_direction_support(self):
        old = {"sde": "flow_matching", "force_unconditional": True, "no_reverse": True}
        flow = build_sde(old)
        self.assertEqual((flow.text_sigma, flow.image_sigma), (0, 0))
        loaded = LoadedModel(None, old, None, None, flow)
        self.assertTrue(loaded.generates("text"))
        new = {**old, "force_unconditional": False, "flow_text_sigma": 0.01}
        flow = build_sde(new)
        self.assertEqual(flow.text_sigma, 0.01)
        self.assertFalse(LoadedModel(None, new, None, None, flow).generates("text"))
        for bad in (-1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                FlowMatchingODE(None, image_sigma=bad)

    def test_training_uses_noisy_targets_and_clean_decoder_input(self):
        import train
        from models.token_decoder import SharedTokenDecoder
        from token_bridge import DEFAULT_TOKEN_BRIDGE_CONFIG as bc, prepare_bridge_batch
        net = RecordingVelocity()
        ema = copy.deepcopy(net)
        flow = FlowMatchingODE(net, text_sigma=0.01, image_sigma=0.02)
        decoder = SharedTokenDecoder(8, hidden_dim=8)
        opt = torch.optim.SGD(net.parameters(), lr=0.01)
        dec_opt = torch.optim.SGD(decoder.parameters(), lr=0.01)
        batch = {"latent": torch.randn(2, *bc.bridge_shape),
                 "text_token_emb": torch.randn(2, bc.token_flat_dim),
                 "prompt_kind_label": torch.zeros(2, dtype=torch.long),
                 "text_token_ids": torch.zeros(2, bc.token_seq_len, dtype=torch.long),
                 "text_token_mask": torch.ones(2, bc.token_seq_len, dtype=torch.bool)}
        clean0, clean1, _, _, _ = prepare_bridge_batch(batch, torch.device("cpu"), use_token_text_bridge=True)
        noisy = flow.perturb_endpoints(clean0, clean1)
        with patch.object(flow, "perturb_endpoints", return_value=noisy), \
             patch.object(train, "autocast", return_value=nullcontext()), \
             patch.object(train, "token_decoder_loss", wraps=train.token_decoder_loss) as decode:
            logs = train.train_step(
                model=net, ema=ema, sde=flow, optimizer=opt,
                scheduler=torch.optim.lr_scheduler.LambdaLR(opt, lambda _: 1),
                batch=batch, device=torch.device("cpu"), train_forward=True, train_reverse=True,
                eps=0, grad_clip=1, ema_decay=0.9, unconditional_percent=0,
                use_token_text_bridge=True, token_layout="row_major", x0_cond_source="x0",
                token_decoder=decoder, token_decoder_optimizer=dec_opt,
                token_decoder_noise_t_max=0, token_decoder_pad_zero_prob=0, bridge_config=bc,
            )
        expected_loss = ((noisy[1] - noisy[0] - 0.25) ** 2).mean().item()
        self.assertAlmostEqual(logs["loss/forward"], expected_loss, places=6)
        self.assertAlmostEqual(logs["loss/reverse"], expected_loss, places=6)
        self.assertTrue(torch.equal(net.calls[0][1], clean0))
        self.assertTrue(torch.equal(net.calls[1][1], clean1))
        self.assertTrue(torch.equal(decode.call_args.kwargs["x_0"], clean0))
        self.assertNotEqual(net.value.item(), 0.25)


if __name__ == "__main__":
    unittest.main()
