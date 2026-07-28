"""
Cross-modal round-trip editing experiment (SDEdit-style) for the text<->image bridge.

Start from real data on one endpoint, walk PARTWAY toward the other endpoint
(stopping at bridge time ``t``), then walk back to the source modality to produce
a *variation*. Sweep how far we walk (corruption fraction) and measure how much
content survives (DINOv2 cosine for images, Qwen3-Embedding cosine for text).

Model kinds are read from each checkpoint's stored args:
  * data-to-data (bridge): text_as_noise == image_as_noise == False.
  * noise-to-data         : text_as_noise or image_as_noise True (the far end is
                            noise; the other modality is cross-attention context).

Restoration logic (see the module functions for the "why" of each choice):
  * data-to-data : simulate source->other CONDITIONAL on the source endpoint down
                   to t, then simulate back UNCONDITIONAL (a variation, not a copy).
  * n2d, 1 model : analytic-noise the data, restore UNCONDITIONAL (no way to build
                   the cross-attention context from the source alone).
  * n2d, 2 models: infer the other modality with the opposite model (optionally
                   snapped to the token vocab table), analytic-noise the data,
                   restore CONDITIONAL on the inferred context.

Writes one self-contained results.json. Cross-run comparison / plotting lives in
editing_plot.py (the "baseline" is simply a second run with a different model).

Run from the text_to_image dir with PYTHONPATH=.:.. (same as train.py).
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import wandb
from tqdm import tqdm

from data_utils.latent_dataset import CommonCatalogLatentDataset
from eval_plot import decode_latents, _temporarily_swap_score_network
from models.dit import DiT_models
from models.token_decoder import SharedTokenDecoder
from sde_utils.loss import sample_p_base_x_t_cond_x_0_x_1
from sde_utils.sde import (
    CosineDecayingVolatilitySDE,
    FlowMatchingODE,
    PeriodicVolatilitySDE,
    UniformVolatilitySDE,
)
from token_bridge import (
    PROMPT_NUM_CLASSES,
    bridge_config_from_manifest,
    bridge_to_token_flat,
    norm_based_token_stops,
    token_flat_to_bridge,
)
from checkpoint_utils import _resolve_ckpt

# The SDE parameters that must agree between two paired data-to-data models: the
# intermediate x_t only means the same thing under one noising kernel.
_SDE_KEYS = ("sde", "K", "periodic_sde_alpha", "periodic_sde_k", "periodic_sde_eps")


def _transport_signature(args: dict) -> tuple:
    if args["sde"] == "flow_matching":
        return ("flow_matching", args["force_unconditional"])
    return tuple(args[key] for key in _SDE_KEYS)


def _is_single_field_flow(args: dict) -> bool:
    return args["sde"] == "flow_matching" and args["force_unconditional"]


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

@dataclass
class LoadedModel:
    net: nn.Module
    args: dict
    token_decoder: Optional[nn.Module]
    tokenizer: object
    sde: object  # this model's own SDE (score_network swapped in per use)

    @property
    def is_noise2data(self) -> bool:
        return bool(self.args.get("text_as_noise") or self.args.get("image_as_noise"))

    def generates(self, modality: str) -> bool:
        """Whether this model produces ``modality`` as (non-noise) output."""
        if modality == "image":
            return not self.args.get("image_as_noise") and not self.args.get("no_forward")
        return not self.args.get("text_as_noise") and (
            not self.args.get("no_reverse") or _is_single_field_flow(self.args)
        )


def build_sde(args: dict):
    kind = args["sde"]
    if kind == "uniform":
        return UniformVolatilitySDE(A=0, K=args["K"], score_network=None)
    if kind == "periodic":
        return PeriodicVolatilitySDE(
            alpha=args["periodic_sde_alpha"], k=args["periodic_sde_k"],
            eps=args["periodic_sde_eps"], score_network=None,
        )
    if kind == "cosine_decay":
        return CosineDecayingVolatilitySDE(
            alpha=args["periodic_sde_alpha"], eps=args["periodic_sde_eps"],
            score_network=None,
        )
    if kind == "flow_matching":
        return FlowMatchingODE(
            score_network=None,
            force_unconditional=args["force_unconditional"],
        )
    raise ValueError(f"unknown sde {kind!r}")


def load_model(ckpt_path: str, data_root: str, device: torch.device) -> LoadedModel:
    """Rebuild a model + (optional) token decoder from a training checkpoint,
    loading the EMA weights (what training-time eval uses)."""
    # mmap keeps the (large) optimizer / alt-EMA tensors on disk; we only copy
    # the EMA weights + token decoder into the model.
    ckpt = torch.load(_resolve_ckpt(ckpt_path), map_location="cpu", weights_only=False, mmap=True)
    a = ckpt["args"]
    assert not a.get("edm_precond", False), "EDM-preconditioned checkpoints are unsupported."
    assert "XA" in a["model"], "cross-attention model required"

    assert a.get("token_layout") == "row_major", "only row_major token layout is supported"

    runtime = bridge_config_from_manifest(data_root, preset=a.get("bridge_preset", "auto"))
    bc = runtime.bridge
    # Build WITHOUT the REPA heads: they are training-only and their dims depend
    # on sidecars we don't have here. We drop those keys when loading (below).
    net = DiT_models[a["model"]](
        input_size=bc.height, in_channels=bc.channels,
        num_classes=PROMPT_NUM_CLASSES if a["use_token_text_bridge"] else a["num_classes"],
        class_dropout_prob=a["prompt_kind_dropout"] if a["use_token_text_bridge"] else 0.0,
        forward_cond_scale=a["forward_cond_scale"],
    ).to(device).eval()
    missing, unexpected = net.load_state_dict(ckpt["ema"], strict=False)
    assert not missing, f"missing keys loading {ckpt_path}: {missing}"
    assert all("repa" in k for k in unexpected), f"unexpected keys: {unexpected}"

    token_decoder, tokenizer = None, None
    if "token_decoder" in ckpt:
        from transformers import AutoTokenizer
        tcfg = json.loads((Path(data_root) / "token_embed_config.json").read_text())["config"]
        tokenizer = AutoTokenizer.from_pretrained(tcfg["text_model"])
        token_decoder = SharedTokenDecoder(
            vocab_size=len(tokenizer),
            hidden_dim=a["token_decoder_hidden_dim"],
            token_seq_len=bc.token_seq_len, token_emb_dim=bc.token_emb_dim,
        ).to(device).eval()
        token_decoder.load_state_dict(ckpt["token_decoder"])
    return LoadedModel(net=net, args=a, token_decoder=token_decoder,
                       tokenizer=tokenizer, sde=build_sde(a))


def resolve_mode(fwd, rev, modality: str) -> str:
    """Validate the model set for ``modality`` and return the experiment mode.

    forward model == text->image; reverse model == image->text.
    """
    present = [m for m in (fwd, rev) if m is not None]
    assert present, "pass at least one of --forward-ckpt / --reverse-ckpt"
    n2d = present[0].is_noise2data
    assert all(m.is_noise2data == n2d for m in present), "cannot mix noise2data and data2data"

    if not n2d:  # data-to-data bridge
        if len(present) == 1:
            m = present[0]
            assert m.generates("image") and m.generates("text"), (
                "a single data-to-data model must support BOTH directions"
            )
            return "data2data_single"
        assert fwd.generates("image") and rev.generates("text"), (
            "the paired data-to-data models must support their inference directions"
        )
        assert _transport_signature(fwd.args) == _transport_signature(rev.args), (
            "paired data-to-data models must share transport parameters"
        )
        return "data2data_two"

    # noise-to-data
    if len(present) == 1:
        assert present[0].generates(modality), (
            f"the single noise-to-data model must generate {modality!r}"
        )
        assert (fwd if modality == "image" else rev) is present[0], (
            f"{modality!r} restoration needs the "
            f"{'forward (text->image)' if modality == 'image' else 'reverse (image->text)'} model"
        )
        return "n2d_single"
    assert fwd is not None and rev is not None
    assert fwd.args.get("text_as_noise") and rev.args.get("image_as_noise") and fwd.args.get("no_reverse") and rev.args.get("no_forward")
    return "n2d_two"


# ---------------------------------------------------------------------------
# Bridge traversal primitives (thin reuse of sde.dX_t)
# ---------------------------------------------------------------------------

class _Uncond(nn.Module):
    """Force the cross-attention conditioning off (the unconditional score)."""

    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net

    def forward(self, x, t, y, x_cond, reverse=False, cond_mask=None):
        cond_mask = torch.zeros((x.shape[0],), dtype=torch.bool, device=x.device)
        return self.net(x, t, y, x_cond, reverse=reverse, cond_mask=cond_mask)


@torch.no_grad()
def _integrate(sde, x, t_from, t_to, nfe, *, reverse, x_cond, y, cfg_scale=0.0):
    """Euler-Maruyama over [t_from, t_to] reusing ``sde.dX_t`` (stochastic)."""
    if nfe <= 0 or float(t_from) == float(t_to):  # degenerate (e.g. fraction 0)
        return x
    assert (t_to < t_from) == reverse, "reverse must match a decreasing t schedule"
    ts = torch.linspace(float(t_from), float(t_to), nfe + 1, device=x.device)
    for i in range(nfe):
        t = ts[i:i + 1].expand(x.shape[0])
        dt = float((ts[i + 1] - ts[i]).abs())
        x = x + sde.dX_t(x_t=x, t=t, x_cond=x_cond, y=y, dt=dt,
                         reverse=reverse, cfg_scale=cfg_scale)
    return x


def _analytic_noise(sde, data, data_t, t_stop):
    """Sample x_t from the bridge kernel with the far (noise) endpoint drawn from
    the prior -- the closed-form forward process of a noise-to-data model."""
    noise = torch.randn_like(data)
    x0, x1 = (noise, data) if data_t == 1.0 else (data, noise)
    t = torch.full((data.shape[0],), float(t_stop), device=data.device)
    return sample_p_base_x_t_cond_x_0_x_1(sde=sde, x_0=x0, x_1=x1, t=t)


def _snap_to_vocab(bridge, token_decoder, vocab_table, token_scale, token_emb_dim, config):
    """Decode inferred text tokens to ids and re-embed via the vocab table,
    projecting an off-manifold inferred text back onto real token embeddings."""
    token_flat = bridge_to_token_flat(bridge.float(), config=config)
    ids = token_decoder(token_flat).argmax(dim=-1)                       # (B, S)
    stops, _ = norm_based_token_stops(bridge.float(), token_scale=token_scale, config=config)
    keep = torch.arange(ids.shape[1], device=ids.device)[None, :] < stops[:, None]
    emb = vocab_table[ids][..., :token_emb_dim].float()                  # (B, S, D)
    assert emb.shape == (ids.shape[0], ids.shape[1], token_emb_dim), f"emb.shape {emb.shape} is not {ids.shape[0], ids.shape[1], token_emb_dim}"
    assert keep.shape == (ids.shape[0], ids.shape[1]) == ids.shape, f"keep.shape {keep.shape} is not {ids.shape[0], ids.shape[1]}"
    emb = F.normalize(emb, dim=-1) * keep[..., None] * token_scale
    return token_flat_to_bridge(emb.reshape(ids.shape[0], -1), config=config)


# ---------------------------------------------------------------------------
# Restoration strategies (each returns V restored bridge tensors in source space)
# ---------------------------------------------------------------------------

def _run(model, net, x, t_from, t_to, nfe, *, reverse, x_cond, y, cfg_scale=0.0):
    """Integrate ``net`` under ITS OWN model's SDE over [t_from, t_to]."""
    assert reverse == (t_from > t_to) or (t_from == t_to), f"reverse {reverse} must match t_from {t_from} > t_to {t_to}"
    with _temporarily_swap_score_network(model.sde, net):
        return _integrate(sde=model.sde, x=x, t_from=t_from, t_to=t_to, nfe=nfe,
                          reverse=reverse, x_cond=x_cond, y=y, cfg_scale=cfg_scale)


def _restore_data2data(fwd, rev, modality, src_img, src_text, y, t_stop, ctx, cached_sample=None, cached_time=None):
    # fwd/rev share SDE params here, but we still route each leg through its own
    # model's SDE for clarity. Noising is conditional; restoration unconditional.
    # Returns (noised, restored): the intermediate state and the final variation.
    if modality == "image":  # source at t=1; noise toward text, restore toward image
        if cached_time is None:
            start_img = src_img
            start_time = 1.0
        else:
            assert cached_sample is not None
            start_img = cached_sample
            start_time = cached_time
        x_t = _run(rev, rev.net, start_img, start_time, t_stop, ctx.noise_nfe,
                   reverse=True, x_cond=src_img, y=y, cfg_scale=ctx.cfg_scale)
        return x_t, _run(fwd, _Uncond(fwd.net), x_t, t_stop, 1.0, ctx.restore_nfe,
                         reverse=False, x_cond=torch.zeros_like(x_t), y=y)
    elif modality == "text":
        if cached_time is None:
            start_text = src_text
            start_time = 0.0
        else:
            assert cached_sample is not None
            start_text = cached_sample
            start_time = cached_time
        x_t = _run(fwd, fwd.net, start_text, start_time, t_stop, ctx.noise_nfe,
               reverse=False, x_cond=src_text, y=y, cfg_scale=ctx.cfg_scale)
        return x_t, _run(rev, _Uncond(rev.net), x_t, t_stop, 0.0, ctx.restore_nfe,
                         reverse=True, x_cond=torch.zeros_like(x_t), y=y)
    raise ValueError(f"unknown modality {modality!r}")


def _restore_n2d_single(model, modality, src_img, src_text, y, t_stop, ctx):
    if modality == "image":
        x_t = _analytic_noise(model.sde, src_img, 1.0, t_stop)
        return x_t, _run(model, _Uncond(model.net), x_t, t_stop, 1.0, ctx.restore_nfe,
                         reverse=False, x_cond=torch.zeros_like(x_t), y=y)
    elif modality == "text":
        x_t = _analytic_noise(model.sde, src_text, 0.0, t_stop)
        return x_t, _run(model, _Uncond(model.net), x_t, t_stop, 0.0, ctx.restore_nfe,
                         reverse=True, x_cond=torch.zeros_like(x_t), y=y)
    raise ValueError(f"unknown modality {modality!r}")


def _restore_n2d_two(fwd, rev, modality, src_img, src_text, y, t_stop, ctx):
    # infer_m builds the cross-modal conditioning (its own SDE); restore_m owns
    # the noising kernel + conditional restoration of the source modality.
    if modality == "image":       # infer text via image->text, restore via text->image
        infer_m, restore_m, src = rev, fwd, src_img
        inferred = _run(infer_m, infer_m.net, torch.randn_like(src), 1.0, 0.0, ctx.infer_nfe,
                        reverse=True, x_cond=src, y=y, cfg_scale=ctx.cfg_scale)
        if ctx.snap:
            inferred = _snap_to_vocab(inferred, infer_m.token_decoder, ctx.vocab_table,
                                      ctx.token_scale, ctx.token_emb_dim, ctx.config)
        x_t = _analytic_noise(restore_m.sde, src, 1.0, t_stop)
        return x_t, _run(restore_m, restore_m.net, x_t, t_stop, 1.0, ctx.restore_nfe,
                         reverse=False, x_cond=inferred, y=y, cfg_scale=ctx.cfg_scale)
    assert modality == "text", f"modality {modality!r} must be text"
    infer_m, restore_m, src = fwd, rev, src_text  # infer image via text->image, restore text
    inferred = _run(infer_m, infer_m.net, torch.randn_like(src), 0.0, 1.0, ctx.infer_nfe,
                    reverse=False, x_cond=src, y=y, cfg_scale=ctx.cfg_scale)
    x_t = _analytic_noise(restore_m.sde, src, 0.0, t_stop)
    return x_t, _run(restore_m, restore_m.net, x_t, t_stop, 0.0, ctx.restore_nfe,
                     reverse=True, x_cond=inferred, y=y, cfg_scale=ctx.cfg_scale)


@torch.no_grad()
def restore(mode, fwd, rev, modality, src_img, src_text, y, t_stop, ctx, cached_sample=None, cached_time=None):
    """Returns (noised, restored) bridge tensors, both shaped like the source."""
    if mode in ("data2data_single", "data2data_two"):
        return _restore_data2data(fwd=fwd, rev=rev, modality=modality, src_img=src_img, src_text=src_text, y=y, t_stop=t_stop, ctx=ctx, cached_sample=cached_sample, cached_time=cached_time)
    assert cached_sample is None and cached_time is None, f"cached_sample {cached_sample} and cached_time {cached_time} must be None"
    if mode == "n2d_single":
        model = fwd if modality == "image" else rev
        return _restore_n2d_single(model=model, modality=modality, src_img=src_img, src_text=src_text, y=y, t_stop=t_stop, ctx=ctx)
    elif mode == "n2d_two":
        return _restore_n2d_two(fwd=fwd, rev=rev, modality=modality, src_img=src_img, src_text=src_text, y=y, t_stop=t_stop, ctx=ctx)
    else:
        raise ValueError(f"unknown mode {mode!r}")


# ---------------------------------------------------------------------------
# Similarity embedders (pluggable)
# ---------------------------------------------------------------------------

class ImageEmbedder:
    """DINOv2 (default), CLIP, or DreamSim CLS-style embedding of uint8 images."""

    _MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    _STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def __init__(self, name, device, img_size=224):
        self.name, self.device, self.img_size = name, device, img_size
        if name == "dinov2":
            self.model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14").to(device).eval()
            self.mean, self.std = self._MEAN.to(device), self._STD.to(device)
        elif name == "clip":
            from transformers import CLIPModel, CLIPProcessor
            self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(device).eval()
            self.proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
        else:
            raise ValueError(f"unknown image embedder {name!r}")

    @torch.no_grad()
    def __call__(self, images_uint8: torch.Tensor) -> torch.Tensor:
        assert images_uint8.dtype == torch.uint8, f"images_uint8.dtype {images_uint8.dtype} is not uint8"
        if self.name == "clip":
            px = self.proc(images=[im.cpu() for im in images_uint8], return_tensors="pt")["pixel_values"]
            emb = self.model.get_image_features(px.to(self.device))
            emb = getattr(emb, "pooler_output", emb)
        else:  # dinov2
            x = F.interpolate(images_uint8.float() / 255.0, (self.img_size, self.img_size),
                              mode="bicubic", align_corners=False, antialias=True)
            x = (x.to(self.device) - self.mean) / self.std
            emb = self.model.forward_features(x)["x_norm_clstoken"]
            assert emb.shape == (images_uint8.shape[0], self.model.num_features), f"emb.shape {emb.shape} is not {self.model.num_features}"
        return F.normalize(emb.float(), dim=-1)


class TextEmbedder:
    """Qwen3-Embedding sentence embedding (last-token pooling, left padded)."""

    def __init__(self, model_id, device, max_length=128):
        from transformers import AutoModel, AutoTokenizer
        self.device, self.max_length = device, max_length
        self.tok = AutoTokenizer.from_pretrained(model_id, padding_side="left")
        self.model = AutoModel.from_pretrained(model_id, torch_dtype=torch.bfloat16).to(device).eval()

    @torch.no_grad()
    def __call__(self, texts: list[str]) -> torch.Tensor:
        batch = self.tok(texts, padding=True, truncation=True,
                         max_length=self.max_length, return_tensors="pt").to(self.device)
        last_hidden = self.model(**batch).last_hidden_state
        emb = last_hidden[:, -1]  # left padding -> real last token
        assert emb.shape == (batch["input_ids"].shape[0], self.model.config.hidden_size), f"emb.shape {emb.shape} is not {self.model.config.hidden_size}"
        return F.normalize(emb.float(), dim=-1)


@torch.no_grad()
def decode_texts(bridges, token_decoder, tokenizer, token_scale, config) -> list[str]:
    token_flat = bridge_to_token_flat(bridges.float(), config=config)
    ids = token_decoder(token_flat).argmax(dim=-1)
    stops, _ = norm_based_token_stops(bridges.float(), token_scale=token_scale, config=config)
    return [tokenizer.decode(ids[i, :int(stops[i])].tolist(), skip_special_tokens=True)
            for i in range(ids.shape[0])]


def _pairwise_mean_cosine(emb: torch.Tensor) -> float:
    """Mean off-diagonal cosine among unit-norm rows (diversity: lower == more)."""
    if emb.shape[0] < 2:
        raise ValueError("should not happen")
        return float("nan")
    sim = emb @ emb.t()
    n = emb.shape[0]
    return float((sim.sum() - n) / (n * (n - 1)))


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

@dataclass
class Ctx:
    noise_nfe: int
    restore_nfe: int
    infer_nfe: int
    cfg_scale: float
    snap: bool
    vocab_table: Optional[torch.Tensor]
    token_scale: float
    token_emb_dim: int
    config: object


def _scale_nfe(base: int, fraction: float) -> int:
    """NFE for a leg that travels ``fraction`` of the bridge, holding step size
    (dt) constant: base is the count extrapolated to a full unit of travel."""
    return max(1, int(round(base * fraction)))


def _load_vocab_table(data_root, device) -> torch.Tensor:
    cfg = json.loads((Path(data_root) / "token_embed_config.json").read_text())["config"]
    import numpy as np
    table = np.load(Path(data_root) / cfg.get("vocab_table", "token_vocab_table.npy"))
    return torch.from_numpy(np.ascontiguousarray(table)).to(device)


def run(args):
    # Data-parallel over images: each rank runs the full model set on a disjoint
    # slice of images (no gradients / param sync), then results are gathered to
    # rank 0. Launch with torchrun; falls back to single-process without it.
    ddp = "RANK" in os.environ
    if ddp:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl", device_id=torch.device(f"cuda:{local_rank}"))
        rank, world_size = dist.get_rank(), dist.get_world_size()
        device = torch.device(f"cuda:{local_rank}")
    else:
        rank, world_size = 0, 1
        device = torch.device(args.device)
    torch.manual_seed(args.seed + rank)

    fwd = load_model(args.forward_ckpt, args.data_root, device) if args.forward_ckpt else None
    rev = load_model(args.reverse_ckpt, args.data_root, device) if args.reverse_ckpt else None
    mode = resolve_mode(fwd, rev, args.generate_modality)
    if mode == "data2data_single":  # one bidirectional model plays both roles
        fwd = rev = (fwd or rev)
    if rank == 0:
        print(f"[editing] mode={mode} modality={args.generate_modality} world_size={world_size}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    wandb.init(project=args.wandb_project, name=args.wandb_name or f"{mode}-{args.generate_modality}",
               config={**vars(args), "mode": mode},
               mode=args.wandb_mode if rank == 0 else "disabled",
               dir=str(Path(args.out).parent))

    # Geometry comes from the dataset manifest (all models share this data_root);
    # each model carries its own SDE (see LoadedModel.sde), so there is no single
    # global SDE -- two noise-to-data models may legitimately differ.
    runtime = bridge_config_from_manifest(args.data_root, preset="auto")
    config = runtime.bridge

    from diffusers.models import AutoencoderKL
    vae_kwargs = {"subfolder": runtime.vae_subfolder} if runtime.vae_subfolder else {}
    vae = AutoencoderKL.from_pretrained(runtime.vae_model, **vae_kwargs).to(device, torch.bfloat16).eval()

    # The text-generating model owns the token decoder used to read text back
    # out (prefer the reverse/image->text model; fall back to a bidirectional one).
    text_model = next((m for m in (rev, fwd) if m is not None and m.token_decoder is not None), None)
    if args.generate_modality == "text":
        assert text_model is not None, "text generation requires a model with a token decoder"

    ctx = Ctx(
        noise_nfe=args.noise_nfe, restore_nfe=args.restore_nfe, infer_nfe=args.infer_nfe,
        cfg_scale=args.cfg_scale, snap=args.snap_inferred_text,
        vocab_table=_load_vocab_table(args.data_root, device) if args.snap_inferred_text else None,
        token_scale=runtime.token_scale, token_emb_dim=config.token_emb_dim, config=config,
    )

    if args.generate_modality == "image":
        img_embedder = ImageEmbedder(args.image_embed, device)
    else:
        txt_embedder = TextEmbedder(args.text_embed_model, device)

    ds = CommonCatalogLatentDataset(
        args.data_root, cast_dtype=torch.float32, return_caption=True, config=config,
        latent_scale=runtime.latent_scale, latent_shift=runtime.latent_shift,
        token_pad_id=(text_model.tokenizer.pad_token_id if text_model and text_model.tokenizer else None),
    )
    n_images = min(args.num_images, len(ds))
    V = args.num_variations

    # Accumulate per-fraction cosines and per-image diversity.
    fids = {f: [] for f in args.time_fractions}
    divs = {f: [] for f in args.time_fractions}

    autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    for i in tqdm(range(rank, n_images, world_size), desc="images", disable=(rank != 0)):
        item = ds[i]
        src_img = item["latent"].to(device)[None].repeat(V, 1, 1, 1)
        src_text = token_flat_to_bridge(
            item["text_token_emb"].to(device)[None], config=config
        ).repeat(V, 1, 1, 1)
        y = item["prompt_kind_label"].to(device)[None].repeat(V)

        # Decode / embed the original once (shared across variations / fractions).
        # In image mode we keep the decoded RGB for the visualization panel too.
        do_viz = args.viz_every > 0 and (i % args.viz_every == 0)
        orig_img = None
        with autocast:
            if args.generate_modality == "image":
                orig_img = _decode(vae, src_img[:1], runtime)
                orig = img_embedder(orig_img)
            else:
                orig = txt_embedder([item["caption"]])
                # NOTE: this may be a little longer than the text that's passed into model, but since it's the same across all variations, it should be fine.

        cached_sample = None
        cached_time = None
        f_prev = 0.0
        if args.cache_perturbed_samples:
            if mode not in ("data2data_single", "data2data_two"):
                raise ValueError(f"cache_perturbed_samples is only supported for data2data_single and data2data_two modes, but got {mode}")
            if not args.proportional_nfe:
                raise ValueError("cache_perturbed_samples requires proportional_nfe to be enabled")
        for f in sorted(args.time_fractions):
            t_stop = (1.0 - f) if args.generate_modality == "image" else f
            # Optionally hold the step size dt constant across fractions by scaling
            # each leg's NFE by the distance it travels (base = NFE per full unit).
            fctx = ctx if not args.proportional_nfe else replace(
                ctx, noise_nfe=_scale_nfe(
                    args.noise_nfe, 
                    f - f_prev if args.cache_perturbed_samples else f
                ),
                restore_nfe=_scale_nfe(args.restore_nfe, f), infer_nfe=_scale_nfe(args.infer_nfe, f))
            with autocast:
                noised, out = restore(mode=mode, fwd=fwd, rev=rev, modality=args.generate_modality,
                                      src_img=src_img, src_text=src_text, y=y, t_stop=t_stop, ctx=fctx,
                                      cached_sample=cached_sample, cached_time=cached_time)
                assert noised.shape == out.shape == src_img.shape == src_text.shape, f"noised.shape {noised.shape} is not {out.shape} is not {src_img.shape} is not {src_text.shape}"
                if args.generate_modality == "image":
                    emb = img_embedder(_decode(vae, out.float(), runtime))
                else:
                    texts = decode_texts(out, text_model.token_decoder,
                                         text_model.tokenizer, runtime.token_scale, config)
                    emb = txt_embedder(texts)
            assert orig.shape == (1, emb.shape[1]), f"orig.shape {orig.shape} is not (1, {emb.shape[1]})"
            assert emb.shape == (V, orig.shape[1]), f"emb.shape {emb.shape} is not (V, {orig.shape[1]})"
            fids[f].extend((emb @ orig[0]).tolist())
            divs[f].append(_pairwise_mean_cosine(emb))

            if do_viz:
                _save_panel(args, mode, f, i, vae, runtime, config, text_model,
                            orig_img, item.get("caption", ""), noised, out)
            if args.cache_perturbed_samples:
                # cache so we can use this as our next start point
                assert f >= f_prev
                f_prev = f
                cached_sample = noised
                cached_time = t_stop

    # Concatenate per-rank shards onto rank 0 (ordering is irrelevant to means).
    if ddp:
        gathered_fids = [None] * world_size if rank == 0 else None
        gathered_divs = [None] * world_size if rank == 0 else None
        dist.gather_object(fids, gathered_fids, dst=0)
        dist.gather_object(divs, gathered_divs, dst=0)
        if rank == 0:
            fids = {f: [x for g in gathered_fids for x in g[f]] for f in args.time_fractions}
            divs = {f: [x for g in gathered_divs for x in g[f]] for f in args.time_fractions}

    if rank == 0:
        _save_results(args, mode, fids, divs)
    if ddp:
        dist.destroy_process_group()


def _decode(vae, latents, runtime):
    return decode_latents(vae, latents.float(), latent_scale=runtime.latent_scale,
                          latent_shift=runtime.latent_shift, to_cpu=False)


def _save_panel(args, mode, f, idx, vae, runtime, config, text_model, orig_img, caption, noised, out, num_variations_rendered=2):
    """Emit one panel per (image, fraction): original / noised / a few restored."""
    from editing_plot import save_edit_image_panel, save_edit_text_panel
    panel_dir = Path(args.out).parent / "panels"
    title = f"{mode} | fraction={f}"
    k = args.viz_samples
    if args.generate_modality == "image":
        path = panel_dir / f"img{idx:04d}" / f"frac{f}.png"
        save_edit_image_panel(
            original=orig_img[0].cpu(),
            noised=_decode(vae, noised[:1], runtime)[0].cpu(),
            restored=[r.cpu() for r in _decode(vae, out[:num_variations_rendered], runtime)],
            out_path=path, title=title,
        )
        if wandb.run is not None:
            wandb.log({f"panels/{args.generate_modality}/frac{f}": wandb.Image(str(path), caption=title)})
    else:
        noised_txt = decode_texts(noised[:1], text_model.token_decoder,
                                  text_model.tokenizer, runtime.token_scale, config)[0]
        restored_txt = decode_texts(out[:num_variations_rendered], text_model.token_decoder,
                                    text_model.tokenizer, runtime.token_scale, config)
        path = panel_dir / f"img{idx:04d}" / f"frac{f}.txt"
        save_edit_text_panel(
            original=caption, noised=noised_txt, restored=restored_txt,
            out_path=path, title=title,
        )
        if wandb.run is not None:
            wandb.log({f"panels/frac{f}": wandb.Html(f"<pre>{path.read_text()}</pre>")})


def _save_results(args, mode, fids, divs):
    import statistics as st
    by_fraction = {}
    for f in args.time_fractions:
        c = fids[f]                          # one fidelity per (image, variation)
        d = [x for x in divs[f] if x == x]   # one diversity per image; nan when V<2
        assert len(c) == len(divs[f]) * args.num_variations, \
            f"fidelity count {len(c)} != images {len(divs[f])} * variations {args.num_variations}"
        by_fraction[f"{f}"] = {
            "fidelity_mean": st.mean(c) if c else None,
            "fidelity_std": st.pstdev(c) if len(c) > 1 else 0.0,
            "diversity_mean_pairwise": st.mean(d) if d else None,
            "n": len(c),
        }
    out = {
        "meta": {
            "forward_ckpt": args.forward_ckpt, "reverse_ckpt": args.reverse_ckpt,
            "mode": mode, "generate_modality": args.generate_modality,
            "image_embed": args.image_embed, "text_embed_model": args.text_embed_model,
            "num_images": args.num_images, "num_variations": args.num_variations,
            "noise_nfe": args.noise_nfe, "restore_nfe": args.restore_nfe,
            "infer_nfe": args.infer_nfe, "proportional_nfe": args.proportional_nfe,
            "cfg_scale": args.cfg_scale,
            "cache_perturbed_samples": args.cache_perturbed_samples,
            "snap_inferred_text": args.snap_inferred_text, "data_root": args.data_root,
        },
        "by_fraction": by_fraction,
        "raw_fidelity": {f"{f}": fids[f] for f in args.time_fractions},
    }
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    print(f"[editing] wrote {path}")
    if args.plot:
        from editing_plot import plot_single
        plot_single(out, str(path.with_suffix(".png")))
        wandb.log({f"results": wandb.Image(str(path.with_suffix(".png")), caption="Results")})

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--forward-ckpt", default=None,
                   help="text->image checkpoint (local path or hf://[repo/]path)")
    p.add_argument("--reverse-ckpt", default=None,
                   help="image->text checkpoint (local path or hf://[repo/]path)")
    p.add_argument("--generate-modality", required=True, choices=["image", "text"])
    p.add_argument("--data-root",
                   default="/pscratch/sd/g/gabeguo/datasets/text_to_image/gpic_latents_sd_test/TEST")
    p.add_argument("--num-images", type=int, default=256)
    p.add_argument("--num-variations", type=int, default=16)
    p.add_argument("--time-fractions", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0],
                   help="Corruption fraction in [0,1]; 0=no change, 1=all the way to the other end.")
    p.add_argument("--noise-nfe", type=int, default=100, help="NFE for the data2data noising leg.")
    p.add_argument("--restore-nfe", type=int, default=100, help="NFE for restoration.")
    p.add_argument("--infer-nfe", type=int, default=100,
                   help="NFE for cross-modal inference (two-model noise-to-data).")
    p.add_argument("--proportional-nfe", action="store_true",
                   help="Scale the noising/restoration/inference NFE by the corruption fraction "
                        "(base NFE = count per full unit of travel), holding the step size dt constant across fractions.")
    p.add_argument("--cache-perturbed-samples", action="store_true", default=False,
                   help="Cache the perturbed samples for the next fraction. This can reduce the total NFE in the data-to-data path. Only supported for data2data_single and data2data_two modes when proportional_nfe is enabled.")
    p.add_argument("--cfg-scale", type=float, default=0.0)
    p.add_argument("--snap-inferred-text", action="store_true", default=True,
                   help="Snap inferred text through the token vocab table (default on).")
    p.add_argument("--no-snap-inferred-text", dest="snap_inferred_text", action="store_false")
    p.add_argument("--image-embed", default="dinov2", choices=["dinov2", "clip"])
    p.add_argument("--text-embed-model", default="Qwen/Qwen3-Embedding-8B")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="editing_results/results.json")
    p.add_argument("--plot", action="store_true", help="Also write a per-run PNG.")
    p.add_argument("--wandb-project", default="bib-editing")
    p.add_argument("--wandb-name", default=None, help="Run name (default: <mode>-<modality>).")
    p.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"])
    p.add_argument("--viz-every", type=int, default=20,
                   help="Save an editing panel every N images (0 disables). Panels go to <out dir>/panels/.")
    p.add_argument("--viz-samples", type=int, default=1,
                   help="Number of restored samples shown per panel.")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
