"""
Angular token-perturbation + slerp probe for the text->image data-to-data bridge.

The text endpoint (x_0) is either the ground-truth caption (default) or, with
``--text-source image``, text HALLUCINATED from the source image: we follow the
(data-to-data) image->text bridge conditioned on the image, then snap the result
onto real vocab-table token embeddings via the token decoder (reusing
``editing_experiment._snap_to_vocab``). Everything downstream is identical.

For each image-text pair we:
  1. Take the (unit-norm, on-hypersphere) NON-PAD text token embeddings.
  2. Perturb a few of them by an angle ``theta ~ N(0, alpha^2)`` along a random
     tangent direction, via the exponential map on the unit hypersphere
     (``exp_u(theta * t) = cos(theta) u + sin(theta) t``, ``t`` tangent unit).
  3. Slerp per-token between the original and perturbed embeddings
     (``--num-slerp`` points, endpoints included). Unperturbed tokens are fixed.
  4. Run the (data-to-data) diffusion generative process text->image for each
     slerp point, holding the Brownian noise FIXED ACROSS slerp points, so the
     only thing that changes down a row is the conditioning text (a clean
     counterfactual). Optionally SDEdit-style: start from a bridge sample at
     time ``--init-t`` between that column's text (x_0) and the ORIGINAL image
     (x_1) via ``sample_p_base_x_t_cond_x_0_x_1``, then integrate to t=1.
  5. Plot one row of images per pair, each captioned with its decoded text, plus
     a bottom arrow annotated with that image's MEAN applied perturbation angle.

Why only data-to-data models (single bidirectional, or two covering both
directions)? The generative process here starts FROM the text as x_0 and
integrates the bridge to the image x_1 -- exactly what a data-to-data bridge
does, and what the sample_p_base interpolation-init between the two real data
endpoints requires. A noise-to-data model never has the text as the state its
process starts from, so the probe would be ill-defined.

Why fixed Brownian noise across slerp points? It turns text->image into a
deterministic function of the conditioning for a fixed noise realization, so any
change down a row is attributable ONLY to the text perturbation (not sampling
variance), yielding smooth, interpretable morphs. We achieve it WITHOUT touching
the SDE: the only randomness in the process is ``torch.randn_like`` inside
``SDE.dX_t`` and the single draw in ``sample_p_base_*``, both from the global
torch RNG. Re-seeding the global RNG to the same value immediately before each
slerp point's trajectory (batch shape + NFE held constant) reproduces the exact
same Brownian path per image-text pair across slerp points. Different pairs in a
batch still get different noise (distinct rows of the same reseeded draw), so we
batch over pairs for GPU utilization but must NOT batch over slerp points.

Run from text_to_image with PYTHONPATH=.:.. (same as train.py / editing_experiment.py).
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from data_utils.latent_dataset import CommonCatalogLatentDataset
from editing_experiment import (
    decode_texts, load_model, resolve_mode, _run, _load_vocab_table, _snap_to_vocab,
)
from eval_plot import decode_latents
from sde_utils.loss import sample_p_base_x_t_cond_x_0_x_1, sample_p_base_x_t_cond_x_prev_x_next
from token_bridge import (
    bridge_config_from_manifest, token_flat_to_bridge, bridge_to_token_flat,
    norm_based_token_stops,
)
from tqdm import tqdm


def infer_noise_samples(x0, x1, y, nfe, fwd, device):
    assert fwd.sde.A == 0
    all_ts = torch.linspace(0.0, 1.0, nfe+1, device=device)
    x_prev = x0
    all_dB_Q = []
    for i in range(1, len(all_ts)):
        x_t = sample_p_base_x_t_cond_x_prev_x_next(
            sde=fwd.sde,
            x_prev=x_prev,
            x_next=x1,
            t=all_ts[i].expand(x0.shape[0]),
            t_prev=all_ts[i-1].expand(x0.shape[0]),
            t_next=torch.full((x0.shape[0],), 1.0, device=device),
        )
        curr_dX = x_t - x_prev
        curr_score = fwd.net(
            x=x_prev, 
            t=all_ts[i-1].expand(x0.shape[0]), 
            y=y, 
            x_cond=x0, 
            reverse=False, 
        )
        curr_sigma = fwd.sde.sigma(all_ts[i-1])
        dt = all_ts[i] - all_ts[i-1]
        dB_Q = curr_dX / curr_sigma - curr_sigma * curr_score * dt
        x_prev = x_t
        all_dB_Q.append(dB_Q)
    retval = torch.stack(all_dB_Q, dim=0)
    assert retval.shape == (nfe, x0.shape[0], x0.shape[1], x0.shape[2], x0.shape[3])
    return retval

def simulate_with_fixed_noise(x0, all_dB, y, nfe, fwd, device):
    assert fwd.sde.A == 0
    all_ts = torch.linspace(0.0, 1.0, nfe+1, device=device)
    x_prev = x0
    for i in range(len(all_ts) - 1):
        curr_score = fwd.net(
            x=x_prev,
            t=all_ts[i].expand(x0.shape[0]),
            y=y,
            x_cond=x0,
            reverse=False,
        )
        curr_sigma = fwd.sde.sigma(all_ts[i])
        dt = all_ts[i+1] - all_ts[i]
        curr_drift = curr_score * curr_sigma**2 * dt
        x_t = x_prev + curr_drift + curr_sigma * all_dB[i]
        x_prev = x_t
    return x_t

# ---------------------------------------------------------------------------
# Perturbation + slerp on the unit hypersphere
# ---------------------------------------------------------------------------

def _slerp(a: torch.Tensor, b: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """Per-token slerp. ``a``, ``b``: (T, D) unit vectors; ``s``: (S,) in [0,1].

    Returns (S, T, D). Tokens where ``a`` and ``b`` are (nearly) identical --
    including the untouched / padding tokens -- fall back to ``a`` (the geodesic
    is degenerate and ``sin(Omega) -> 0``).
    """
    T = a.shape[0]
    dot = (a * b).sum(-1).clamp(-1.0, 1.0)          # (T,)
    omega = torch.arccos(dot)                        # (T,)
    sin_o = torch.sin(omega)                         # (T,)
    s = s.view(-1, 1)                                # (S, 1)
    # Safe denominator; degenerate tokens are overwritten below.
    denom = torch.where(sin_o.abs() < 1e-6, torch.ones_like(sin_o), sin_o)
    ca = torch.sin((1.0 - s) * omega) / denom        # (S, T)
    cb = torch.sin(s * omega) / denom                # (S, T)
    res = ca[..., None] * a + cb[..., None] * b      # (S, T, D)
    degenerate = (sin_o.abs() < 1e-6).view(1, T, 1)
    return torch.where(degenerate, a.unsqueeze(0).expand_as(res), res)


def perturb_and_slerp(
    token_emb: torch.Tensor,   # (token_flat_dim,) scaled by token_scale
    mask: torch.Tensor,        # (token_seq_len,) bool: True = real (non-pad) token
    *,
    num_perturb_tokens: int,
    alpha: float,
    num_slerp: int,
    token_scale: float,
    config,
    gen: torch.Generator,
    device: torch.device,
):
    """Returns (bridges, mean_angle):
    ``bridges`` is (S, C, H, W) bridge-space text for the S slerp points;
    ``mean_angle`` is the mean applied angle (rad) over this image's perturbed
    tokens (0 if none were perturbed).
    """
    T, D = config.token_seq_len, config.token_emb_dim
    toks = token_emb.to(device).float().view(T, D)
    unit = toks / token_scale                        # real: unit-norm; pad: ~0

    valid = mask.to(device).nonzero(as_tuple=False).flatten()   # non-pad positions
    k = min(int(num_perturb_tokens), int(valid.numel()))
    perm = torch.randperm(valid.numel(), generator=gen, device=device)[:k]
    chosen = valid[perm]                             # (k,)

    perturbed = unit.clone()
    angles = []
    for pos in chosen.tolist():
        u = unit[pos]                                # (D,), unit norm
        theta = torch.tensor(alpha, device=device) #torch.randn((), generator=gen, device=device) * alpha
        g = torch.randn(D, generator=gen, device=device)
        v = g - (g @ u) * u                          # project onto tangent space at u
        v_norm = v.norm()
        if v_norm < 1e-8:
            continue
        t_dir = v / v_norm
        pu = torch.cos(theta) * u + torch.sin(theta) * t_dir     # exponential map
        perturbed[pos] = pu
        angles.append(torch.arccos(torch.clamp((u * pu).sum(), -1.0, 1.0)))
    mean_angle = torch.stack(angles).mean() if angles else torch.zeros((), device=device)

    s = torch.linspace(0.0, 1.0, int(num_slerp), device=device)
    slerp = _slerp(unit, perturbed, s)               # (S, T, D)
    assert slerp.shape == (int(num_slerp), T, D)
    slerp = slerp * token_scale
    slerp = slerp * mask.to(device).view(1, T, 1).float()        # keep padding at 0
    bridges = token_flat_to_bridge(slerp.reshape(int(num_slerp), T * D), config=config)
    return bridges, mean_angle


# ---------------------------------------------------------------------------
# LLM (semantic) perturbation: ask the LLM to rewrite the caption changing k
# words, re-embed the whole new sentence via the dataset's tokenizer +
# token-embedding lookup table, then slerp toward it.
#
# This is the on-manifold counterpart to the angular perturbation above: the new
# endpoint is built from REAL vocab-table token embeddings (the exact context-
# independent, MRL-truncated, L2-normalized rows the dataset uses), so it is a
# valid data point rather than an arbitrary point in embedding space.
# ---------------------------------------------------------------------------

def ids_to_unit_tokens(ids, vocab_table, token_emb_dim, token_seq_len, device):
    """Gather lookup-table rows for a list of token ids (truncated to
    ``token_emb_dim``, L2-normalized), padded to ``token_seq_len``.
    Returns (unit (T, D), mask (T,) bool)."""
    unit = torch.zeros(token_seq_len, token_emb_dim, device=device)
    mask = torch.zeros(token_seq_len, dtype=torch.bool, device=device)
    L = min(len(ids), token_seq_len)
    if L:
        idt = torch.tensor(ids[:L], dtype=torch.long, device=device)
        unit[:L] = F.normalize(vocab_table[idt][:, :token_emb_dim].float(), dim=-1)
        mask[:L] = True
    return unit, mask


def _caption_slerp(orig_unit, new_unit, orig_mask, new_mask, s, token_scale):
    """Per-position interpolation between two (possibly different-length)
    captions. Positions real in BOTH captions are slerped on the sphere;
    positions real in only one are magnitude-faded (its unit vector scaled from
    0->1 or 1->0), so a token continuously appears/disappears as its norm crosses
    the decoder's stop threshold. Returns (S, T, D) SCALED by ``token_scale``."""
    T, D = orig_unit.shape
    S = s.shape[0]
    both = (orig_mask & new_mask).view(1, T, 1)
    only_o = (orig_mask & ~new_mask).view(1, T, 1)
    only_n = (new_mask & ~orig_mask).view(1, T, 1)

    slerp_both = _slerp(orig_unit, new_unit, s) * token_scale                     # (S, T, D)
    fade_o = (1.0 - s).view(S, 1, 1) * orig_unit.unsqueeze(0) * token_scale
    fade_n = s.view(S, 1, 1) * new_unit.unsqueeze(0) * token_scale

    out = torch.zeros(S, T, D, device=orig_unit.device)
    out = torch.where(both, slerp_both, out)
    out = torch.where(only_o, fade_o, out)
    out = torch.where(only_n, fade_n, out)
    return out

def _strip_think(text: str) -> str:
    """Drop any Qwen3 <think>...</think> block and surrounding quotes/whitespace."""
    if "</think>" in text:
        text = text.split("</think>")[-1]
    return text.strip().strip('"').strip("'").strip()


@torch.no_grad()
def llm_edit_caption(caption, k, llm, llm_tok, device, max_tries=3):
    """Ask the LLM to change ``k`` words in ``caption`` and return the WHOLE
    edited sentence."""
    content = (
        f"Here is a sentence: {caption}\n"
        f"Change exactly {k} words in it to {k} other words that keep the sentence "
        f"semantically meaningful, and keep all other words the same. "
        f"Reply with only the full edited sentence."
    )
    messages = [{"role": "user", "content": content}]
    text = llm_tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = llm_tok([text], return_tensors="pt").to(device)
    orig_ids = torch.tensor(
        llm_tok(caption, add_special_tokens=False)["input_ids"], device=device
    )
    n_orig = orig_ids.numel()
    num_return_tokens = n_orig + 4          # headroom for a same-length rewrite + EOS
    prompt_len = inputs["input_ids"].shape[1]
    best = None
    best_score = 100000
    for trial in range(max_tries):
        gen = llm.generate(**inputs, max_new_tokens=num_return_tokens, do_sample=True,
                       temperature=0.7, top_p=0.8, pad_token_id=llm_tok.eos_token_id)
        new_ids = gen[0][prompt_len:]
        new_ids = new_ids[new_ids != llm_tok.eos_token_id]           # drop trailing EOS/pad
        n_new = new_ids.numel()
        L = min(n_new, n_orig)
        num_diff = (new_ids[:L] != orig_ids[:L]).sum() + abs(n_new - n_orig)
        score = abs(num_diff - k)
        if best is None:
            best = new_ids
            best_score = score
        elif score <= best_score:
            best = new_ids
            best_score = score
        if n_new == n_orig and num_diff == k: # stop when we've got perfect satisfcation
            break
    resp = llm_tok.decode(best, skip_special_tokens=True)
    new_caption = _strip_think(resp)
    return new_caption or caption


def llm_perturb_and_slerp(
    caption: str,
    orig_token_ids: torch.Tensor,   # (token_seq_len,) long, dataset-tokenizer ids
    orig_mask: torch.Tensor,        # (token_seq_len,) bool
    *,
    k: int,
    num_slerp: int,
    llm,
    llm_tok,
    dataset_tokenizer,
    vocab_table: torch.Tensor,
    token_scale: float,
    config,
    device: torch.device,
    max_tries: int,
):
    """Ask the LLM to rewrite the caption (k words changed), tokenize the whole
    new sentence with the dataset tokenizer, re-embed via the vocab table, and
    slerp from the original toward it. Returns (bridges (S,C,H,W), mean_angle,
    new_caption)."""
    T, D = config.token_seq_len, config.token_emb_dim
    orig_ids = orig_token_ids.to(device).long()
    orig_mask = orig_mask.to(device).bool()
    n_valid = int(orig_mask.sum())
    orig_valid = [int(x) for x in orig_ids[:n_valid].tolist()]

    new_caption = llm_edit_caption(caption, k, llm, llm_tok, device, max_tries=max_tries)
    enc = dataset_tokenizer(new_caption, add_special_tokens=False, truncation=True, max_length=T)
    new_ids = [int(x) for x in enc["input_ids"][:T]]

    orig_unit, orig_umask = ids_to_unit_tokens(orig_valid, vocab_table, D, T, device)
    new_unit, new_mask = ids_to_unit_tokens(new_ids, vocab_table, D, T, device)

    s = torch.linspace(0.0, 1.0, int(num_slerp), device=device)
    scaled = _caption_slerp(orig_unit, new_unit, orig_umask, new_mask, s, token_scale)
    bridges = token_flat_to_bridge(scaled.reshape(int(num_slerp), T * D), config=config)

    # Mean applied angle over tokens present in BOTH sequences that changed.
    both = orig_umask & new_mask
    if both.any():
        dot = (orig_unit[both] * new_unit[both]).sum(-1).clamp(-1.0, 1.0)
        ang = torch.arccos(dot)
        changed = ang > 1e-3
        mean_angle = ang[changed].mean() if changed.any() else torch.zeros((), device=device)
    else:
        mean_angle = torch.zeros((), device=device)
    return bridges, mean_angle, new_caption


# ---------------------------------------------------------------------------
# Optional: hallucinate the text endpoint FROM the source image
#
# Instead of the ground-truth caption, follow the (data-to-data) image->text
# bridge conditioned on the image to infer text, then SNAP it onto real vocab-
# table token embeddings with the token decoder (exactly editing_experiment's
# `_snap_to_vocab`). The snapped bridge is a valid on-manifold text data point,
# so the rest of the probe (perturb -> slerp -> regenerate image) is unchanged.
# ---------------------------------------------------------------------------

@torch.no_grad()
def infer_text_from_image(
    rev,
    src_img: torch.Tensor,       # (B, C, H, W) image bridge tensor (latents)
    y: torch.Tensor,             # (B,)
    *,
    nfe: int,
    cfg_scale: float,
    vocab_table: torch.Tensor,
    token_scale: float,
    config,
    autocast,
):
    """Returns (token_emb_flat (B, T*D) scaled by token_scale, mask (B, T) bool,
    ids (B, T) long) for text inferred from ``src_img`` and snapped to the vocab
    table -- drop-in replacements for the dataset's ground-truth text fields."""
    T, D = config.token_seq_len, config.token_emb_dim
    with autocast:
        # Data-to-data image->text: start at the image (t=1), integrate the bridge
        # backward to the text endpoint (t=0), conditioned on the source image.
        text_bridge = _run(rev, rev.net, src_img, 1.0, 0.0, nfe,
                           reverse=True, x_cond=src_img, y=y, cfg_scale=cfg_scale)
        snapped = _snap_to_vocab(text_bridge, rev.token_decoder, vocab_table,
                                 token_scale, D, config)
    flat = bridge_to_token_flat(snapped.float(), config=config)            # (B, T*D), scaled
    # ids + stop positions from the pre-snap decode (what _snap_to_vocab used).
    ids = rev.token_decoder(bridge_to_token_flat(text_bridge.float(), config=config)).argmax(dim=-1)
    stops, _ = norm_based_token_stops(text_bridge.float(), token_scale=token_scale, config=config)
    mask = torch.arange(T, device=ids.device)[None, :] < stops[:, None]    # (B, T) bool
    return flat, mask, ids


# ---------------------------------------------------------------------------
# Generation with Brownian noise held fixed across slerp points
# ---------------------------------------------------------------------------

def _reseed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def generate_row(
    fwd,
    slerp_texts: torch.Tensor,   # (S, B, C, H, W)
    img: torch.Tensor,           # (B, C, H, W) original image latents (for init)
    y: torch.Tensor,             # (B,)
    *,
    nfe: int,
    cfg_scale: float,
    init_t,
    seed_chunk: int,
    autocast,
    infer_noise: bool,
) -> torch.Tensor:
    """Returns (S, B, C, H, W) generated image latents.

    Re-seeds the global RNG to the SAME value before each slerp point so the
    per-pair Brownian path (and the optional sample_p_base init draw) is
    identical across slerp points; the batch dimension B still gets distinct
    noise per pair.
    """
    S = slerp_texts.shape[0]
    outs = []
    if infer_noise:
        with autocast:
            all_dB = infer_noise_samples(
                x0=slerp_texts[0], 
                x1=img, 
                y=y, 
                nfe=nfe, 
                fwd=fwd, 
                device=img.device
            )
    for j in range(S):
        _reseed(seed_chunk)                          # identical draw sequence per j
        text_j = slerp_texts[j]                      # (B, C, H, W)
        with autocast:
            if init_t is not None:
                t = torch.full((text_j.shape[0],), float(init_t), device=text_j.device)
                x_start = sample_p_base_x_t_cond_x_0_x_1(sde=fwd.sde, x_0=text_j, x_1=img, t=t)
                t_from = float(init_t)
            else:
                x_start = text_j
                t_from = 0.0
            if infer_noise:
                assert t_from == 0.0
                out_j = simulate_with_fixed_noise(
                    x0=x_start, 
                    all_dB=all_dB,
                    y=y, 
                    nfe=nfe, 
                    fwd=fwd, 
                    device=img.device,
                )
            else:
                out_j = _run(fwd, fwd.net, x_start, t_from, 1.0, nfe,
                         reverse=False, x_cond=text_j, y=y, cfg_scale=cfg_scale)
        outs.append(out_j.float())
    return torch.stack(outs, dim=0)                  # (S, B, C, H, W)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def save_panel(images_uint8, captions, mean_angle: float, alpha: float, out_path: Path,
               orig_caption: str | None = None, new_caption: str | None = None) -> None:
    """One row of S images, each captioned with its decoded text, plus a bottom
    arrow labeled with the mean applied perturbation angle for this image. When
    ``new_caption`` is given (LLM mode), the original vs edited caption is shown
    side by side as a suptitle."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import textwrap

    S = images_uint8.shape[0]
    fig, axes = plt.subplots(1, S, figsize=(2.4 * S, 3.4))
    if S == 1:
        axes = [axes]
    for j in range(S):
        ax = axes[j]
        ax.imshow(images_uint8[j].permute(1, 2, 0).numpy())
        ax.set_xticks([])
        ax.set_yticks([])
        cap = captions[j] if captions[j] else "(empty)"
        ax.set_title("\n".join(textwrap.wrap(cap, 22)), fontsize=7)

    cos_v = math.cos(mean_angle)
    fig.subplots_adjust(bottom=0.20, top=0.80, wspace=0.05)
    bar = fig.add_axes([0.08, 0.05, 0.84, 0.06])
    bar.axis("off")
    bar.annotate("", xy=(1.0, 0.4), xytext=(0.0, 0.4),
                 arrowprops=dict(arrowstyle="-|>", lw=2, color="black"))
    bar.text(0.0, 0.75, "original\n0.000 rad, cos=1.000", ha="left", va="bottom", fontsize=8)
    bar.text(1.0, 0.75, f"perturbed\nmean {mean_angle:.3f} rad, cos={cos_v:.3f}",
             ha="right", va="bottom", fontsize=8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run(args):
    device = torch.device(args.device)

    fwd = load_model(args.forward_ckpt, args.data_root, device) if args.forward_ckpt else None
    rev = load_model(args.reverse_ckpt, args.data_root, device) if args.reverse_ckpt else None
    mode = resolve_mode(fwd, rev, "image")
    assert mode in ("data2data_single", "data2data_two"), (
        f"this experiment requires a data-to-data model set; got mode={mode!r}"
    )
    if mode == "data2data_single":                   # one bidirectional model plays both roles
        fwd = rev = (fwd or rev)
    assert fwd is not None, "need a text->image (forward) data-to-data model"
    print(f"[perturb-slerp] mode={mode} num_images={args.num_images} num_slerp={args.num_slerp}")

    runtime = bridge_config_from_manifest(args.data_root, preset="auto")
    config = runtime.bridge
    token_scale = runtime.token_scale

    # A model with a token decoder is needed to read the (perturbed) text back
    # out for the captions (prefer the reverse / image->text model).
    text_model = next((m for m in (rev, fwd) if m is not None and m.token_decoder is not None), None)
    assert text_model is not None, "need a model with a token decoder to caption the text"
    if args.text_source == "image":
        assert rev is not None and rev.token_decoder is not None, (
            "--text-source image needs an image->text (reverse) model with a token decoder"
        )
        print(f"[perturb-slerp] text_source=image (infer text from image, infer_nfe={args.infer_nfe})")

    from diffusers.models import AutoencoderKL
    vae_kwargs = {"subfolder": runtime.vae_subfolder} if runtime.vae_subfolder else {}
    vae = AutoencoderKL.from_pretrained(runtime.vae_model, **vae_kwargs).to(device, torch.bfloat16).eval()

    # Vocab table: needed to snap LLM-rewritten captions AND image-inferred text
    # back onto real token embeddings.
    vocab_table = None
    if args.perturb_mode == "llm" or args.text_source == "image":
        vocab_table = _load_vocab_table(args.data_root, device)

    # LLM (semantic) perturbation set-up: rewrite captions with an LLM, then
    # re-embed via the dataset's tokenizer + token-embedding lookup table.
    llm = llm_tok = None
    if args.perturb_mode == "llm":
        from transformers import AutoModelForCausalLM, AutoTokenizer
        llm_tok = AutoTokenizer.from_pretrained(args.llm_model)
        llm = AutoModelForCausalLM.from_pretrained(
            args.llm_model, torch_dtype=torch.bfloat16
        ).to(device).eval()
        print(f"[perturb-slerp] perturb_mode=llm  llm_model={args.llm_model}")

    ds = CommonCatalogLatentDataset(
        args.data_root, cast_dtype=torch.float32, return_caption=True, config=config,
        latent_scale=runtime.latent_scale, latent_shift=runtime.latent_shift,
        token_pad_id=(text_model.tokenizer.pad_token_id if text_model.tokenizer else None),
    )
    out_dir = Path(args.out)
    autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16)

    # Collect the first ``num_images`` pairs whose caption is short enough
    # (token length, incl. special tokens, capped at token_seq_len by the dataset).
    eligible = []
    for i in range(len(ds)):
        if len(eligible) >= args.num_images:
            break
        if int(ds[i]["text_token_length"]) <= args.max_token_length:
            eligible.append(i)
    if len(eligible) < args.num_images:
        print(f"[perturb-slerp] only {len(eligible)} pairs have caption token length "
              f"<= {args.max_token_length} (asked for {args.num_images})")
    n = len(eligible)

    for bstart in tqdm(range(0, n, args.batch_size), total=math.ceil(n / args.batch_size)):
        idxs = eligible[bstart:bstart + args.batch_size]
        items = [ds[i] for i in idxs]
        img = torch.stack([it["latent"] for it in items]).to(device).float()
        y = torch.stack([it["prompt_kind_label"] for it in items]).to(device).long()

        # Optionally replace the ground-truth text endpoint with text HALLUCINATED
        # from the source image (image->text bridge, snapped to the vocab table).
        # Batched over the pairs; independent of the fixed-noise generation below.
        inf_flat = inf_mask = inf_ids = inf_caps = None
        if args.text_source == "image":
            _reseed(args.seed + bstart)
            inf_flat, inf_mask, inf_ids = infer_text_from_image(
                rev, img, y, nfe=args.infer_nfe, cfg_scale=args.cfg_scale,
                vocab_table=vocab_table, token_scale=token_scale, config=config,
                autocast=autocast,
            )
            inf_caps = [text_model.tokenizer.decode(
                inf_ids[bi][inf_mask[bi]].tolist(), skip_special_tokens=True)
                for bi in range(len(idxs))]

        slerp_list, mean_angles = [], []
        new_captions = [None] * len(idxs)
        orig_captions = [""] * len(idxs)
        for bi, i in enumerate(idxs):
            # Text endpoint (x_0) to perturb: dataset ground truth, or image-inferred.
            if args.text_source == "image":
                src_emb, src_mask = inf_flat[bi], inf_mask[bi]
                src_ids, src_caption = inf_ids[bi], inf_caps[bi]
            else:
                src_emb, src_mask = items[bi]["text_token_emb"], items[bi]["text_token_mask"]
                src_ids, src_caption = items[bi]["text_token_ids"], items[bi].get("caption", "")

            if args.perturb_mode == "angular":
                # Dedicated generator (its own seed) so sampling the perturbation
                # does NOT disturb the global RNG that drives the Brownian noise.
                gen = torch.Generator(device=device).manual_seed(args.seed * 1_000_003 + i)
                bridges, ma = perturb_and_slerp(
                    src_emb, src_mask,
                    num_perturb_tokens=args.num_perturb_tokens, alpha=args.alpha,
                    num_slerp=args.num_slerp, token_scale=token_scale, config=config,
                    gen=gen, device=device,
                )
            else:  # llm
                torch.manual_seed(args.seed + i)         # reproducible LLM sampling
                bridges, ma, new_cap = llm_perturb_and_slerp(
                    caption=src_caption,
                    orig_token_ids=src_ids,
                    orig_mask=src_mask,
                    k=args.num_perturb_tokens, num_slerp=args.num_slerp,
                    llm=llm, llm_tok=llm_tok, dataset_tokenizer=text_model.tokenizer,
                    vocab_table=vocab_table, token_scale=token_scale, config=config,
                    device=device, max_tries=args.llm_max_tries,
                )
                new_captions[bi] = new_cap
                print(f"[llm] orig: {src_caption!r}\n[llm]  new: {new_cap!r}")
            orig_captions[bi] = src_caption
            slerp_list.append(bridges)
            mean_angles.append(float(ma))
        slerp_texts = torch.stack(slerp_list, dim=1)  # (S, B, C, H, W)

        outs = generate_row(
            fwd, slerp_texts, img, y, nfe=args.nfe, cfg_scale=args.cfg_scale,
            init_t=args.init_t, seed_chunk=args.seed + bstart, autocast=autocast,
            infer_noise=args.infer_noise,
        )                                             # (S, B, C, H, W)

        for bi, i in enumerate(idxs):
            with autocast:
                imgs_uint8 = decode_latents(
                    vae, outs[:, bi], latent_scale=runtime.latent_scale,
                    latent_shift=runtime.latent_shift, to_cpu=True,
                )
                caps = decode_texts(slerp_texts[:, bi], text_model.token_decoder,
                                    text_model.tokenizer, token_scale, config)
            save_panel(imgs_uint8, caps, mean_angles[bi], args.alpha,
                       out_dir / f"panel_img{i:04d}.png",
                       orig_caption=(orig_captions[bi] if args.perturb_mode == "llm" else None),
                       new_caption=new_captions[bi])
            print(f"[perturb-slerp] wrote {out_dir / f'panel_img{i:04d}.png'}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--forward-ckpt", default=None,
                   help="text->image data-to-data checkpoint (local path or hf://[repo/]path)")
    p.add_argument("--reverse-ckpt", default=None,
                   help="image->text data-to-data checkpoint (for the two-model set / token decoder)")
    p.add_argument("--data-root",
                   default="/pscratch/sd/g/gabeguo/datasets/text_to_image/gpic_latents_sd_test/TEST")
    p.add_argument("--num-images", type=int, default=8, help="How many image-text pairs to run.")
    p.add_argument("--max-token-length", type=int, default=10,
                   help="Only use pairs whose caption token length is <= this (set large to disable).")
    p.add_argument("--batch-size", type=int, default=8,
                   help="Image-text pairs per GPU batch (slerp points are looped, not batched).")
    p.add_argument("--num-slerp", type=int, default=8,
                   help="Number of slerp points, endpoints included.")
    p.add_argument("--num-perturb-tokens", type=int, default=3,
                   help="Tokens to change: angular mode perturbs this many; llm mode asks the "
                        "LLM to replace this many (clamped to #non-pad tokens in angular mode).")
    p.add_argument("--perturb-mode", choices=["angular", "llm"], default="angular",
                   help="angular: exponential-map perturbation of token embeddings; "
                        "llm: rewrite the caption with an LLM, then re-embed via the vocab table.")
    p.add_argument("--text-source", choices=["dataset", "image"], default="dataset",
                   help="dataset: use the ground-truth caption as the text endpoint; "
                        "image: hallucinate text from the source image via the image->text "
                        "bridge and snap it to the vocab table, then perturb THAT.")
    p.add_argument("--infer-nfe", type=int, default=250,
                   help="(--text-source image) Euler-Maruyama steps for the image->text inference.")
    p.add_argument("--llm-model", default="Qwen/Qwen3-8B",
                   help="HF causal LM used to edit captions in --perturb-mode llm.")
    p.add_argument("--llm-max-tries", type=int, default=3,
                   help="Max resampling attempts for a same-length, <=k-token caption edit.")
    p.add_argument("--alpha", type=float, default=math.pi / 8,
                   help="(angular mode) Std (rad) of the perturbation angle theta ~ N(0, alpha^2).")
    p.add_argument("--init-t", type=float, default=None,
                   help="If set, SDEdit-style: start from a bridge sample at this time between "
                        "the column's text (x_0) and the original image (x_1), then integrate to 1.")
    p.add_argument("--infer-noise", action="store_true", default=False,
                   help="If set, use the inferred noise to simulate the generation process.")
    p.add_argument("--nfe", type=int, default=250, help="Euler-Maruyama steps for generation.")
    p.add_argument("--cfg-scale", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default="perturb_slerp_results",
                   help="Output directory for the per-image panels.")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
