"""
Distributed evaluation metrics for the text-to-image DiT bridge.

Two metrics, both DDP-aware (every rank participates so the work is sharded
across all GPUs):

  1. ``compute_fid_distributed`` -- FID over the text -> image direction.
     Each rank generates a shard of ``num_samples`` text-conditioned images by
     simulating the forward SDE, decodes them through the VAE, and pushes both
     the GT and generated images into a ``torchmetrics`` FID accumulator.
     ``FrechetInceptionDistance.compute()`` all-reduces its running state
     across the default process group, so the final FID is computed over the
     full ``num_samples`` regardless of how many ranks participated.

  2. ``compute_text_decode_distributed`` -- reverse-simulate image -> token
     endpoint, decode token IDs with the shared token MLP, and report token
     accuracy plus optional CIDEr/CLIPScore if their packages are installed.

Both functions must be called on every rank (they synchronize internally).
``compute_fid_distributed`` returns a non-empty dict only on rank 0;
``compute_text_decode_distributed`` returns the same dict on every rank
(the metrics are all-reduced) so any rank can read them, but rank 0 is the
one that should log to wandb.
"""

from __future__ import annotations

import json
import math
import warnings
from contextlib import nullcontext
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from eval_plot import _temporarily_swap_score_network, decode_latents
from token_bridge import (
    BRIDGE_RUNTIME_PRESETS,
    BridgeRuntimeConfig,
    TokenBridgeConfig,
    bridge_to_token_flat,
    norm_based_token_stops,
    pad_id_token_stops,
    prepare_bridge_batch,
)
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def _clip_score_manual(
    clip_model,
    clip_processor,
    images_uint8: torch.Tensor,
    texts: list[str],
    device: torch.device,
) -> torch.Tensor:
    """Per-sample CLIPScore (``100 * cosine_similarity``) for image/text pairs.

    Mirrors torchmetrics' CLIPScore math (L2-normalize each embedding, dot,
    scale by 100, no negative clamp) but calls the HF CLIP model directly to
    avoid the torchmetrics<->transformers private-API mismatch. transformers
    5.x returns a ``BaseModelOutputWithPooling`` whose ``pooler_output`` holds
    the projected embedding, so we unwrap it.

    ``images_uint8`` is a ``(B, 3, H, W)`` uint8 ``[0, 255]`` tensor; the CLIP
    processor handles resizing/rescaling/normalization internally.
    """
    img_inputs = clip_processor(
        images=[img.cpu() for img in images_uint8],
        return_tensors="pt",
        padding=True,
    )
    img_emb = clip_model.get_image_features(img_inputs["pixel_values"].to(device))
    img_emb = getattr(img_emb, "pooler_output", img_emb)

    txt_inputs = clip_processor(
        text=texts, return_tensors="pt", padding=True, truncation=True
    )
    txt_emb = clip_model.get_text_features(
        txt_inputs["input_ids"].to(device),
        txt_inputs["attention_mask"].to(device),
    )
    txt_emb = getattr(txt_emb, "pooler_output", txt_emb)

    img_emb = img_emb / img_emb.norm(p=2, dim=-1, keepdim=True)
    txt_emb = txt_emb / txt_emb.norm(p=2, dim=-1, keepdim=True)
    return 100.0 * (img_emb * txt_emb).sum(dim=-1)


def _save_distributed_clip_scores(
    path: Optional[str | Path],
    local_scores: list[float],
    eval_indices: list[int],
    rank: int,
    world_size: int,
    eval_pg,
) -> None:
    """Gather per-example scores in dataset-index order and save them on rank 0."""
    if path is None:
        return

    gathered: list[dict | None] = [None] * world_size
    dist.all_gather_object(
        gathered,
        {
            "indices": eval_indices[rank::world_size],
            "scores": local_scores,
        },
        group=eval_pg,
    )
    parts = [part for part in gathered if part is not None]

    # Preserve the existing behavior when CLIP initialization is disabled or
    # fails consistently across ranks: report no score and write no file.
    if all(not part["scores"] for part in parts):
        return
    if any(len(part["indices"]) != len(part["scores"]) for part in parts):
        raise RuntimeError("Missing CLIP scores on one or more evaluation ranks")

    pairs = sorted(
        (index, score)
        for part in parts
        for index, score in zip(part["indices"], part["scores"])
    )
    if len({index for index, _ in pairs}) != len(pairs):
        raise RuntimeError("Duplicate evaluation indices in CLIP score output")

    if rank == 0:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(
                {
                    "indices": [index for index, _ in pairs],
                    "scores": [score for _, score in pairs],
                },
                indent=2,
            )
            + "\n"
        )


def _make_shard_loader(
    val_ds,
    eval_indices: list[int],
    rank: int,
    world_size: int,
    batch_size: int,
    collate_fn,
    drop_last: bool,
) -> DataLoader:
    """Return a DataLoader over this rank's slice of ``eval_indices``.

    We deterministically slice ``eval_indices`` (already a flat list of
    dataset positions) round-robin across ranks so each rank sees a disjoint
    subset and the union covers exactly ``eval_indices``.
    """
    shard = eval_indices[rank::world_size]
    subset = Subset(val_ds, shard)
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=drop_last,
        collate_fn=collate_fn,
    )

# ---------------------------------------------------------------------------
# FID (text -> image)
# ---------------------------------------------------------------------------

# Cache the torchmetrics FID metric (which owns an InceptionV3 feature network)
# so it is built once and reused across eval calls. Re-instantiating it on every
# call -- there are ``len(cfg_scales) * len(ode_flags)`` calls per eval -- churns
# the CUDA caching allocator and fragments memory, which slows training down
# after the eval loop. We ``.reset()`` the cached metric between calls instead.
_FID_METRIC_CACHE: dict = {}


def _get_or_make_fid(feature: int, device: torch.device, process_group=None):
    key = (feature, str(device))
    fid = _FID_METRIC_CACHE.get(key)
    if fid is None:
        from torchmetrics.image.fid import FrechetInceptionDistance
        fid = FrechetInceptionDistance(
            feature=feature,
            normalize=False,           # we feed uint8 in [0, 255]
            reset_real_features=True,
            process_group=process_group,  # long-timeout eval group, not default
        ).to(device)
        fid.set_dtype(torch.float64)   # standard FID convention
        _FID_METRIC_CACHE[key] = fid
    else:
        fid.reset()
    return fid


_CLIP_CACHE: dict = {}


def _get_or_make_clip(device: torch.device):
    """Cache the CLIPScore model + processor (built once per device).

    Same rationale as ``_get_or_make_fid``: re-instantiating CLIP on every eval
    call re-reads ~600 MB of weights off Lustre on all ranks and churns the CUDA
    caching allocator, fragmenting memory and slowing training after the eval.
    """
    key = str(device)
    cached = _CLIP_CACHE.get(key)
    if cached is None:
        from transformers import CLIPModel, CLIPProcessor
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(device).eval()
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
        cached = (model, processor)
        _CLIP_CACHE[key] = cached
    return cached


_QWEN_CACHE: dict = {}


def _get_or_make_qwen(device: torch.device, model_id: str = "Qwen/Qwen3-1.7B"):
    """Cache the oracle causal LM + tokenizer used for generative perplexity.

    Same caching rationale as ``_get_or_make_clip``: the oracle is a few-GB
    checkpoint that would otherwise be re-read off Lustre on every eval call.
    On offline compute nodes the model must already live in the HF cache.
    """
    key = (model_id, str(device))
    cached = _QWEN_CACHE.get(key)
    if cached is None:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token_id is None:
            print(f"[rank {rank}] Qwen3 tokenizer has no pad token, setting to eos token")
            tokenizer.pad_token = tokenizer.eos_token
        model = (
            AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)
            .to(device)
            .eval()
        )
        cached = (model, tokenizer)
        _QWEN_CACHE[key] = cached
    return cached


@torch.no_grad()
def _generative_perplexity_totals(
    model,
    tokenizer,
    texts: list[str],
    device: torch.device,
    batch_size: int = 16,
    max_length: int = 128,
) -> tuple[float, float]:
    """Return ``(sum_nll, n_tokens)`` for ``texts`` under the oracle LM.

    Corpus-level generative perplexity is ``exp(sum_nll / n_tokens)``; returning
    the two sums (rather than a ratio) lets the caller SUM-all-reduce them across
    ranks so the perplexity is computed over the full eval set. Empty strings are
    dropped (they contribute no scored tokens). NLL is summed over next-token
    targets only (positions with attention_mask == 1, shifted by one).
    """
    texts = [t for t in texts if t and t.strip()]
    sum_nll = 0.0
    n_tokens = 0.0
    for start in range(0, len(texts), batch_size):
        chunk = texts[start:start + batch_size]
        enc = tokenizer(
            chunk, return_tensors="pt", padding=True,
            truncation=True, max_length=max_length,
        )
        input_ids = enc["input_ids"].to(device)
        attn = enc["attention_mask"].to(device)
        logits = model(input_ids=input_ids, attention_mask=attn).logits[:, :-1, :]
        labels = input_ids[:, 1:]
        mask = attn[:, 1:].to(torch.bool)
        logp = torch.log_softmax(logits.float(), dim=-1)
        nll = -logp.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        sum_nll += float(nll[mask].sum().item())
        n_tokens += float(mask.sum().item())
    return sum_nll, n_tokens


@torch.no_grad()
def compute_fid_distributed(
    *,
    eval_model: torch.nn.Module,
    sde,
    vae,
    val_ds,
    eval_indices: list[int],
    rank: int,
    world_size: int,
    device: torch.device,
    num_steps: int,
    batch_size: int,
    collate_fn,
    autocast_dtype: Optional[torch.dtype] = torch.bfloat16,
    feature: int = 2048,
    cfg_scale=0.0,
    ode=False,
    use_token_text_bridge: bool = False,
    token_layout: str = "row_major",
    x0_cond_source: str = "x0",
    runtime_config: BridgeRuntimeConfig = BRIDGE_RUNTIME_PRESETS["sd"],
    compute_clipscore: bool = True,
    clip_scores_path: Optional[str | Path] = None,
    eval_pg=None,
) -> dict[str, float]:
    """Sharded text->image FID over ``len(eval_indices)`` samples.

    Every rank must call this. The ``torchmetrics`` FID metric all-reduces
    its sufficient statistics inside ``.compute()`` over the default process
    group, so the returned FID is computed over the full sample set even
    though each rank only generated ``len(eval_indices)//world_size`` of
    them. Rank 0 returns ``{"eval/fid": <value>, ...}``; other ranks return
    ``{}`` (they still call ``.compute()`` to participate in the collective).
    """
    was_training = eval_model.training
    eval_model.eval()
    bridge_config = runtime_config.bridge

    loader = _make_shard_loader(
        val_ds=val_ds,
        eval_indices=eval_indices,
        rank=rank, world_size=world_size,
        batch_size=batch_size,
        collate_fn=collate_fn,
        drop_last=False,
    )

    # Reused across calls (built once); reset internally on each retrieval.
    fid = _get_or_make_fid(feature, device, process_group=eval_pg)

    # Text->image CLIPScore: alignment of the generated image with the GT
    # caption (prompt adherence), scored on the same images FID already decodes.
    clip_model = None
    clip_processor = None
    if compute_clipscore:
        try:
            clip_model, clip_processor = _get_or_make_clip(device)
        except Exception as e:
            print(f"[rank {rank}] T2I CLIPScore init failed, disabling: {e}")
            clip_model = None
            clip_processor = None
    clip_sum = 0.0
    clip_sum_sq = 0.0
    clip_n = 0.0
    clip_scores_local: list[float] = []

    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=autocast_dtype)
        if autocast_dtype is not None
        else nullcontext()
    )

    n_generated_local = 0
    with _temporarily_swap_score_network(sde, eval_model), autocast_ctx:
        for batch in tqdm(loader, desc="Computing FID"):
            x_0, latent_gt, y, x_cond_0, x_cond_1 = prepare_bridge_batch(
                batch,
                device,
                use_token_text_bridge=use_token_text_bridge,
                token_layout=token_layout,
                x0_cond_source=x0_cond_source,
                config=bridge_config,
            )
            assert not bridge_config.image_as_noise, "This will just make us generate noise"
            assert torch.allclose(latent_gt, x_cond_1)

            # conditioning signal may be different from physical x_0
            x_1_pred = sde.simulate(
                x_0, num_steps=num_steps, reverse=False, return_all=False,
                cfg_scale=cfg_scale, ode=ode, x_cond=x_cond_0, y=y,
            )

            real_uint8 = decode_latents(
                vae, latent_gt, to_cpu=False,
                latent_scale=runtime_config.latent_scale,
                latent_shift=runtime_config.latent_shift,
            )
            fake_uint8 = decode_latents(
                vae, x_1_pred.float(), to_cpu=False,
                latent_scale=runtime_config.latent_scale,
                latent_shift=runtime_config.latent_shift,
            )

            fid.update(real_uint8, real=True)
            fid.update(fake_uint8, real=False)
            n_generated_local += latent_gt.shape[0]

            if clip_model is not None:
                caps = batch.get("caption")
                if caps is not None:
                    caps = list(caps)
                    scores = _clip_score_manual(
                        clip_model, clip_processor, fake_uint8, caps, device
                    )
                    clip_sum += float(scores.sum().item())
                    clip_sum_sq += float(scores.square().sum().item())
                    clip_n += float(len(caps))
                    clip_scores_local.extend(scores.float().cpu().tolist())

    _save_distributed_clip_scores(
        clip_scores_path,
        clip_scores_local,
        eval_indices,
        rank,
        world_size,
        eval_pg,
    )

    # ``compute`` will all-reduce internal state across ranks.
    fid_value = fid.compute()

    # Sum samples + CLIPScore stats across ranks for logging.
    stats = torch.tensor(
        [n_generated_local, clip_sum, clip_n, clip_sum_sq], device=device, dtype=torch.float64
    )
    dist.all_reduce(stats, op=dist.ReduceOp.SUM, group=eval_pg)
    total = int(stats[0].item())
    clip_n_g = stats[2].item()

    if was_training:
        eval_model.train()

    if rank == 0:
        suffix = f"cfg_{cfg_scale}_{'ode' if ode else 'sde'}"
        out = {
            f"eval/fid_{suffix}": float(fid_value.item()),
            f"eval/fid_num_samples_{suffix}": total,
        }
        if clip_n_g > 0:
            clip_mean = stats[1].item() / clip_n_g
            out[f"eval/fid/clipscore_t2i_{suffix}"] = clip_mean
            out[f"eval/fid/clipscore_t2i_std_{suffix}"] = math.sqrt(
                max(stats[3].item() / clip_n_g - clip_mean**2, 0.0)
            )
        return out
    return {}


# ---------------------------------------------------------------------------
# Image -> Text decoded caption metrics
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_text_decode_distributed(
    *,
    eval_model: torch.nn.Module,
    sde,
    token_decoder: torch.nn.Module,
    tokenizer,
    val_ds,
    eval_indices: list[int],
    rank: int,
    world_size: int,
    device: torch.device,
    num_steps: int,
    batch_size: int,
    collate_fn,
    autocast_dtype: Optional[torch.dtype] = torch.bfloat16,
    cfg_scale=0.0,
    ode=False,
    token_layout: str = "row_major",
    runtime_config: BridgeRuntimeConfig = BRIDGE_RUNTIME_PRESETS["sd"],
    compute_cider: bool = True,
    compute_clipscore: bool = True,
    compute_genppl: bool = True,
    genppl_model: str = "Qwen/Qwen3-1.7B",
    genppl_batch_size: int = 16,
    include_padding_in_accuracy: bool = False,
    vae=None,
    clip_scores_path: Optional[str | Path] = None,
    eval_pg=None,
) -> dict[str, float]:
    was_training = eval_model.training
    eval_model.eval()
    token_decoder.eval()
    bridge_config = runtime_config.bridge
    token_scale = runtime_config.token_scale

    loader = _make_shard_loader(
        val_ds=val_ds,
        eval_indices=eval_indices,
        rank=rank, world_size=world_size,
        batch_size=batch_size,
        collate_fn=collate_fn,
        drop_last=False,
    )

    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=autocast_dtype)
        if autocast_dtype is not None
        else nullcontext()
    )

    token_correct = 0.0
    token_total = 0.0
    cider_sum = 0.0
    cider_n = 0.0
    clip_sum = 0.0
    clip_sum_sq = 0.0
    clip_n = 0.0
    clip_scores_local: list[float] = []
    # Stop-detection consistency: magnitude-based stop (truth) vs pad-id stop.
    stop_exact = 0.0      # examples where the two stop indices agree exactly
    stop_abs_diff = 0.0   # sum of |stop_norm - stop_pad| (in tokens)
    stop_n = 0.0          # number of examples compared
    first_norm_sum = 0.0  # sum of first-token embedding norms (expected ~1.0)

    # Mirror the dataset/train fallback so the pad-id stop matches the id that
    # was actually written into the masked token-id positions.
    pad_id = val_ds.token_pad_id
    assert pad_id is not None
    cider_metric = None
    clip_model = None
    clip_processor = None
    if compute_cider:
        try:
            from pycocoevalcap.cider.cider import Cider
            cider_metric = Cider()
        except Exception:
            cider_metric = None
    if compute_clipscore and vae is not None:
        try:
            clip_model, clip_processor = _get_or_make_clip(device)
        except Exception as e:
            print(f"[rank {rank}] CLIPScore init failed, disabling: {e}")
            clip_model = None
            clip_processor = None
    qwen_model = None
    qwen_tokenizer = None
    if compute_genppl:
        try:
            qwen_model, qwen_tokenizer = _get_or_make_qwen(device, genppl_model)
        except Exception as e:
            print(f"[rank {rank}] gen-ppl oracle init failed, disabling: {e}")
            qwen_model = None
            qwen_tokenizer = None

    assert not bridge_config.text_as_noise, "This will just make us generate noise"

    with _temporarily_swap_score_network(sde, eval_model), autocast_ctx:
        all_ref_caps: list[list[str]] = []
        all_pred_caps: list[list[str]] = []
        for batch in tqdm(loader, desc="Computing text decode"):
            latent = batch["latent"].to(device, non_blocking=True).float()
            token_ids = batch["text_token_ids"].to(device, non_blocking=True).long()
            token_mask = batch["text_token_mask"].to(device, non_blocking=True).bool()
            y = batch["prompt_kind_label"].to(device, non_blocking=True).long()
            B = latent.shape[0]
            # The latent is always conditioning, but the actual x1 may be noise if the image end is noise
            x1 = torch.randn_like(latent) if runtime_config.bridge.image_as_noise else latent

            x_0_pred = sde.simulate(
                x1, num_steps=num_steps, reverse=True, return_all=False,
                cfg_scale=cfg_scale, ode=ode, x_cond=latent, y=y,
            )
            token_flat = bridge_to_token_flat(
                x_0_pred.float(), layout=token_layout, config=bridge_config
            )
            logits = token_decoder(token_flat)
            pred_ids = logits.argmax(dim=-1)
            acc_token_mask = torch.ones_like(token_mask) if include_padding_in_accuracy else token_mask
            token_correct += float(((pred_ids == token_ids) & acc_token_mask).sum().item())
            token_total += float(acc_token_mask.sum().item())
            # NOTE: I expect the token accuracy to be low

            # Two independent estimates of where padding begins. The magnitude
            # check is the source of truth for truncating predicted captions;
            # the pad-id check is logged only for consistency monitoring.
            norm_stops, token_norms = norm_based_token_stops(
                x_0_pred.float(), token_scale=token_scale,
                layout=token_layout, zero_thresh=0.1, config=bridge_config,
            )
            pad_stops = pad_id_token_stops(pred_ids, pad_id)
            stop_exact += float((norm_stops == pad_stops).sum().item())
            stop_abs_diff += float((norm_stops - pad_stops).abs().sum().item())
            stop_n += float(B)
            first_norm_sum += float(token_norms[:, 0].sum().item())

            pred_caps: list[str] = []
            ref_caps: list[str] = []
            for i in range(B):
                length = int(token_mask[i].sum().item())
                pred_row = pred_ids[i].detach().cpu().tolist()
                stop = int(norm_stops[i].item())  # magnitude-based truth; honest (output-driven) length
                pred_caps.append(tokenizer.decode(pred_row[:stop], skip_special_tokens=True))
                ref_caps.append(tokenizer.decode(token_ids[i, :length].detach().cpu().tolist(), skip_special_tokens=True))

            # Collected unconditionally: CIDEr gathers them for corpus IDF, and
            # gen-ppl scores the same predicted/reference captions.
            all_ref_caps.extend(ref_caps)
            all_pred_caps.extend(pred_caps)

            if clip_model is not None and pred_caps:
                images = decode_latents(
                    vae, latent, to_cpu=False,
                    latent_scale=runtime_config.latent_scale,
                    latent_shift=runtime_config.latent_shift,
                )
                scores = _clip_score_manual(clip_model, clip_processor, images, pred_caps, device)
                clip_sum += float(scores.sum().item())
                clip_sum_sq += float(scores.square().sum().item())
                clip_n += float(len(pred_caps))
                clip_scores_local.extend(scores.float().cpu().tolist())

    _save_distributed_clip_scores(
        clip_scores_path,
        clip_scores_local,
        eval_indices,
        rank,
        world_size,
        eval_pg,
    )

    # Gather caption pairs across ranks so CIDEr's corpus-level IDF is computed
    # over the FULL eval set (not per-rank shards), then score once. The result
    # is identical on every rank, so the SUM all-reduce below preserves the mean.
    if cider_metric is not None:
        gathered_refs: list = [None] * world_size
        gathered_preds: list = [None] * world_size
        dist.all_gather_object(gathered_refs, all_ref_caps, group=eval_pg)
        dist.all_gather_object(gathered_preds, all_pred_caps, group=eval_pg)
        global_refs = [c for part in gathered_refs for c in part]
        global_preds = [c for part in gathered_preds for c in part]
        if global_refs and global_preds:
            assert len(global_refs) == len(global_preds), f"len(global_refs) != len(global_preds): {len(global_refs)} != {len(global_preds)}"
            refs = {i: [global_refs[i]] for i in range(len(global_refs))}
            hyps = {i: [global_preds[i]] for i in range(len(global_preds))}
            score, _ = cider_metric.compute_score(refs, hyps)
            cider_sum = float(score) * len(global_preds)
            cider_n = float(len(global_preds))

    # Generative perplexity: oracle-LM NLL over the decoded captions (and the
    # reference captions, as a baseline). Local NLL/token sums are SUM-all-reduced
    # so the reported perplexity is corpus-level over the full eval set.
    genppl_pred_nll = genppl_pred_tokens = 0.0
    genppl_ref_nll = genppl_ref_tokens = 0.0
    if qwen_model is not None:
        genppl_pred_nll, genppl_pred_tokens = _generative_perplexity_totals(
            qwen_model, qwen_tokenizer, all_pred_caps, device, batch_size=genppl_batch_size
        )
        genppl_ref_nll, genppl_ref_tokens = _generative_perplexity_totals(
            qwen_model, qwen_tokenizer, all_ref_caps, device, batch_size=genppl_batch_size
        )

    # All-reduce over ranks so every rank has the same averages.
    stats = torch.tensor(
        [token_correct, token_total, cider_sum, cider_n, clip_sum, clip_n,
         stop_exact, stop_abs_diff, stop_n, first_norm_sum,
         genppl_pred_nll, genppl_pred_tokens, genppl_ref_nll, genppl_ref_tokens,
         clip_sum_sq],
        device=device, dtype=torch.float64,
    )
    dist.all_reduce(stats, op=dist.ReduceOp.SUM, group=eval_pg)

    if was_training:
        eval_model.train()

    n_tokens = stats[1].item()
    if n_tokens <= 0:
        return {}

    # Metric values are batch-size-independent (per-token acc; per-sample CLIP;
    # corpus-level CIDEr/gen-ppl over the globally gathered captions), so the
    # keys carry no batch-size tag.
    tag = f"cfg_{cfg_scale}_{'ode' if ode else 'sde'}"
    out = {
        f"eval/text_decode/token_acc_{tag}": (stats[0] / stats[1]).item(),
        f"eval/text_decode/num_tokens_{tag}": int(stats[1].item()),
    }
    if stats[3].item() > 0:
        out[f"eval/text_decode/cider_{tag}"] = (stats[2] / stats[3]).item()
    if stats[5].item() > 0:
        clip_mean = (stats[4] / stats[5]).item()
        out[f"eval/text_decode/clipscore_i2t_{tag}"] = clip_mean
        out[f"eval/text_decode/clipscore_i2t_std_{tag}"] = math.sqrt(
            max((stats[14] / stats[5]).item() - clip_mean**2, 0.0)
        )
    if stats[11].item() > 0:
        out[f"eval/text_decode/genppl_{tag}"] = math.exp(stats[10].item() / stats[11].item())
    if stats[13].item() > 0:
        out[f"eval/text_decode/genppl_gt_{tag}"] = math.exp(stats[12].item() / stats[13].item())

    # Stop-detection diagnostics: how well the decoder's pad-id stop agrees with
    # the magnitude-based stop (truth), plus the mean first-token norm (~1.0).
    n_stop = stats[8].item()
    if n_stop > 0:
        suffix = f"_{tag}"
        mean_first_norm = stats[9].item() / n_stop
        out[f"eval/text_decode/stop_consistency_exact{suffix}"] = stats[6].item() / n_stop
        out[f"eval/text_decode/stop_consistency_mae{suffix}"] = stats[7].item() / n_stop
        out[f"eval/text_decode/first_token_norm{suffix}"] = mean_first_norm
        if rank == 0 and not (0.9 <= mean_first_norm <= 1.1):
            warnings.warn(
                f"[text decode] mean first-token embedding norm "
                f"{mean_first_norm:.4f} is outside [0.9, 1.1]; the bridge may "
                f"not be producing unit-norm content tokens (expected ~1.0)."
            )
    return out
