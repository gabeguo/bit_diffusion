"""
Evaluation + plotting for the text-to-image DiT bridge.

Produces VAE-decoded PNGs and image-only, text-only, and side-by-side MP4s for
both cycle directions, subsampled every `decode_every_k` steps.

Also computes validation loss (forward + reverse directions) on the supplied
val batch. Everything is single-rank — call only from rank 0.
"""

from __future__ import annotations

import math
import textwrap
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Optional

import torch
import torchvision
from PIL import Image, ImageDraw, ImageFont

from sde_utils.loss import dsm_loss, sample_p_base_x_t_cond_x_0_x_1
from token_bridge import (
    BRIDGE_RUNTIME_PRESETS,
    BridgeRuntimeConfig,
    TokenBridgeConfig,
    bridge_to_token_flat,
    norm_based_token_stops,
    prepare_bridge_batch,
)
from sde_utils.precond import EDMScoreWrapper


# ---------------------------------------------------------------------------
# Latent <-> image conversion
# ---------------------------------------------------------------------------

@torch.no_grad()
def decode_latents(
    vae,
    latents: torch.Tensor,
    unscale: bool = True,
    to_cpu: bool = True,
    latent_scale: float = BRIDGE_RUNTIME_PRESETS["sd"].latent_scale,
    latent_shift: float = 0.0,
) -> torch.Tensor:
    """Decode SD-1.x VAE latents to uint8 RGB images.

    Args:
        vae: a loaded ``diffusers.AutoencoderKL``.
        latents: ``(B, 4, H, W)`` tensor on the VAE's device.
        unscale: if True, applies ``latent_scale`` / ``latent_shift`` to invert the
            ``scale_latents=True`` convention used by the dataset.

    Returns:
        ``(B, 3, 8H, 8W)`` uint8 CPU tensor with values in ``[0, 255]``.
    """
    if unscale:
        latents = latents / latent_scale + latent_shift
    out = vae.decode(latents.to(vae.dtype)).sample  # roughly [-1, 1]
    out = (out.clamp(-1, 1) + 1) * 127.5
    return out.to(torch.uint8).cpu() if to_cpu else out.to(torch.uint8)


def _save_grid_png(images_uint8: torch.Tensor, path: Path, nrow: int) -> None:
    grid = torchvision.utils.make_grid(images_uint8, nrow=nrow, padding=2)
    arr = grid.permute(1, 2, 0).numpy()
    Image.fromarray(arr).save(path)


def _save_trajectory_video(
    frames_uint8_list: list[torch.Tensor],
    path: Path,
    fps: int,
    nrow: int,
) -> None:
    """Tile each timestep's batch into a grid then write an MP4."""
    grids = [
        torchvision.utils.make_grid(frame, nrow=nrow, padding=2)
        for frame in frames_uint8_list
    ]
    _write_video(grids, path, fps)


def _write_video(grids: list[torch.Tensor], path: Path, fps: int) -> None:
    import imageio.v2 as imageio

    imageio.mimwrite(
        str(path),
        [grid.permute(1, 2, 0).numpy() for grid in grids],
        fps=fps,
        codec="libx264",
        quality=8,
        macro_block_size=1,
    )


def _render_text_panels(
    texts: list[str], height: int, width: int,
) -> torch.Tensor:
    """Render one readable text panel per batch item."""
    font_size = max(12, min(22, width // 14))
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()
    margin, spacing = max(6, width // 32), max(2, font_size // 5)
    chars_per_line = max(12, (width - 2 * margin) // max(1, font_size // 2))
    max_lines = max(1, (height - 2 * margin) // (font_size + spacing))
    panels = []
    for text in texts:
        lines = textwrap.wrap(text or "<empty>", width=chars_per_line) or ["<empty>"]
        if len(lines) > max_lines:
            lines = lines[:max_lines]
            lines[-1] = lines[-1][:-3] + "..."
        panel = Image.new("RGB", (width, height), "white")
        ImageDraw.Draw(panel).multiline_text(
            (margin, margin), "\n".join(lines), fill="black",
            font=font, spacing=spacing,
        )
        panels.append(torchvision.transforms.functional.pil_to_tensor(panel))
    return torch.stack(panels)


@torch.no_grad()
def _decode_texts(
    states: torch.Tensor,
    token_decoder,
    token_tokenizer,
    token_scale: float,
    token_layout: str,
    bridge_config: TokenBridgeConfig,
) -> list[str]:
    token_flat = bridge_to_token_flat(
        states.float(), layout=token_layout, config=bridge_config,
    )
    token_ids = token_decoder(token_flat).argmax(dim=-1)
    stops, _ = norm_based_token_stops(
        states.float(), token_scale=token_scale,
        layout=token_layout, zero_thresh=0.1, config=bridge_config,
    )
    return [
        token_tokenizer.decode(
            token_ids[i, :int(stops[i])].tolist(), skip_special_tokens=True,
        )
        for i in range(token_ids.shape[0])
    ]


def _save_multimodal_trajectory(
    image_frames: list[torch.Tensor],
    text_frames: Optional[list[list[str]]],
    image_path: Path,
    text_path: Path,
    combined_path: Path,
    fps: int,
    nrow: int,
) -> bool:
    """Save image-only and, when available, text-only and side-by-side videos."""
    image_grids = [
        torchvision.utils.make_grid(frame, nrow=nrow, padding=2)
        for frame in image_frames
    ]
    _write_video(image_grids, image_path, fps)
    if text_frames is None:
        return False

    height, width = image_frames[0].shape[-2:]
    text_grids = [
        torchvision.utils.make_grid(
            _render_text_panels(texts, height, width), nrow=nrow, padding=2,
        )
        for texts in text_frames
    ]
    _write_video(text_grids, text_path, fps)
    _write_video(
        [torch.cat((image, text), dim=2) for image, text in zip(image_grids, text_grids)],
        combined_path, fps,
    )
    return True


# ---------------------------------------------------------------------------
# SDE simulation with snapshots
# ---------------------------------------------------------------------------

@contextmanager
def _temporarily_swap_score_network(sde, new_net):
    """Use ``new_net`` as the SDE's score network inside the block.

    If the score network is an ``EDMScoreWrapper``, swap the raw net inside it
    so the preconditioning is preserved around ``new_net``.
    """
    if hasattr(sde.score_network, "net"):
        assert isinstance(sde.score_network, EDMScoreWrapper)
        holder, attr = sde.score_network, "net"
    else:
        holder, attr = sde, "score_network"
    orig = getattr(holder, attr)
    setattr(holder, attr, new_net)
    try:
        yield
    finally:
        setattr(holder, attr, orig)


@torch.no_grad()
def simulate_with_snapshots(
    sde, x_start, num_steps, decode_every_k, reverse=False,
    cfg_scale=0.0, ode=False, x_cond=None, y=None,
):
    all_x_t = sde.simulate(
        x_start, num_steps=num_steps, reverse=reverse, return_all=True,
        cfg_scale=cfg_scale, ode=ode, x_cond=x_cond, y=y,
    )
    return [x_start.clone()] + [
        all_x_t[i] for i in range(num_steps)
        if (i + 1) % decode_every_k == 0 or i == num_steps - 1
    ]


# ---------------------------------------------------------------------------
# Top-level eval entry point
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_and_log_visuals(
    *,
    eval_model: torch.nn.Module,   # un-wrapped model (e.g. EMA) for scoring
    sde,
    vae,
    val_batch: dict,
    step: int,
    device: torch.device,
    out_dir: str | Path,
    num_steps: int = 500,
    decode_every_k: int = 20,
    n_decode: int = 8,
    grid_nrow: Optional[int] = None,
    fps: int = 10,
    wandb_logger=None,
    autocast_dtype: Optional[torch.dtype] = torch.bfloat16,
    cfg_scale=0.0,
    ode=False,
    use_token_text_bridge: bool = False,
    token_layout: str = "row_major",
    x0_cond_source: str = "x0",
    runtime_config: BridgeRuntimeConfig = BRIDGE_RUNTIME_PRESETS["sd"],
    token_decoder: Optional[torch.nn.Module] = None,
    token_tokenizer=None,
    cycle_text_tokenizer=None,
    cycle_text_model=None,
    cycle_text_max_length: int = 128,
    no_reverse: bool = False,
    no_forward: bool = False,
) -> dict[str, float]:
    """Produce eval artifacts (PNGs, MP4s) and return a dict of val losses.

    ``val_batch`` is expected to contain ``latent`` ``(B, 4, 32, 32)`` and
    ``text_emb`` ``(B, 4096)`` tensors, and optionally a ``caption`` list.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bridge_config = runtime_config.bridge
    token_scale = runtime_config.token_scale

    was_training = eval_model.training
    eval_model.eval()

    # ---- Prep small batch for visualization
    captions = val_batch.get("caption", [""] * n_decode)
    captions = list(captions)[:n_decode]
    vis_batch = {
        key: value[:n_decode] if hasattr(value, "__getitem__") else value
        for key, value in val_batch.items()
    }
    x_0, x_1_gt, y, x_cond_0, x_cond_1 = prepare_bridge_batch(
        vis_batch,
        device,
        use_token_text_bridge=use_token_text_bridge,
        token_layout=token_layout,
        x0_cond_source=x0_cond_source,
        config=bridge_config,
    )

    if not bridge_config.image_as_noise:
        assert torch.allclose(x_1_gt, x_cond_1), "We should have the same x_1_gt and x_cond_1"
    else:
        assert not torch.allclose(x_1_gt, x_cond_1), "We should have different x_1_gt and x_cond_1"
    if not bridge_config.text_as_noise:
        assert torch.allclose(x_0, x_cond_0), "We should have the same x_0 and x_cond_0"
    else:
        assert not torch.allclose(x_0, x_cond_0), "We should have different x_0 and x_cond_0"
    
    B = x_1_gt.shape[0]
    nrow = grid_nrow or int(math.ceil(math.sqrt(B)))
    if nrow < 1:
        raise ValueError("grid_nrow must be positive or None")

    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=autocast_dtype)
        if autocast_dtype is not None
        else nullcontext()
    )

    with _temporarily_swap_score_network(sde, eval_model), autocast_ctx:
      # ---- Forward: text -> image
      if (not no_forward) and (not bridge_config.image_as_noise):
        fwd_snaps = simulate_with_snapshots(
            sde=sde, x_start=x_0,
            num_steps=num_steps, decode_every_k=decode_every_k,
            reverse=False,
            cfg_scale=cfg_scale,
            ode=ode,
            x_cond=x_cond_0,
            y=y,
        )
        x_1_pred = fwd_snaps[-1]
      else:
        fwd_snaps = x_1_pred = None

      # ---- Reverse: image -> text
      if (not no_reverse) and (not bridge_config.text_as_noise):
        rev_to_text = simulate_with_snapshots(
            sde=sde, x_start=x_1_gt,
            num_steps=num_steps, decode_every_k=decode_every_k,
            reverse=True,
            cfg_scale=cfg_scale,
            ode=ode,
            x_cond=x_cond_1, # conditioning signal may be different from physical x_1
            y=y,
        )
        x_0_pred = rev_to_text[-1]
      else:
        rev_to_text = x_0_pred = None

    # Get the text captions back
    pred_captions: list[str] = []
    x_cond_cycle = None
    if x0_cond_source == "x0":
        x_cond_cycle = x_0_pred
    if (
        use_token_text_bridge
        and token_decoder is not None
        and token_tokenizer is not None
        and (not no_reverse)
        and (not bridge_config.text_as_noise)
    ):
        token_decoder.eval()
        pred_captions = _decode_texts(
            x_0_pred, token_decoder, token_tokenizer, token_scale,
            token_layout, bridge_config,
        )

        # re-embed global text
        if x0_cond_source == "global_text":
            raise ValueError("No longer supporting that.")
            
    if x_cond_cycle is not None \
    and (not bridge_config.text_as_noise) and (not no_reverse) \
    and (not bridge_config.image_as_noise) and (not no_forward):
      with _temporarily_swap_score_network(sde, eval_model), autocast_ctx:
        cyc_back = simulate_with_snapshots(
            sde=sde, x_start=x_0_pred,
            num_steps=num_steps, decode_every_k=decode_every_k,
            reverse=False,
            cfg_scale=cfg_scale,
            ode=ode,
            x_cond=x_cond_cycle,
            y=y,
        )
        x_1_back = cyc_back[-1]
    else:
        cyc_back = x_1_back = None

    # Complete the opposite cycle: text -> image -> text.
    if x_1_pred is not None \
    and (not bridge_config.text_as_noise) and (not no_reverse) \
    and (not bridge_config.image_as_noise) and (not no_forward):
      with _temporarily_swap_score_network(sde, eval_model), autocast_ctx:
        rev_back = simulate_with_snapshots(
            sde=sde, x_start=x_1_pred,
            num_steps=num_steps, decode_every_k=decode_every_k,
            reverse=True,
            cfg_scale=cfg_scale,
            ode=ode,
            x_cond=x_1_pred,
            y=y,
        )
    else:
        rev_back = None

    # ---- Decode + save still PNGs (VAE not under autocast; we want it in
    #      whatever dtype it was loaded in)
    gt_imgs = decode_latents(
        vae, x_1_gt.float(),
        latent_scale=runtime_config.latent_scale,
        latent_shift=runtime_config.latent_shift,
    ) if x_1_gt is not None else None
    
    fwd_imgs = decode_latents(
        vae, x_1_pred.float(),
        latent_scale=runtime_config.latent_scale,
        latent_shift=runtime_config.latent_shift,
    ) if x_1_pred is not None else None
    cyc_imgs = decode_latents(
        vae, x_1_back.float(),
        latent_scale=runtime_config.latent_scale,
        latent_shift=runtime_config.latent_shift,
    ) if x_1_back is not None else None

    gt_path = out_dir / f"step_{step:07d}_gt_cfg_{cfg_scale}_{'ode' if ode else 'sde'}.png"
    fwd_path = out_dir / f"step_{step:07d}_fwd_cfg_{cfg_scale}_{'ode' if ode else 'sde'}.png"
    cyc_path = out_dir / f"step_{step:07d}_cycle_cfg_{cfg_scale}_{'ode' if ode else 'sde'}.png"
    if gt_imgs is not None:
      _save_grid_png(gt_imgs, gt_path, nrow=nrow)
    if fwd_imgs is not None:
      _save_grid_png(fwd_imgs, fwd_path, nrow=nrow)
    if cyc_imgs is not None:
      _save_grid_png(cyc_imgs, cyc_path, nrow=nrow)

    # ---- Trajectory videos (decoded on the fly, snapshot-by-snapshot, to keep
    #      peak memory bounded)
    def decode_trajectory(snapshots):
        return [
            decode_latents(
                vae, state.float(),
                latent_scale=runtime_config.latent_scale,
                latent_shift=runtime_config.latent_shift,
            )
            for state in snapshots
        ]

    fwd_traj_decoded = decode_trajectory(fwd_snaps) if fwd_snaps is not None else None
    fwd_video_path = out_dir / f"step_{step:07d}_fwd_trajectory_cfg_{cfg_scale}_{'ode' if ode else 'sde'}.mp4"
    if fwd_traj_decoded is not None:
      _save_trajectory_video(fwd_traj_decoded, fwd_video_path, fps=fps, nrow=nrow)
      fwd_traj_decoded = True

    can_decode_text = (
        use_token_text_bridge
        and token_decoder is not None
        and token_tokenizer is not None
    )
    mode = "ode" if ode else "sde"
    cycle_outputs = {}
    cycles = {
        "image_to_text_to_image": (
            rev_to_text + cyc_back[1:]
            if rev_to_text is not None and cyc_back is not None else None
        ),
        "text_to_image_to_text": (
            fwd_snaps + rev_back[1:]
            if fwd_snaps is not None and rev_back is not None else None
        ),
    }
    for cycle_name, snapshots in cycles.items():
        if snapshots is None:
            continue
        image_frames = decode_trajectory(snapshots)
        text_frames = [
            _decode_texts(
                state, token_decoder, token_tokenizer, token_scale,
                token_layout, bridge_config,
            )
            for state in snapshots
        ] if can_decode_text else None
        paths = {
            kind: out_dir / (
                f"step_{step:07d}_{cycle_name}_{kind}_cfg_{cfg_scale}_{mode}.mp4"
            )
            for kind in ("image", "text", "combined")
        }
        has_text = _save_multimodal_trajectory(
            image_frames, text_frames,
            paths["image"], paths["text"], paths["combined"],
            fps, nrow,
        )
        cycle_outputs[cycle_name] = (paths, has_text)

    # ---- Captions sidecar
    if any(c for c in captions) and (not no_reverse) and (not bridge_config.text_as_noise):
        with open(out_dir / f"step_{step:07d}_captions_cfg_{cfg_scale}_{'ode' if ode else 'sde'}.txt", "w") as f:
            for i, c in enumerate(captions):
                f.write(f"{i:02d}: {c}\n")
                if i < len(pred_captions):
                    f.write(f"\t{i:02d} predicted: {pred_captions[i]}\n")

    if wandb_logger is not None:
        import wandb
        log_dict: dict = dict()
        if gt_imgs is not None:
            log_dict[f"eval/gt_cfg_{cfg_scale}_{'ode' if ode else 'sde'}"] = wandb.Image(str(gt_path), caption="ground-truth image latents (VAE decoded)")
        if fwd_imgs is not None:
            log_dict[f"eval/forward_cfg_{cfg_scale}_{'ode' if ode else 'sde'}"] = wandb.Image(str(fwd_path), caption="text -> image")
        if cyc_imgs is not None:
            log_dict[f"eval/cycle_cfg_{cfg_scale}_{'ode' if ode else 'sde'}"] = wandb.Image(str(cyc_path), caption="image -> text -> image")
        if fwd_traj_decoded is not None:
            log_dict[f"eval/forward_trajectory_cfg_{cfg_scale}_{'ode' if ode else 'sde'}"] = wandb.Video(str(fwd_video_path), fps=fps)
        for cycle_name, (paths, has_text) in cycle_outputs.items():
            log_dict[f"eval/{cycle_name}_image_cfg_{cfg_scale}_{mode}"] = (
                wandb.Video(str(paths["image"]), fps=fps)
            )
            if has_text:
                for kind in ("text", "combined"):
                    log_dict[f"eval/{cycle_name}_{kind}_cfg_{cfg_scale}_{mode}"] = (
                        wandb.Video(str(paths[kind]), fps=fps)
                    )
        if any(c for c in captions) and (not no_reverse) and (not bridge_config.text_as_noise):
            log_dict[f"eval/captions_cfg_{cfg_scale}_{'ode' if ode else 'sde'}"] = wandb.Table(
                columns=["idx", "caption", "predicted_caption"],
                data=[[i, c, pred_captions[i] if i < len(pred_captions) else ""] for i, c in enumerate(captions)],
            )
        wandb_logger.log(log_dict, step=step)

    if was_training:
        eval_model.train()

    return

