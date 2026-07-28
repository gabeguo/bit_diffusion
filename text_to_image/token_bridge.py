from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass(frozen=True)
class TokenBridgeConfig:
    token_seq_len: int = 64
    token_emb_dim: int = 64
    bridge_shape: tuple[int, int, int] = (4, 32, 32)
    patch_size: int = 2
    text_as_noise: bool = False
    image_as_noise: bool = False

    @property
    def channels(self) -> int:
        return self.bridge_shape[0]

    @property
    def height(self) -> int:
        return self.bridge_shape[1]

    @property
    def width(self) -> int:
        return self.bridge_shape[2]

    @property
    def patch_h(self) -> int:
        return self.height // self.patch_size

    @property
    def patch_w(self) -> int:
        return self.width // self.patch_size

    @property
    def num_patches(self) -> int:
        return self.patch_h * self.patch_w

    @property
    def patch_payload_dim(self) -> int:
        return self.channels * self.patch_size * self.patch_size

    @property
    def patches_per_token(self) -> int:
        return self.token_emb_dim // self.patch_payload_dim

    @property
    def token_flat_dim(self) -> int:
        return self.token_seq_len * self.token_emb_dim

    @property
    def bridge_flat_dim(self) -> int:
        return self.channels * self.height * self.width

    def validate(self) -> None:
        if self.height % self.patch_size != 0 or self.width % self.patch_size != 0:
            raise ValueError(
                f"bridge_shape={self.bridge_shape} must be divisible by patch_size={self.patch_size}."
            )
        if self.token_emb_dim % self.patch_payload_dim != 0:
            raise ValueError(
                f"token_emb_dim={self.token_emb_dim} must be divisible by "
                f"patch_payload_dim={self.patch_payload_dim}."
            )
        if self.token_flat_dim != self.bridge_flat_dim:
            raise ValueError(
                f"token_seq_len * token_emb_dim ({self.token_flat_dim}) must exactly "
                f"equal bridge volume ({self.bridge_flat_dim})."
            )
        if self.token_seq_len * self.patches_per_token != self.num_patches:
            raise ValueError(
                f"token_seq_len * patches_per_token "
                f"({self.token_seq_len * self.patches_per_token}) must exactly "
                f"equal num_patches ({self.num_patches})."
            )


@dataclass(frozen=True)
class BridgeRuntimeConfig:
    bridge: TokenBridgeConfig
    bridge_preset: str
    vae_model: str
    vae_subfolder: str | None = None
    vae_kind: str = "sd"
    latent_scale: float = 0.18215
    latent_shift: float = 0.0

    @property
    def token_scale(self) -> float:
        return self.bridge.token_emb_dim ** 0.5


TOKEN_BRIDGE_CONFIGS = {
    "sd": TokenBridgeConfig(
        token_seq_len=64,
        token_emb_dim=64,
        bridge_shape=(4, 32, 32),
        patch_size=2,
    ),
    "flux": TokenBridgeConfig(
        token_seq_len=128,
        token_emb_dim=128,
        bridge_shape=(16, 32, 32),
        patch_size=2,
    ),
}

BRIDGE_RUNTIME_PRESETS = {
    "sd": BridgeRuntimeConfig(
        bridge=TOKEN_BRIDGE_CONFIGS["sd"],
        bridge_preset="sd",
        vae_model="stabilityai/sd-vae-ft-mse",
        vae_kind="sd",
        latent_scale=0.18215,
        latent_shift=0.0,
    ),
    "flux": BridgeRuntimeConfig(
        bridge=TOKEN_BRIDGE_CONFIGS["flux"],
        bridge_preset="flux",
        vae_model="black-forest-labs/FLUX.1-dev",
        vae_subfolder="vae",
        vae_kind="flux",
        latent_scale=0.3611,
        latent_shift=0.1159,
    ),
}

for _runtime in BRIDGE_RUNTIME_PRESETS.values():
    _runtime.bridge.validate()

DEFAULT_TOKEN_BRIDGE_CONFIG = TOKEN_BRIDGE_CONFIGS["sd"]
DEFAULT_TOKEN_BRIDGE_CONFIG.validate()

TOKEN_SEQ_LEN = DEFAULT_TOKEN_BRIDGE_CONFIG.token_seq_len
TOKEN_EMB_DIM = DEFAULT_TOKEN_BRIDGE_CONFIG.token_emb_dim
BRIDGE_SHAPE = DEFAULT_TOKEN_BRIDGE_CONFIG.bridge_shape
PATCH_SIZE = DEFAULT_TOKEN_BRIDGE_CONFIG.patch_size
PATCH_GRID = DEFAULT_TOKEN_BRIDGE_CONFIG.patch_h
PATCH_PAYLOAD_DIM = DEFAULT_TOKEN_BRIDGE_CONFIG.patch_payload_dim
PATCHES_PER_TOKEN = DEFAULT_TOKEN_BRIDGE_CONFIG.patches_per_token

PROMPT_KIND_TO_LABEL = {"original": 0, "short": 1, "medium": 2}
PROMPT_LABEL_TO_KIND = {v: k for k, v in PROMPT_KIND_TO_LABEL.items()}
PROMPT_NUM_CLASSES = len(PROMPT_KIND_TO_LABEL)

TOKEN_LAYOUTS = ("row_major",)#, "local_2x2")


def runtime_from_preset(name: str) -> BridgeRuntimeConfig:
    try:
        return BRIDGE_RUNTIME_PRESETS[str(name)]
    except KeyError as exc:
        raise ValueError(
            f"Unknown bridge preset {name!r}; expected one of {tuple(BRIDGE_RUNTIME_PRESETS)}."
        ) from exc


def bridge_config_from_manifest(root: str | Path, preset: str = "auto") -> BridgeRuntimeConfig:
    """Resolve bridge geometry from a dataset manifest, falling back to SD for old data."""
    root = Path(root)
    manifest = root / "config.json"
    cfg = json.loads(manifest.read_text()).get("config", {}) if manifest.exists() else {}
    name = str(cfg.get("bridge_preset") or cfg.get("vae_kind") or "sd")
    if preset != "auto":
        name = preset
    base = runtime_from_preset(name)
    bridge = base.bridge

    if "latent_shape" in cfg:
        latent_shape = tuple(int(x) for x in cfg["latent_shape"])
        if latent_shape != bridge.bridge_shape:
            raise ValueError(
                f"{manifest} latent_shape={latent_shape} does not match "
                f"{name} preset bridge_shape={bridge.bridge_shape}."
            )

    token_cfg_path = root / "token_embed_config.json"
    token_cfg = (
        json.loads(token_cfg_path.read_text()).get("config", {})
        if token_cfg_path.exists()
        else {}
    )
    stored_seq = int(token_cfg.get("token_seq_len", bridge.token_seq_len))
    stored_dim = int(token_cfg.get("mrl_dim", token_cfg.get("token_emb_dim", bridge.token_emb_dim)))
    if stored_seq < bridge.token_seq_len or stored_dim < bridge.token_emb_dim:
        raise ValueError(
            f"{token_cfg_path} stores {stored_seq}x{stored_dim}, but "
            f"{name} training needs {bridge.token_seq_len}x{bridge.token_emb_dim}."
        )

    return base


def prompt_kind_to_label(kind: str) -> int:
    try:
        return PROMPT_KIND_TO_LABEL[str(kind)]
    except KeyError as exc:
        raise ValueError(f"Unknown prompt kind {kind!r}") from exc


def _patch_positions(
    layout: str,
    device: torch.device,
    config: TokenBridgeConfig = DEFAULT_TOKEN_BRIDGE_CONFIG,
) -> torch.Tensor:
    config.validate()
    if layout not in TOKEN_LAYOUTS:
        raise ValueError(f"Unknown token layout {layout!r}; expected one of {TOKEN_LAYOUTS}.")

    if layout == "row_major":
        positions = torch.arange(
            config.token_seq_len * config.patches_per_token,
            device=device,
            dtype=torch.long,
        )
        return positions.view(config.token_seq_len, config.patches_per_token)

    raise ValueError(f"For now, we only support row_major layout; got {layout!r}. (Code for local_2x2 layout has been vetted, but it's kinda complicated, and I don't think it has any benefit.)")
    """
    # Place tokens on an 8x8 token grid; each token owns a local 2x2 block of
    # patches in the 16x16 DiT patch grid.
    if config.patches_per_token != 4:
        raise ValueError(
            "local_2x2 layout requires exactly four patches per token; "
            f"got {config.patches_per_token}."
        )
    token_grid_h = config.patch_h // 2
    token_grid_w = config.patch_w // 2
    if token_grid_h * token_grid_w != config.token_seq_len:
        raise ValueError(
            "local_2x2 layout requires token_seq_len to fill the 2x2 patch-block grid; "
            f"got token_seq_len={config.token_seq_len}, grid={token_grid_h}x{token_grid_w}."
        )
    positions = torch.empty(
        (config.token_seq_len, config.patches_per_token),
        device=device,
        dtype=torch.long,
    )
    for token_idx in range(config.token_seq_len):
        tr = token_idx // token_grid_w
        tc = token_idx % token_grid_w
        pr = 2 * tr
        pc = 2 * tc
        positions[token_idx] = torch.tensor(
            [
                pr * config.patch_w + pc,
                pr * config.patch_w + pc + 1,
                (pr + 1) * config.patch_w + pc,
                (pr + 1) * config.patch_w + pc + 1,
            ],
            device=device,
            dtype=torch.long,
        )
    return positions
    """


def _unpatchify_payloads(
    payloads: torch.Tensor,
    config: TokenBridgeConfig = DEFAULT_TOKEN_BRIDGE_CONFIG,
) -> torch.Tensor:
    """Map patch payloads shaped like DiT outputs back to (B, 4, 32, 32)."""
    config.validate()
    bsz = payloads.shape[0]
    p = config.patch_size
    c = config.channels
    h = config.patch_h
    w = config.patch_w
    # NOTE: this implies that most of the content is in the first patch per token (and first "pixel" per patch), due to MRL embeddings. The pixel content asymmetry should be fine, since Conv blocks look at the whole patch anyways. I also think the patch content asymmetry should be fine, since we have attention. (Maybe it's even easier, since the model can just focus on these.)
    # NOTE: by reshaping this way, we keep the token/patch content together (that is, having same h and w guarantees that the values refer to the same token/patch)
    x = payloads.reshape(bsz, h, w, p, p, c)
    x = torch.einsum("nhwpqc->nchpwq", x)
    # NOTE: the h*p and w*p keeps each patch together spatially, as the conv expects (but patches within a token may be scattered, depending on the layout)
    return x.reshape(bsz, c, h * p, w * p)


def _patchify_bridge(
    x: torch.Tensor,
    config: TokenBridgeConfig = DEFAULT_TOKEN_BRIDGE_CONFIG,
) -> torch.Tensor:
    """Inverse of _unpatchify_payloads for bridge tensors."""
    config.validate()
    bsz, c, height, width = x.shape
    expected_c, expected_h, expected_w = config.bridge_shape
    if (c, height, width) != (expected_c, expected_h, expected_w):
        raise ValueError(f"Expected bridge shape (B, {config.bridge_shape}), got {tuple(x.shape)}")
    p = config.patch_size
    h = height // p
    w = width // p
    x = x.reshape(bsz, c, h, p, w, p)
    x = torch.einsum("nchpwq->nhwpqc", x)
    return x.reshape(bsz, h * w, p * p * c)


def token_flat_to_bridge(
    token_flat: torch.Tensor,
    layout: str = "row_major",
    config: TokenBridgeConfig = DEFAULT_TOKEN_BRIDGE_CONFIG,
) -> torch.Tensor:
    """Pack flattened 64x64 token embeddings into the bridge's 4x32x32 tensor."""
    # NOTE: token embeddings are basically flattened patchified, which is why we need to do unpatchify only at the end
    config.validate()
    if token_flat.ndim != 2 or token_flat.shape[1] != config.token_flat_dim:
        raise ValueError(
            f"Expected (B, {config.token_flat_dim}) token_flat, got {tuple(token_flat.shape)}"
        )
    bsz = token_flat.shape[0]
    # (word, patch_per_word, features_per_patch)
    # NOTE: this is actually unnecessary, just helps interpretability
    token_payloads = token_flat.reshape(
        bsz,
        config.token_seq_len,
        config.patches_per_token,
        config.patch_payload_dim,
    )
    # NOTE: seems that this would focus most of the content into the first patch per token, since we have MRL embeddings.
    # (patches, features_per_patch)
    patch_payloads = token_flat.new_zeros(
        (bsz, config.num_patches, config.patch_payload_dim)
    )
    positions = _patch_positions(layout, token_flat.device, config=config)
    assert torch.all(positions.reshape(-1) == torch.arange(config.num_patches, device=token_flat.device))
    patch_payloads[:, positions.reshape(-1), :] = token_payloads.reshape(
        bsz,
        config.token_seq_len * config.patches_per_token,
        config.patch_payload_dim,
    )
    # (patches, features_per_patch) = (h * w, patch_dim^2 * c) -> (c, h * patch_dim, w * patch_dim)
    return _unpatchify_payloads(patch_payloads, config=config)


def bridge_to_token_flat(
    x: torch.Tensor,
    layout: str = "row_major",
    config: TokenBridgeConfig = DEFAULT_TOKEN_BRIDGE_CONFIG,
) -> torch.Tensor:
    """Unpack a bridge tensor back to flattened 64x64 token embeddings. Inverse of token_flat_to_bridge."""
    config.validate()
    # (c, h*p, w*p) -> (h*w, p^2 * c)
    patch_payloads = _patchify_bridge(x, config=config)
    positions = _patch_positions(layout, x.device, config=config)
    # NOTE: this is a no-op for row_major layout
    token_payloads = patch_payloads[:, positions.reshape(-1), :]
    assert torch.all(token_payloads == patch_payloads)
    return token_payloads.reshape(x.shape[0], config.token_flat_dim)


def prepare_bridge_batch(
    batch: dict,
    device: torch.device,
    *,
    use_token_text_bridge: bool,
    token_layout: str = "row_major",
    x0_cond_source: str = "x0",
    non_blocking: bool = True,
    config: TokenBridgeConfig = DEFAULT_TOKEN_BRIDGE_CONFIG,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Map a dataset batch to ``(x_0, x_1, y, x_cond_0)`` for the bridge."""
    if x0_cond_source != "x0":
        raise ValueError("global_text conditioning is deprecated; use --x0-cond-source x0.")
    if not use_token_text_bridge:
        raise ValueError("prepare_bridge_batch now requires --use-token-text-bridge.")
    latent = batch["latent"].to(device, non_blocking=non_blocking).float()
    bsz = latent.shape[0]
    assert latent.shape == (bsz, *config.bridge_shape), latent.shape

    x_1 = latent
    token_emb = batch["text_token_emb"].to(device, non_blocking=non_blocking).float()
    assert token_emb.shape == (bsz, config.token_flat_dim), token_emb.shape
    x_0 = token_flat_to_bridge(token_emb, layout=token_layout, config=config)
    y = batch["prompt_kind_label"].to(device, non_blocking=non_blocking).long()

    x_0_process = x_0 if not config.text_as_noise else torch.randn_like(x_0)
    x_1_process = x_1 if not config.image_as_noise else torch.randn_like(x_1)
    x_0_cond = x_0
    x_1_cond = x_1

    return x_0_process, x_1_process, y, x_0_cond, x_1_cond


# ---------------------------------------------------------------------------
# Padding / stop detection on token embeddings
#
# Two independent ways to locate where a caption ends and padding begins:
#   * magnitude: real token embeddings are unit-norm and padding is the zero
#     vector, so an unscaled per-token norm near zero marks padding. This keys
#     off the bridge network's own regression target and is decoder-independent.
#   * decoded ids: the first token the decoder emits as ``pad_id``.
# ---------------------------------------------------------------------------

def compute_token_norms(
    x_0_bridge: torch.Tensor,
    *,
    token_scale: float,
    layout: str = "row_major",
    config: TokenBridgeConfig = DEFAULT_TOKEN_BRIDGE_CONFIG,
) -> torch.Tensor:
    """Per-token L2 norm of the unscaled token embeddings packed in a bridge tensor.

    Args:
        x_0_bridge: ``(B, C, H, W)`` bridge-space tensor (e.g. a predicted x_0).
        token_scale: divisor applied before measuring norms, to undo the
            dataset's ``TEXT_TOKEN_SCALE``. Must be passed explicitly.

    Returns:
        ``(B, token_seq_len)`` per-token norms. Ground-truth content tokens are
        unit-norm and padding is zero, so a well-trained bridge yields norms
        near 1.0 for content and near 0.0 for padding.
    """
    token_flat = bridge_to_token_flat(x_0_bridge, layout=layout, config=config)
    toks = token_flat.view(x_0_bridge.shape[0], config.token_seq_len, config.token_emb_dim)
    return (toks / token_scale).norm(dim=-1)


def _first_true_index(mask: torch.Tensor) -> torch.Tensor:
    """First ``True`` column per row of a ``(B, T)`` bool mask, else ``T``."""
    bsz, length = mask.shape
    idx = torch.arange(length, device=mask.device).unsqueeze(0).expand(bsz, length)
    filled = torch.where(mask, idx, torch.full_like(idx, length))
    retval = filled.min(dim=1).values
    if (retval == 0).any():
        warnings.warn("token stop at position 0: degenerate empty prediction")
    return retval


def norm_based_token_stops(
    x_0_bridge: torch.Tensor,
    *,
    token_scale: float,
    layout: str = "row_major",
    zero_thresh: float = 0.1,
    config: TokenBridgeConfig = DEFAULT_TOKEN_BRIDGE_CONFIG,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Locate where padding starts via token-embedding magnitude.

    The stop index for each row is the position of the first token whose
    unscaled norm falls below ``zero_thresh`` (the first ~zero "padding"
    embedding), or ``token_seq_len`` if there is none. ``token_scale`` (the
    dataset's ``TEXT_TOKEN_SCALE``) must be passed explicitly.

    Returns ``(stops, norms)`` with shapes ``(B,)`` and ``(B, token_seq_len)``.
    """
    norms = compute_token_norms(
        x_0_bridge, layout=layout, token_scale=token_scale, config=config
    )
    stops = _first_true_index(norms < zero_thresh)
    return stops, norms


def pad_id_token_stops(pred_ids: torch.Tensor, pad_id: int) -> torch.Tensor:
    """Locate where padding starts via decoded token ids.

    The stop index for each row is the position of the first token equal to
    ``pad_id``, or ``T`` if there is none. ``pred_ids`` is ``(B, T)``.
    """
    return _first_true_index(pred_ids == int(pad_id))
