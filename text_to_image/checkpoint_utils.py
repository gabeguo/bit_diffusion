def _resolve_ckpt(ckpt_path: str) -> str:
    """Resolve a checkpoint reference to a local file, downloading from the
    Hugging Face Hub when given an ``hf://[repo_id/]path/in/repo`` reference.
    """
    if not ckpt_path.startswith("hf://"):
        return ckpt_path
    from huggingface_hub import hf_hub_download
    ref = ckpt_path[len("hf://"):]
    parts = ref.split("/")
    repo_id, filename = "/".join(parts[:2]), "/".join(parts[2:])
    assert repo_id == "therealgabeguo/BiB_generative", "only the default repo is supported"
    return hf_hub_download(repo_id=repo_id, filename=filename)


# Shared inference loader; training imports only _resolve_ckpt, so keep heavy
# model imports inside load_model to avoid cycles and optional dependencies.
from dataclasses import dataclass
from typing import Optional
import json
from pathlib import Path
import torch
import torch.nn as nn


def _is_single_field_flow(args: dict) -> bool:
    return args["sde"] == "flow_matching" and args.get("force_unconditional", False)


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
    from sde_utils.sde import (UniformVolatilitySDE, PeriodicVolatilitySDE,
                               CosineDecayingVolatilitySDE, FlowMatchingODE)
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
            force_unconditional=args.get("force_unconditional", False),
            text_sigma=args.get("flow_text_sigma", 0.0),
            image_sigma=args.get("flow_image_sigma", 0.0),
        )
    raise ValueError(f"unknown sde {kind!r}")


def load_model(ckpt_path: str, data_root: str, device: torch.device) -> LoadedModel:
    """Rebuild a model + (optional) token decoder from a training checkpoint,
    loading the EMA weights (what training-time eval uses)."""
    from models.dit import DiT_models
    from models.token_decoder import SharedTokenDecoder
    from token_bridge import bridge_config_from_manifest, PROMPT_NUM_CLASSES
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
