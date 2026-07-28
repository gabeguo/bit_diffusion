# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
from functools import partial
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


#################################################################################
#                                 Core DiT Model                                #
#################################################################################

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1000,
        learn_sigma=True,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x, t, y):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        t = self.t_embedder(t)                   # (N, D)
        y = self.y_embedder(y, self.training)    # (N, D)
        c = t + y                                # (N, D)
        for block in self.blocks:
            x = block(x, c)                      # (N, T, D)
        x = self.final_layer(x, c)                # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)                   # (N, out_channels, H, W)
        return x

    def forward_with_cfg(self, x, t, y, cfg_scale):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)


class CrossAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, qkv_bias=True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = hidden_size // num_heads
        self.scale = head_dim ** -0.5
        self.q = nn.Linear(hidden_size, hidden_size, bias=qkv_bias)
        self.k = nn.Linear(hidden_size, hidden_size, bias=qkv_bias)
        self.v = nn.Linear(hidden_size, hidden_size, bias=qkv_bias)
        self.proj = nn.Linear(hidden_size, hidden_size)

        # prevent explosion by projecting onto hypersphere
        # TODO: temperature?
        self.q_norm = nn.RMSNorm(head_dim, elementwise_affine=False)
        self.k_norm = nn.RMSNorm(head_dim, elementwise_affine=False)

        assert hidden_size % num_heads == 0
        return

    def forward(self, x, cond_tokens):
        N, T, D = x.shape
        assert cond_tokens.shape == (N, T, D)

        assert D % self.num_heads == 0

        q = self.q(x)
        k = self.k(cond_tokens)
        v = self.v(cond_tokens)

        assert q.shape == k.shape == v.shape == (N, T, D)
        
        q = q.reshape(N, T, self.num_heads, D // self.num_heads)
        q = q.permute(0, 2, 1, 3).contiguous()

        k = k.reshape(N, T, self.num_heads, D // self.num_heads)
        k = k.permute(0, 2, 1, 3).contiguous()

        v = v.reshape(N, T, self.num_heads, D // self.num_heads)
        v = v.permute(0, 2, 1, 3).contiguous()

        assert q.shape == k.shape == v.shape == (N, self.num_heads, T, D // self.num_heads)

        with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
            out = F.scaled_dot_product_attention(
                query=self.q_norm(q),
                key=self.k_norm(k),
                value=v,
                # is_causal=False,
                # scale=self.scale,
            )
        assert out.shape == (N, self.num_heads, T, D // self.num_heads)
        out = out.permute(0, 2, 1, 3).reshape(N, T, D)

        return self.proj(out)

class DiTBlockWithCrossAttention(DiTBlock):
    """
    An (inherited) DiT block with adaptive layer norm zero (adaLN-Zero) conditioning and cross attention.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__(hidden_size, num_heads, mlp_ratio=mlp_ratio, **block_kwargs)
        self.norm_cross = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.cross_attn = CrossAttention(hidden_size, num_heads=num_heads, qkv_bias=True) # TODO: **kwargs? dropout?
        self.norm_cond = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 12 * hidden_size, bias=True)
        )
        return

    def forward(self, x, c, cond_tokens):
        shift_msa, scale_msa, gate_msa, shift_xa, scale_xa, gate_xa, shift_cond, scale_cond, gate_cond, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(c).chunk(12, dim=1)

        assert isinstance(self.attn.q_norm, nn.RMSNorm)
        assert isinstance(self.attn.k_norm, nn.RMSNorm)

        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))

        N, T, D = x.shape
        assert x.shape == cond_tokens.shape

        x = x + gate_xa.unsqueeze(1) * self.cross_attn(
            modulate(self.norm_cross(x), shift_xa, scale_xa), 
            modulate(self.norm_cond(cond_tokens), shift_cond, scale_cond), 
        )
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x

class DiTWithCrossAttention(DiT):
    """
    A DiT model (inherited) with cross attention.
    """
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1000,
        forward_cond_scale=1.0,
        use_repa_text=False,
        repa_text_dim=1024,
        repa_text_layer=None,
        use_repa_image=False,
        repa_image_dim=1024,
        repa_image_layer=None,
        repa_train_reverse=True,
    ):
        super().__init__(
            input_size=input_size, patch_size=patch_size, in_channels=in_channels, hidden_size=hidden_size, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio, class_dropout_prob=class_dropout_prob, num_classes=num_classes,
            learn_sigma=False,
        )
        # Multiplier on x_cond in the forward direction, e.g. sqrt(4096) = 64
        # to bring the unit-norm global text embedding to unit per-coordinate
        # scale (matching everything else the trunk sees). Reverse-direction
        # conditioning (image latents) is already unit-scale.
        self.forward_cond_scale = forward_cond_scale
        assert not self.learn_sigma, "DiTWithCrossAttention must be initialized with learn_sigma=False"

        # have separate patch and timestep embedders for the conditioning images
        self.cond_embedder_forward = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True) # condition on text
        self.cond_embedder_reverse = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True) # condition on image
        # separate cond_pos_embed not necessary, as positional embeddings are fixed
        self.cond_t_embedder = TimestepEmbedder(hidden_size)

        self.blocks = nn.ModuleList([
            DiTBlockWithCrossAttention(
                hidden_size=hidden_size, 
                num_heads=num_heads, 
                mlp_ratio=mlp_ratio,
                qk_norm=True,
                norm_layer=partial(nn.RMSNorm, elementwise_affine=False)
            ) for i in range(depth)
        ])

        # REPA (REPresentation Alignment): tap an intermediate block's tokens
        # and project them to be cosine-aligned against an external target in
        # the loss. Two independent flavours:
        #   text : mean-pool tokens -> project to a (truncated) global text
        #          embedding (one global vector per sample).
        #   image: project EACH token -> the corresponding DINOv2 patch token
        #          (spatial, 16x16 == 256 latent patches == 256 DINO patches).
        # Forward / reverse see different conditioning, so each direction gets
        # its own head; the reverse heads are only built when the reverse
        # direction is trained, else they'd be unused params under DDP.
        self.use_repa_text = use_repa_text
        self.use_repa_image = use_repa_image
        self.repa_train_reverse = repa_train_reverse
        if use_repa_text:
            self.repa_text_dim = repa_text_dim
            self.repa_text_layer = (depth // 3) if repa_text_layer is None else repa_text_layer
            assert 0 <= self.repa_text_layer < depth, (
                f"repa_text_layer={self.repa_text_layer} out of range for depth={depth}"
            )
            self.repa_text_head_forward = self._build_repa_head(hidden_size, repa_text_dim)
            if repa_train_reverse:
                self.repa_text_head_reverse = self._build_repa_head(hidden_size, repa_text_dim)
        if use_repa_image:
            self.repa_image_dim = repa_image_dim
            self.repa_image_layer = (depth // 3) if repa_image_layer is None else repa_image_layer
            assert 0 <= self.repa_image_layer < depth, (
                f"repa_image_layer={self.repa_image_layer} out of range for depth={depth}"
            )
            self.repa_image_head_forward = self._build_repa_head(hidden_size, repa_image_dim)
            if repa_train_reverse:
                self.repa_image_head_reverse = self._build_repa_head(hidden_size, repa_image_dim)

        self.initialize_weights() # call initialization again to initialize the cross attention blocks

        self.time_scale = 1000 # we have timesteps between 0 and 1, but we want to embed them in the desired range (i.e., 0 to 1000)

        return

    @staticmethod
    def _build_repa_head(hidden_size, repa_dim):
        return nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, repa_dim),
        )
    
    def initialize_weights(self):
        super().initialize_weights() # this takes care of the blocks
        if not hasattr(self, 'cond_t_embedder'):
            return

        # Initialize conditioning's timestep embedding MLP:
        nn.init.normal_(self.cond_t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.cond_t_embedder.mlp[2].weight, std=0.02)

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        for cond_embedder in [self.cond_embedder_forward, self.cond_embedder_reverse]:
            w = cond_embedder.proj.weight.data
            nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
            nn.init.constant_(cond_embedder.proj.bias, 0)

        print("Initialized conditioning embedders")

        return

    def forward(self, x, t, y, x_cond, reverse=False, cond_mask=None, return_repa=False):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps: must be between 0 and 1
        y: (N,) tensor of class labels
        x_cond: (N, C, H, W) tensor of conditional images
        reverse: bool indicating whether we go in reverse time or forward time. True if x_cond is from time 1 (image), False if x_cond is from time 0 (text)
        cond_mask: (N,) bool tensor indicating which batch elements should look at the conditioning image
        return_repa: if True, also return a dict of REPA projections from the
            tapped intermediate block(s): {"text": (N, repa_text_dim)} (mean-
            pooled) and/or {"image": (N, T, repa_image_dim)} (per-token).
            Requires the model to be built with use_repa_text/use_repa_image.
        """
        if return_repa:
            assert self.use_repa_text or self.use_repa_image, (
                "return_repa=True requires use_repa_text and/or use_repa_image"
            )
            assert (not reverse) or self.repa_train_reverse, (
                "return_repa=True with reverse=True requires repa_train_reverse=True"
            )

        N, C, H, W = x.shape
        assert x_cond.shape == (N, C, H, W)
        assert H == W
        assert C in (4, 16)

        assert torch.all(t <= 1.0) and torch.all(t >= 0.0), "Timesteps must be between 0 and 1"
        assert t.dtype == torch.float32

        if reverse:
            t_cond = torch.ones_like(t) # condition on endpoint (image)
        else:
            t_cond = torch.zeros_like(t) # condition on start point (text)

        x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        t = self.t_embedder(t * self.time_scale)                   # (N, D)
        t_cond = self.cond_t_embedder(t_cond * self.time_scale)    # (N, D)
        y = self.y_embedder(y, self.training)    # (N, D)
        c = (t + t_cond) / 2 + y                 # (N, D)

        if reverse:
            cond_tokens = self.cond_embedder_reverse(x_cond) + self.pos_embed
        else:
            cond_tokens = self.cond_embedder_forward(x_cond * self.forward_cond_scale) + self.pos_embed

        if cond_mask is None:
            cond_mask = torch.ones((N,), dtype=torch.bool, device=x.device)
        assert cond_tokens.shape == x.shape and len(cond_tokens.shape) == 3
        assert cond_mask.shape == (N,) == (cond_tokens.shape[0],)
        cond_tokens = cond_tokens * cond_mask.view(-1, 1, 1)

        need_text = return_repa and self.use_repa_text
        need_image = return_repa and self.use_repa_image
        feat_text = feat_image = None
        for block_idx, block in enumerate(self.blocks):
            x = block(x, c, cond_tokens)                      # (N, T, D)
            if need_text and block_idx == self.repa_text_layer:
                feat_text = x
            if need_image and block_idx == self.repa_image_layer:
                feat_image = x
        if return_repa:
            repa: dict[str, torch.Tensor] = {}
            if need_text:
                head = self.repa_text_head_reverse if reverse else self.repa_text_head_forward
                repa["text"] = head(feat_text.mean(dim=1))    # (N, repa_text_dim)
            if need_image:
                head = self.repa_image_head_reverse if reverse else self.repa_image_head_forward
                repa["image"] = head(feat_image)              # (N, T, repa_image_dim)
        x = self.final_layer(x, c)                # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)                   # (N, out_channels, H, W)
        if return_repa:
            return x, repa
        return x

    def forward_with_cfg(self, x, t, y, x_cond, reverse=False, cfg_scale=0.0):
        cond_output = self.forward(
            x=x,
            t=t,
            y=y,
            x_cond=x_cond,
            reverse=reverse,
            cond_mask=None,
        )
        uncond_output = self.forward(
            x=x,
            t=t,
            y=y,
            x_cond=x_cond,
            reverse=reverse,
            cond_mask=torch.zeros((t.shape[0],), dtype=torch.bool, device=x.device),
        )
        return cond_output + cfg_scale * (cond_output - uncond_output)


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


#################################################################################
#                                   DiT Configs                                  #
#################################################################################

def DiT_XL_2(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

def DiT_XL_4(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=4, num_heads=16, **kwargs)

def DiT_XL_8(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=8, num_heads=16, **kwargs)

def DiT_L_2(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)

def DiT_L_4(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=4, num_heads=16, **kwargs)

def DiT_L_8(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=8, num_heads=16, **kwargs)

def DiT_B_2(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)

def DiT_B_4(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs)

def DiT_B_8(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=8, num_heads=12, **kwargs)

def DiT_S_2(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)

def DiT_S_4(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)

def DiT_S_8(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)

# DiT Cross Attention Models
def DiTXA_XL_2(**kwargs):
    return DiTWithCrossAttention(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

def DiTXA_B_2(**kwargs):
    return DiTWithCrossAttention(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)

def DiTXA_M_2(**kwargs):
    return DiTWithCrossAttention(depth=16, hidden_size=896, patch_size=2, num_heads=14, **kwargs)

def DiTXA_B_2_double_depth(**kwargs):
    return DiTWithCrossAttention(depth=24, hidden_size=768, patch_size=2, num_heads=12, **kwargs)

def DiTXA_L_2(**kwargs):
    return DiTWithCrossAttention(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)

def DiTXA_B_4(**kwargs):
    return DiTWithCrossAttention(depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs)

def DiTXA_B_8(**kwargs):
    return DiTWithCrossAttention(depth=12, hidden_size=768, patch_size=8, num_heads=12, **kwargs)

def DiTXA_S_2(**kwargs):
    return DiTWithCrossAttention(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)

def DiTXA_S_4(**kwargs):
    return DiTWithCrossAttention(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)

def DiTXA_S_8(**kwargs):
    return DiTWithCrossAttention(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)


DiT_models = {
    'DiT-XL/2': DiT_XL_2,  'DiT-XL/4': DiT_XL_4,  'DiT-XL/8': DiT_XL_8,
    'DiT-L/2':  DiT_L_2,   'DiT-L/4':  DiT_L_4,   'DiT-L/8':  DiT_L_8,
    'DiT-B/2':  DiT_B_2,   'DiT-B/4':  DiT_B_4,   'DiT-B/8':  DiT_B_8,
    'DiT-S/2':  DiT_S_2,   'DiT-S/4':  DiT_S_4,   'DiT-S/8':  DiT_S_8,
    'DiTXA-B/2': DiTXA_B_2, 'DiTXA-B/4': DiTXA_B_4, 'DiTXA-B/8': DiTXA_B_8,
    'DiTXA-M/2': DiTXA_M_2,
    'DiTXA-L/2': DiTXA_L_2,
    'DiTXA-S/2': DiTXA_S_2, 'DiTXA-S/4': DiTXA_S_4, 'DiTXA-S/8': DiTXA_S_8,
    'DiTXA-XL/2': DiTXA_XL_2,
    'DiTXA-B/2-double-depth': DiTXA_B_2_double_depth,
}