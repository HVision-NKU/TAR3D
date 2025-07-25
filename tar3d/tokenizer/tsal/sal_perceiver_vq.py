# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from typing import Optional
from einops import repeat
import math

from ..modules import checkpoint
from ..modules.embedder import FourierEmbedder
from ..modules.distributions import DiagonalGaussianDistribution
from ..modules.transformer_blocks import (
    ResidualCrossAttentionBlock,
    Transformer
)
from ..modules.triplane_blocks import TriplaneTransformer

from .tsal_base import ShapeAsLatentModule
from .triplane_decoder import TriplaneSynthesizer, UpsampleResNet_ConvTranspose
from .vq_model import VectorQuantizer, ModelArgs
import einops


class CrossAttentionEncoder(nn.Module):

    def __init__(self, *,
                 device: Optional[torch.device],
                 dtype: Optional[torch.dtype],
                 num_latents: int,
                 fourier_embedder: FourierEmbedder,
                 point_feats: int,
                 width: int,
                 heads: int,
                 layers: int,
                 init_scale: float = 0.25,
                 qkv_bias: bool = True,
                 flash: bool = False,
                 use_ln_post: bool = False,
                 use_checkpoint: bool = False):

        super().__init__()

        self.use_checkpoint = use_checkpoint
        self.num_latents = num_latents

        self.query = nn.Parameter(torch.randn((num_latents, width), device=device, dtype=dtype) * 0.02)

        self.fourier_embedder = fourier_embedder
        self.input_proj = nn.Linear(self.fourier_embedder.out_dim + point_feats, width, device=device, dtype=dtype)
        self.cross_attn = ResidualCrossAttentionBlock(
            device=device,
            dtype=dtype,
            width=width,
            heads=heads,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
            flash=flash,
        )

        self.self_attn = Transformer(
            device=device,
            dtype=dtype,
            n_ctx=num_latents,
            width=width,
            layers=layers,
            heads=heads,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
            flash=flash,
            use_checkpoint=False
        )

        if use_ln_post:
            self.ln_post = nn.LayerNorm(width, dtype=dtype, device=device)
        else:
            self.ln_post = None

    def _forward(self, pc, feats):
        """

        Args:
            pc (torch.FloatTensor): [B, N, 3]
            feats (torch.FloatTensor or None): [B, N, C]

        Returns:

        """

        bs = pc.shape[0]

        data = self.fourier_embedder(pc)
        if feats is not None:
            data = torch.cat([data, feats], dim=-1)
        data = self.input_proj(data)

        query = repeat(self.query, "m c -> b m c", b=bs)
        latents = self.cross_attn(query, data)
        latents = self.self_attn(latents)
        # print('latents in cross_att_encoder: ', latents.shape)

        if self.ln_post is not None:
            latents = self.ln_post(latents)

        return latents, pc

    def forward(self, pc: torch.FloatTensor, feats: Optional[torch.FloatTensor] = None):
        """

        Args:
            pc (torch.FloatTensor): [B, N, 3]
            feats (torch.FloatTensor or None): [B, N, C]

        Returns:
            dict
        """

        return checkpoint(self._forward, (pc, feats), self.parameters(), self.use_checkpoint)


class CrossAttentionDecoder(nn.Module):

    def __init__(self, *,
                 device: Optional[torch.device],
                 dtype: Optional[torch.dtype],
                 num_latents: int,
                 out_channels: int,
                 fourier_embedder: FourierEmbedder,
                 width: int,
                 heads: int,
                 init_scale: float = 0.25,
                 qkv_bias: bool = True,
                 flash: bool = False,
                 use_checkpoint: bool = False):

        super().__init__()

        self.use_checkpoint = use_checkpoint
        self.fourier_embedder = fourier_embedder

        self.query_proj = nn.Linear(self.fourier_embedder.out_dim, width, device=device, dtype=dtype)

        self.cross_attn_decoder = ResidualCrossAttentionBlock(
            device=device,
            dtype=dtype,
            n_data=num_latents,
            width=width,
            heads=heads,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
            flash=flash
        )

        self.ln_post = nn.LayerNorm(width, device=device, dtype=dtype)
        self.output_proj = nn.Linear(width, out_channels, device=device, dtype=dtype)

    def _forward(self, queries: torch.FloatTensor, latents: torch.FloatTensor):
        queries = self.query_proj(self.fourier_embedder(queries))
        x = self.cross_attn_decoder(queries, latents)
        x = self.ln_post(x)
        x = self.output_proj(x)
        return x

    def forward(self, queries: torch.FloatTensor, latents: torch.FloatTensor):
        return checkpoint(self._forward, (queries, latents), self.parameters(), self.use_checkpoint)


class ShapeAsLatentPerceiver(ShapeAsLatentModule):
    def __init__(self, *,
                 device: Optional[torch.device],
                 dtype: Optional[torch.dtype],
                 triplane_res: int,
                 point_feats: int = 0,
                 embed_dim: int = 0,
                 num_freqs: int = 8,
                 include_pi: bool = True,
                 width: int,
                 z_dim: int,
                 heads: int,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 init_scale: float = 0.25,
                 qkv_bias: bool = True,
                 flash: bool = False,
                 use_ln_post: bool = False,
                 use_checkpoint: bool = False):

        super().__init__()

        self.use_checkpoint = use_checkpoint
        self.triplane_res = triplane_res
        self.z_dim = z_dim
        self.width = width

        self.fourier_embedder = FourierEmbedder(num_freqs=num_freqs, include_pi=include_pi)

        init_scale = init_scale * math.sqrt(1.0 / width)
        num_latents = 1 + 3*triplane_res**2
        self.encoder = CrossAttentionEncoder(
            device=device,
            dtype=dtype,
            fourier_embedder=self.fourier_embedder,
            num_latents=num_latents,
            point_feats=point_feats,
            width=width,
            heads=heads,
            layers=num_encoder_layers,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
            flash=flash,
            use_ln_post=use_ln_post,
            use_checkpoint=use_checkpoint
        )

        self.embed_dim = embed_dim
        if embed_dim > 0:
            # VAE embed
            self.pre_kl = nn.Linear(width, embed_dim * 2, device=device, dtype=dtype)
            self.post_kl = nn.Linear(embed_dim, width, device=device, dtype=dtype)
            self.latent_shape = (num_latents, embed_dim)
        else:
            self.latent_shape = (num_latents, width)

          
        # self.preupsampler = Pre_upsampleResNet(in_channels=z_dim, device=device, dtype=dtype)
        # self.up_conv_decoder = UpsampleResNet_4layer(in_channels=z_dim, device=device, dtype=dtype)
        self.up_conv_decoder = UpsampleResNet_ConvTranspose(in_channels=z_dim, device=device, dtype=dtype)
        self.synthesizer = TriplaneSynthesizer(triplane_dim=32, samples_per_ray=128, device=device, dtype=dtype)

        self.latent_proj = nn.Linear(embed_dim if embed_dim > 0 else width, z_dim, device=device, dtype=dtype)

        # vq quanter by llamagen
        config = ModelArgs
        self.config = config
        self.quant = VectorQuantizer(config.codebook_size, config.codebook_embed_dim, 
            config.commit_loss_beta, config.entropy_loss_ratio,
            config.codebook_l2_norm, config.codebook_show_usage)

        self.quant_conv = nn.Conv2d(self.width, config.codebook_embed_dim, 1)
        self.post_quant_conv = nn.Conv2d(config.codebook_embed_dim, self.width, 1)
        
        self.transformer = TriplaneTransformer(
            device=None,
            dtype=None,
            n_ctx=3*32**2,
            width=768,
            layers=6,
            heads=12,
            init_scale=0.25,
            qkv_bias=False,
            flash=True,
            use_checkpoint=True
        )
        # self.trans_conv = nn.Conv2d(self.z_dim, 768, 1)
        self.post_trans_conv = nn.Conv2d(self.width, self.z_dim, 1)


    def encode(self,
               pc: torch.FloatTensor,
               feats: Optional[torch.FloatTensor] = None,
               sample_posterior: bool = True):
        """

        Args:
            pc (torch.FloatTensor): [B, N, 3]
            feats (torch.FloatTensor or None): [B, N, C]
            sample_posterior (bool):

        Returns:
            latents (torch.FloatTensor)
            center_pos (torch.FloatTensor or None):
            posterior (DiagonalGaussianDistribution or None):
        """

        latents, center_pos = self.encoder(pc, feats)
        latents = latents[:, 1:]
        posterior = None
        # if self.embed_dim > 0:
        #     moments = self.pre_kl(latents)
        #     posterior = DiagonalGaussianDistribution(moments, feat_dim=-1)

        #     if sample_posterior:
        #         latents = posterior.sample()
        #     else:
        #         latents = posterior.mode()
        # latents = self.latent_proj(latents)

        return latents, center_pos, posterior

    def decode(self, latents: torch.FloatTensor):
        latents = self.post_kl(latents)
        return self.transformer(latents)

    def query_geometry(self, queries: torch.FloatTensor, latents: torch.FloatTensor):
        logits = self.geo_decoder(queries, latents).squeeze(-1)
        return logits

    def decode_latent_to_triplane(self, latent_z: torch.FloatTensor):
        B = latent_z.shape[0]
        W = latent_z.shape[-1]
        H = latent_z.shape[-2] // 3
        latent_z = einops.rearrange(latent_z, 'B C (P H) W -> B (P H W) C', P=3)
        latent_z = self.transformer(latent_z)
        latent_z = einops.rearrange(latent_z, 'B (P H W) C -> B C (P H) W', P=3, W=W)
        latent_z =  self.post_trans_conv(latent_z)
        planes = self.up_conv_decoder(latent_z)
        dim, H, W = planes.shape[-3:]
        planes = planes.view(B, dim, 3, H//3, W).permute(0, 2, 1, 3, 4).contiguous()
        return planes

    def decode_triplane_to_sdf(self, volume_queries: torch.FloatTensor, planes: torch.FloatTensor):
        logits = torch.utils.checkpoint.checkpoint(
            self.synthesizer.get_geometry_prediction,
            planes, 
            volume_queries, 
            use_reentrant=False,
        )
        return logits.squeeze(-1)

    def forward(self,
                pc: torch.FloatTensor,
                feats: torch.FloatTensor,
                volume_queries: torch.FloatTensor,
                sample_posterior: bool = True):
        """

        Args:
            pc (torch.FloatTensor): [B, N, 3]
            feats (torch.FloatTensor or None): [B, N, C]
            volume_queries (torch.FloatTensor): [B, P, 3]
            sample_posterior (bool):

        Returns:
            logits (torch.FloatTensor): [B, P]
            center_pos (torch.FloatTensor): [B, M, 3]
            posterior (DiagonalGaussianDistribution or None).

        """
        # 1. encoder
        latents, center_pos, posterior = self.encode(pc, feats, sample_posterior=sample_posterior)
        B = latents.shape[0]
        latent_z = latents.view(B, 3*self.triplane_res, self.triplane_res, self.width).permute(0, 3, 1, 2).contiguous()     # [B, 768, 96, 32]

        # 2. vq quanter
        latent_z = self.quant_conv(latent_z)
        latent_z = einops.rearrange(latent_z, 'B C (P H) W -> (B P) C H W', P=3)
        latent_z, _, commit_loss, codebook_usage  = self.quant(latent_z)
        latent_z = einops.rearrange(latent_z, '(B P) C H W -> B C (P H) W', B=B)
        latent_z = self.post_quant_conv(latent_z)                                                                           # [B, 768, 96, 32]

        # 3. decoder
        planes = self.decode_latent_to_triplane(latent_z)
        logits = self.decode_triplane_to_sdf(volume_queries, planes)

        return logits, center_pos, posterior, commit_loss, codebook_usage

