"""Vision Transformer (ViT) Encoder for Yaya multimodal model.

Converts images into a sequence of patch embeddings that can be projected
into the LLM's embedding space for multimodal understanding.

Architecture:
    Image -> Patch Embedding -> [ViT Block x M] -> Patch Features
    
Reference: https://arxiv.org/abs/2010.11929
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

from src.model.normalization import RMSNorm
from src.utils.config import VisionConfig


class PatchEmbedding(nn.Module):
    """Convert image into a sequence of flattened patch embeddings.
    
    Splits image into non-overlapping patches, then linearly projects
    each flattened patch into an embedding vector.
    
    For a 336x336 image with 14x14 patches: 24x24 = 576 patches.
    """

    def __init__(self, image_size: int = 336, patch_size: int = 14, hidden_size: int = 1024):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.hidden_size = hidden_size

        # Conv2d acts as a linear projection of flattened patches
        self.projection = nn.Conv2d(
            in_channels=3,
            out_channels=hidden_size,
            kernel_size=patch_size,
            stride=patch_size,
            bias=True,
        )

        # Learnable position embeddings for each patch
        self.position_embedding = nn.Embedding(self.num_patches, hidden_size)
        self.register_buffer(
            "position_ids",
            torch.arange(self.num_patches).unsqueeze(0),
            persistent=False,
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Embed image patches.

        Args:
            pixel_values: Images [batch, 3, image_size, image_size]

        Returns:
            Patch embeddings [batch, num_patches, hidden_size]
        """
        batch_size = pixel_values.shape[0]

        # [B, 3, H, W] -> [B, hidden_size, H/P, W/P]
        patch_embeds = self.projection(pixel_values)

        # [B, hidden_size, H/P, W/P] -> [B, num_patches, hidden_size]
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)

        # Add positional embeddings
        position_ids = self.position_ids[:, :patch_embeds.shape[1]]
        patch_embeds = patch_embeds + self.position_embedding(position_ids)

        return patch_embeds


class VisionAttention(nn.Module):
    """Standard multi-head self-attention for the vision encoder.
    
    Uses full MHA (not GQA) since the vision encoder is smaller
    and doesn't need KV-cache optimization.
    """

    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Multi-head self-attention.

        Args:
            x: Input [batch, seq_len, hidden_size]

        Returns:
            Output [batch, seq_len, hidden_size]
        """
        B, N, C = x.shape

        q = self.q_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # Use PyTorch SDPA for fused attention
        if hasattr(F, "scaled_dot_product_attention"):
            attn_output = F.scaled_dot_product_attention(q, k, v, scale=self.scale)
        else:
            attn_weights = (q @ k.transpose(-2, -1)) * self.scale
            attn_weights = F.softmax(attn_weights, dim=-1)
            attn_output = attn_weights @ v

        attn_output = attn_output.transpose(1, 2).reshape(B, N, C)
        return self.out_proj(attn_output)


class VisionMLP(nn.Module):
    """Feed-forward network for vision encoder blocks."""

    def __init__(self, hidden_size: int, intermediate_size: Optional[int] = None):
        super().__init__()
        intermediate_size = intermediate_size or hidden_size * 4
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class VisionTransformerBlock(nn.Module):
    """Single ViT encoder block: LayerNorm + Attention + LayerNorm + MLP."""

    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.attn = VisionAttention(hidden_size, num_heads)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.mlp = VisionMLP(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionEncoder(nn.Module):
    """Full Vision Transformer encoder.

    Converts raw images into a sequence of visual feature vectors
    ready to be projected into the LLM's embedding space.

    Architecture:
        Image [B, 3, H, W]
        -> PatchEmbedding -> [B, num_patches, vision_hidden]
        -> VisionTransformerBlock x N
        -> LayerNorm
        -> Visual features [B, num_patches, vision_hidden]
    """

    def __init__(self, config: VisionConfig):
        super().__init__()
        self.config = config

        self.patch_embed = PatchEmbedding(
            image_size=config.image_size,
            patch_size=config.patch_size,
            hidden_size=config.vision_hidden_size,
        )

        self.blocks = nn.ModuleList([
            VisionTransformerBlock(config.vision_hidden_size, config.vision_heads)
            for _ in range(config.vision_layers)
        ])

        self.norm = nn.LayerNorm(config.vision_hidden_size)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Encode images into visual features.

        Args:
            pixel_values: Images [batch, 3, image_size, image_size]

        Returns:
            Visual features [batch, num_patches, vision_hidden_size]
        """
        x = self.patch_embed(pixel_values)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        return x
