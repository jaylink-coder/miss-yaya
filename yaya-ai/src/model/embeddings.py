"""Token embeddings and Rotary Positional Embeddings (RoPE).

RoPE encodes position information by rotating query/key vectors in pairs,
enabling relative position encoding with good length extrapolation.
Reference: https://arxiv.org/abs/2104.09864
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Tuple


class RotaryPositionalEmbedding(nn.Module):
    """Rotary Positional Embedding (RoPE).

    Applies rotation to pairs of dimensions in query and key vectors.
    Supports variable sequence lengths and good extrapolation beyond
    training context length.
    """

    def __init__(self, head_dim: int, max_position_embeddings: int = 4096, base: float = 10000.0):
        super().__init__()
        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        # Precompute inverse frequencies: theta_i = base^(-2i/d)
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Cache cos and sin values
        self._update_cache(max_position_embeddings)

    def _update_cache(self, seq_len: int):
        """Precompute cos and sin for positions up to seq_len."""
        t = torch.arange(seq_len, dtype=torch.float32, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(
        self, x: torch.Tensor, position_ids: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return cos and sin for the given positions.

        Args:
            x: Input tensor, used only to determine seq_len and device.
            position_ids: Optional explicit position indices [batch, seq_len].

        Returns:
            Tuple of (cos, sin) each shaped [1, seq_len, head_dim] or indexed by position_ids.
        """
        seq_len = x.shape[-2]

        if seq_len > self.cos_cached.shape[0]:
            self._update_cache(seq_len)

        if position_ids is not None:
            cos = self.cos_cached[position_ids].unsqueeze(1)
            sin = self.sin_cached[position_ids].unsqueeze(1)
        else:
            cos = self.cos_cached[:seq_len].unsqueeze(0).unsqueeze(0)
            sin = self.sin_cached[:seq_len].unsqueeze(0).unsqueeze(0)

        return cos.to(x.dtype), sin.to(x.dtype)


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary embeddings to query and key tensors.

    Args:
        q: Query tensor [batch, num_heads, seq_len, head_dim]
        k: Key tensor [batch, num_kv_heads, seq_len, head_dim]
        cos: Cosine values [1, 1, seq_len, head_dim]
        sin: Sine values [1, 1, seq_len, head_dim]

    Returns:
        Tuple of rotated (q, k) tensors.
    """
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half of the hidden dims of the input: [-x2, x1]."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class YayaEmbeddings(nn.Module):
    """Token embedding layer for Yaya model.

    Maps token IDs to dense vectors. No positional embedding here
    since RoPE is applied in the attention layer.
    """

    def __init__(self, vocab_size: int, hidden_size: int):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.hidden_size = hidden_size

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Embed token IDs.

        Args:
            input_ids: Token indices [batch, seq_len]

        Returns:
            Embeddings [batch, seq_len, hidden_size]
        """
        return self.word_embeddings(input_ids)
