"""Grouped-Query Attention (GQA) with RoPE and optional FlashAttention.

GQA shares key-value heads across multiple query heads, reducing KV-cache
memory during inference while maintaining most of multi-head attention quality.
Reference: https://arxiv.org/abs/2305.13245
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

from src.model.embeddings import RotaryPositionalEmbedding, apply_rotary_pos_emb

# Try to import FlashAttention for optimized attention computation
try:
    from flash_attn import flash_attn_func
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False


class GroupedQueryAttention(nn.Module):
    """Grouped-Query Attention (GQA).

    Multiple query heads share fewer key-value heads. When num_kv_heads == num_heads,
    this is standard Multi-Head Attention. When num_kv_heads == 1, this is Multi-Query
    Attention.

    Architecture:
        Input [B, S, H] -> Q projection [B, S, num_heads * head_dim]
                        -> K projection [B, S, num_kv_heads * head_dim]
                        -> V projection [B, S, num_kv_heads * head_dim]
        -> RoPE on Q, K
        -> Expand KV heads to match Q heads
        -> Scaled Dot-Product Attention
        -> Output projection [B, S, H]
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        max_position_embeddings: int = 4096,
        rope_theta: float = 10000.0,
        attention_dropout: float = 0.0,
        attention_bias: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.num_kv_heads = num_key_value_heads
        self.head_dim = head_dim
        self.num_key_value_groups = self.num_heads // self.num_kv_heads
        self.attention_dropout = attention_dropout

        # Projection layers (no bias following LLaMA/Mistral pattern)
        self.q_proj = nn.Linear(hidden_size, num_attention_heads * head_dim, bias=attention_bias)
        self.k_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=attention_bias)
        self.v_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=attention_bias)
        self.o_proj = nn.Linear(num_attention_heads * head_dim, hidden_size, bias=attention_bias)

        # Rotary positional embeddings
        self.rotary_emb = RotaryPositionalEmbedding(
            head_dim=head_dim,
            max_position_embeddings=max_position_embeddings,
            base=rope_theta,
        )

        self.scaling = 1.0 / math.sqrt(head_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Forward pass for grouped-query attention.

        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_size]
            attention_mask: Causal mask [batch, 1, seq_len, kv_seq_len]
            position_ids: Position indices [batch, seq_len]
            past_key_value: Cached (key, value) for incremental decoding
            use_cache: Whether to return updated cache

        Returns:
            Tuple of (output [batch, seq_len, hidden_size], optional cache)
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Project to Q, K, V
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape: [B, S, num_heads * head_dim] -> [B, num_heads, S, head_dim]
        query_states = query_states.view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            batch_size, seq_len, self.num_kv_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            batch_size, seq_len, self.num_kv_heads, self.head_dim
        ).transpose(1, 2)

        # Apply RoPE to queries and keys
        cos, sin = self.rotary_emb(query_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Handle KV cache for incremental decoding
        if past_key_value is not None:
            past_key, past_value = past_key_value
            key_states = torch.cat([past_key, key_states], dim=2)
            value_states = torch.cat([past_value, value_states], dim=2)

        new_cache = (key_states, value_states) if use_cache else None

        # Expand KV heads to match query heads for GQA
        # [B, num_kv_heads, S, D] -> [B, num_heads, S, D]
        key_states = self._repeat_kv(key_states)
        value_states = self._repeat_kv(value_states)

        # Compute attention
        if FLASH_ATTN_AVAILABLE and not use_cache and attention_mask is None:
            attn_output = self._flash_attention(query_states, key_states, value_states)
        else:
            attn_output = self._standard_attention(
                query_states, key_states, value_states, attention_mask
            )

        # Reshape back: [B, num_heads, S, head_dim] -> [B, S, hidden_size]
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, -1)

        # Output projection
        attn_output = self.o_proj(attn_output)

        return attn_output, new_cache

    def _repeat_kv(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Repeat KV heads to match query head count for GQA.

        [B, num_kv_heads, S, D] -> [B, num_heads, S, D]
        """
        if self.num_key_value_groups == 1:
            return hidden_states

        batch, num_kv_heads, slen, head_dim = hidden_states.shape
        hidden_states = hidden_states[:, :, None, :, :].expand(
            batch, num_kv_heads, self.num_key_value_groups, slen, head_dim
        )
        return hidden_states.reshape(batch, num_kv_heads * self.num_key_value_groups, slen, head_dim)

    def _standard_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Standard scaled dot-product attention with causal mask.

        Uses PyTorch's built-in SDPA when possible for kernel fusion.
        """
        # Try PyTorch native SDPA (fuses operations, uses FlashAttention-like kernels)
        if hasattr(F, "scaled_dot_product_attention"):
            is_causal = attention_mask is None
            attn_mask = attention_mask if not is_causal else None

            return F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=attn_mask,
                dropout_p=self.attention_dropout if self.training else 0.0,
                is_causal=is_causal,
                scale=self.scaling,
            )

        # Manual attention fallback
        attn_weights = torch.matmul(query, key.transpose(2, 3)) * self.scaling

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)

        if self.training and self.attention_dropout > 0:
            attn_weights = F.dropout(attn_weights, p=self.attention_dropout)

        return torch.matmul(attn_weights, value)

    def _flash_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        """FlashAttention-2 for memory-efficient fused attention.

        Requires flash-attn package. 2-4x faster, O(N) memory instead of O(N^2).
        """
        # FlashAttention expects [B, S, H, D] layout
        q = query.transpose(1, 2)
        k = key.transpose(1, 2)
        v = value.transpose(1, 2)

        output = flash_attn_func(
            q, k, v,
            dropout_p=self.attention_dropout if self.training else 0.0,
            causal=True,
            softmax_scale=self.scaling,
        )

        return output.transpose(1, 2)
