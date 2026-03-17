"""Transformer Block — the repeating unit of the Yaya model.

Each block applies pre-norm attention followed by pre-norm feed-forward,
with residual connections around both. This is the standard modern
transformer architecture used in LLaMA, Mistral, Qwen, etc.

Architecture per block:
    x -> RMSNorm -> GQA Attention -> + residual
                                     |
                                     v
                                  RMSNorm -> SwiGLU FFN -> + residual -> output
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

from src.model.attention import GroupedQueryAttention
from src.model.feedforward import SwiGLUFeedForward
from src.model.normalization import RMSNorm
from src.utils.config import ModelConfig


class TransformerBlock(nn.Module):
    """Single transformer decoder block with pre-normalization.

    Components:
        1. Pre-norm + Grouped-Query Attention + residual
        2. Pre-norm + SwiGLU Feed-Forward + residual
    """

    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx

        # Pre-attention normalization
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Grouped-Query Attention
        self.self_attn = GroupedQueryAttention(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            rope_theta=config.rope_theta,
            attention_dropout=config.attention_dropout,
            attention_bias=config.attention_bias,
        )

        # Pre-FFN normalization
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # SwiGLU Feed-Forward Network
        self.mlp = SwiGLUFeedForward(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            bias=config.mlp_bias,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Forward pass through one transformer block.

        Args:
            hidden_states: Input [batch, seq_len, hidden_size]
            attention_mask: Causal attention mask
            position_ids: Position indices for RoPE
            past_key_value: Cached KV for this layer
            use_cache: Whether to return updated KV cache

        Returns:
            Tuple of (output hidden_states, optional KV cache)
        """
        # 1. Pre-norm + Self-Attention + Residual
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        # 2. Pre-norm + FFN + Residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, present_key_value
