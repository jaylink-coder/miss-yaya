"""Yaya Model — Top-level model classes.

YayaModel: Base transformer model (embeddings + N transformer blocks + final norm)
YayaForCausalLM: Adds language modeling head on top for next-token prediction

This is the main entry point for the model architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict, Any
import math

from src.model.transformer import TransformerBlock
from src.model.embeddings import YayaEmbeddings
from src.model.normalization import RMSNorm
from src.utils.config import ModelConfig


class YayaModel(nn.Module):
    """Base Yaya transformer model.

    Architecture:
        Token IDs -> Embedding -> [TransformerBlock x N] -> RMSNorm -> hidden_states

    This model outputs raw hidden states. Use YayaForCausalLM for language modeling.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Token embeddings (no positional embedding — RoPE is in attention)
        self.embed_tokens = YayaEmbeddings(config.vocab_size, config.hidden_size)

        # Transformer blocks
        self.layers = nn.ModuleList(
            [TransformerBlock(config, layer_idx=i) for i in range(config.num_hidden_layers)]
        )

        # Final layer normalization
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Gradient checkpointing flag
        self.gradient_checkpointing = False

    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing to trade compute for memory."""
        self.gradient_checkpointing = True

    def disable_gradient_checkpointing(self):
        """Disable gradient checkpointing."""
        self.gradient_checkpointing = False

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[List[Tuple[torch.Tensor, torch.Tensor]]], Optional[torch.Tensor]]:
        """Forward pass through the base model.

        Args:
            input_ids: Token indices [batch, seq_len]
            attention_mask: Padding mask [batch, seq_len] (1 = attend, 0 = mask)
            position_ids: Position indices [batch, seq_len]
            past_key_values: List of cached (K, V) per layer for incremental decoding
            use_cache: Whether to return KV caches
            inputs_embeds: Pre-computed embeddings (used for multimodal input)

        Returns:
            Tuple of (hidden_states, optional KV caches, optional MoE aux loss)
        """
        # Get embeddings
        if inputs_embeds is None:
            hidden_states = self.embed_tokens(input_ids)
        else:
            hidden_states = inputs_embeds

        batch_size, seq_len = hidden_states.shape[:2]

        # Build position IDs if not provided
        if position_ids is None:
            past_len = past_key_values[0][0].shape[2] if past_key_values is not None else 0
            position_ids = torch.arange(
                past_len, past_len + seq_len, dtype=torch.long, device=hidden_states.device
            ).unsqueeze(0)

        # Build causal attention mask
        causal_mask = self._build_causal_mask(
            batch_size, seq_len, hidden_states.device, hidden_states.dtype,
            past_key_values_length=past_key_values[0][0].shape[2] if past_key_values else 0,
            attention_mask=attention_mask,
        )

        # Pass through transformer layers
        all_present_key_values = [] if use_cache else None
        total_moe_aux_loss: Optional[torch.Tensor] = None

        for idx, layer in enumerate(self.layers):
            past_kv = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                hidden_states, present_kv, aux_loss = self._gradient_checkpointing_forward(
                    layer, hidden_states, causal_mask, position_ids, past_kv, use_cache
                )
            else:
                hidden_states, present_kv, aux_loss = layer(
                    hidden_states=hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_kv,
                    use_cache=use_cache,
                )

            if aux_loss is not None:
                total_moe_aux_loss = (
                    aux_loss if total_moe_aux_loss is None else total_moe_aux_loss + aux_loss
                )

            if use_cache:
                all_present_key_values.append(present_kv)

        # Final normalization
        hidden_states = self.norm(hidden_states)

        return hidden_states, all_present_key_values, total_moe_aux_loss

    def _gradient_checkpointing_forward(self, layer, *args):
        """Wrap layer forward in gradient checkpointing."""
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)
            return custom_forward

        return torch.utils.checkpoint.checkpoint(
            create_custom_forward(layer),
            *args,
            use_reentrant=False,
        )

    def _build_causal_mask(
        self,
        batch_size: int,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
        past_key_values_length: int = 0,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """Build combined causal + padding attention mask.

        Returns:
            Mask tensor [batch, 1, seq_len, total_len] where masked positions are -inf.
            Returns None if no masking is needed (for SDPA/FlashAttention causal mode).
        """
        total_len = seq_len + past_key_values_length

        # If no padding mask and using SDPA, return None (let SDPA handle causal)
        if attention_mask is None and past_key_values_length == 0:
            return None

        # Build causal mask: upper triangular = -inf
        causal_mask = torch.full(
            (seq_len, total_len), fill_value=float("-inf"), device=device, dtype=dtype
        )
        # Allow attending to current and all past positions
        causal_mask = torch.triu(causal_mask, diagonal=past_key_values_length + 1)
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, S, T]
        causal_mask = causal_mask.expand(batch_size, 1, seq_len, total_len)

        # Combine with padding mask if provided
        if attention_mask is not None:
            # attention_mask: [B, total_len] with 1=attend, 0=pad
            # Use masked_fill to avoid 0.0 * (-inf) = NaN in IEEE 754
            pad_positions = attention_mask[:, None, None, :] == 0  # [B, 1, 1, T]
            causal_mask = causal_mask.masked_fill(pad_positions, float("-inf"))

        return causal_mask


class YayaForCausalLM(nn.Module):
    """Yaya model with causal language modeling head.

    Adds a linear projection from hidden states to vocabulary logits
    for next-token prediction training and generation.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.model = YayaModel(config)

        # Language modeling head
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Optionally tie input/output embeddings
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.word_embeddings.weight

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        """Initialize weights following standard transformer practice."""
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Forward pass with optional loss computation.

        Args:
            input_ids: Token indices [batch, seq_len]
            attention_mask: Padding mask [batch, seq_len]
            position_ids: Position indices [batch, seq_len]
            past_key_values: Cached KV pairs per layer
            labels: Target token IDs for loss [batch, seq_len]
            use_cache: Return KV caches for generation
            inputs_embeds: Pre-computed embeddings (for multimodal)

        Returns:
            Dict with keys: 'loss' (if labels), 'logits', 'past_key_values' (if use_cache)
        """
        # Get hidden states from base model
        hidden_states, present_key_values = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            inputs_embeds=inputs_embeds,
        )

        # Project to vocabulary
        logits = self.lm_head(hidden_states)

        # Compute loss if labels provided
        # Labels are already shifted by 1 in the dataset (labels[i] = input_ids[i+1]),
        # so we compute loss directly between logits and labels without any extra shift.
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                labels.view(-1),
                ignore_index=-100,  # Ignore padding tokens in loss
            )

        output = {"logits": logits}
        if loss is not None:
            output["loss"] = loss
        if use_cache:
            output["past_key_values"] = present_key_values

        return output

    def get_num_params(self, non_embedding: bool = True) -> int:
        """Return total number of parameters.

        Args:
            non_embedding: If True, exclude token embedding parameters.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.model.embed_tokens.word_embeddings.weight.numel()
            if not self.config.tie_word_embeddings:
                n_params -= self.lm_head.weight.numel()
        return n_params

    def estimate_flops_per_token(self) -> int:
        """Estimate FLOPs per token for training (forward + backward ~= 6N).

        Based on the Chinchilla scaling law approximation:
        FLOPs per token ~= 6 * N (for forward + backward pass)
        """
        n_params = self.get_num_params(non_embedding=True)
        return 6 * n_params

    @torch.no_grad()
    def generate_summary(self) -> str:
        """Print a human-readable model summary."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        non_emb_params = self.get_num_params(non_embedding=True)

        lines = [
            f"{'=' * 60}",
            f"  Yaya Model Summary: {self.config.model_name}",
            f"{'=' * 60}",
            f"  Layers:              {self.config.num_hidden_layers}",
            f"  Hidden size:         {self.config.hidden_size}",
            f"  FFN size:            {self.config.intermediate_size}",
            f"  Attention heads:     {self.config.num_attention_heads}",
            f"  KV heads (GQA):      {self.config.num_key_value_heads}",
            f"  Head dim:            {self.config.head_dim}",
            f"  Vocab size:          {self.config.vocab_size}",
            f"  Max seq length:      {self.config.max_position_embeddings}",
            f"  RoPE theta:          {self.config.rope_theta}",
            f"  Tied embeddings:     {self.config.tie_word_embeddings}",
            f"{'─' * 60}",
            f"  Total params:        {total_params:,} ({total_params / 1e9:.2f}B)",
            f"  Non-embedding:       {non_emb_params:,} ({non_emb_params / 1e9:.2f}B)",
            f"  Trainable params:    {trainable_params:,}",
            f"  Est. FLOPs/token:    {self.estimate_flops_per_token():,}",
            f"{'=' * 60}",
        ]
        return "\n".join(lines)
