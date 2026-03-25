"""Mixture-of-Experts (MoE) Feed-Forward Network.

Replaces the dense SwiGLU FFN in each selected TransformerBlock with a sparse
MoE layer: num_experts independent FFNs, with a learned router that selects
the top-K experts for each token.

Total parameters scale with num_experts; FLOPs per token scale with K (sparse
activation).  A load-balancing auxiliary loss prevents all tokens from routing
to the same expert (expert collapse).

Reference: Switch Transformer — https://arxiv.org/abs/2101.03961
           Mixtral — https://arxiv.org/abs/2401.04088
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class MoEConfig:
    enabled: bool = False
    num_experts: int = 8
    top_k: int = 2
    # Which transformer layer indices use MoE: "all", "alternate", or "0,2,4,..."
    moe_layers: str = "alternate"
    # Load-balance auxiliary loss weight (Switch Transformer recommends ~0.01)
    load_balance_loss_coeff: float = 0.01
    # Random jitter added to router logits during training to improve load balance
    router_jitter_noise: float = 0.01

    def is_moe_layer(self, layer_idx: int) -> bool:
        if not self.enabled:
            return False
        if self.moe_layers == "all":
            return True
        if self.moe_layers == "alternate":
            return layer_idx % 2 == 1
        try:
            indices = {int(x.strip()) for x in self.moe_layers.split(",")}
            return layer_idx in indices
        except ValueError:
            return False


# ---------------------------------------------------------------------------
# ExpertFFN
# ---------------------------------------------------------------------------

class ExpertFFN(nn.Module):
    """Single expert — identical architecture to SwiGLUFeedForward."""

    def __init__(self, hidden_size: int, intermediate_size: int, bias: bool = False):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ---------------------------------------------------------------------------
# MoERouter
# ---------------------------------------------------------------------------

class MoERouter(nn.Module):
    """Learned top-K router over experts.

    For each token, computes a softmax distribution over experts and selects
    the top-K.  Returns dispatch weights and a load-balance auxiliary loss.
    """

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        top_k: int,
        jitter_noise: float = 0.01,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.jitter_noise = jitter_noise
        # Lightweight linear gate — one scalar per expert per token
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)

    def forward(
        self, hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Route tokens to top-K experts.

        Args:
            hidden_states: [batch * seq_len, hidden_size]

        Returns:
            router_weights:      [batch*seq, top_k]  — softmax weights for chosen experts
            selected_experts:    [batch*seq, top_k]  — expert indices
            load_balance_loss:   scalar tensor
        """
        # Router logits
        logits = self.gate(hidden_states)  # [T, num_experts]

        # Optional jitter during training for better load balance
        if self.training and self.jitter_noise > 0.0:
            noise = torch.empty_like(logits).uniform_(
                1.0 - self.jitter_noise, 1.0 + self.jitter_noise
            )
            logits = logits * noise

        # Router probabilities
        router_probs = F.softmax(logits, dim=-1, dtype=torch.float32)  # [T, E]

        # Select top-K experts per token
        router_weights, selected_experts = torch.topk(router_probs, self.top_k, dim=-1)
        # Renormalise top-K weights so they sum to 1
        # Clamp denominator to avoid divide-by-zero from numerical instability
        router_weights = router_weights / router_weights.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        router_weights = router_weights.to(hidden_states.dtype)

        # Load-balance auxiliary loss (Switch Transformer Eq. 4-5)
        # Encourages uniform routing: penalises the correlation between
        # the fraction of tokens dispatched to each expert and the mean
        # router probability for each expert.
        tokens_per_expert = torch.zeros(
            self.num_experts, dtype=torch.float32, device=hidden_states.device
        )
        # fraction of tokens routed to each expert (in top-1 for aux loss)
        top1_experts = selected_experts[:, 0]  # [T]
        tokens_per_expert.scatter_add_(
            0, top1_experts, torch.ones_like(top1_experts, dtype=torch.float32)
        )
        fraction_routed = tokens_per_expert / hidden_states.shape[0]          # [E]
        mean_router_prob = router_probs.mean(dim=0)                             # [E]
        load_balance_loss = self.num_experts * (fraction_routed * mean_router_prob).sum()

        return router_weights, selected_experts, load_balance_loss


# ---------------------------------------------------------------------------
# MoEFeedForward
# ---------------------------------------------------------------------------

class MoEFeedForward(nn.Module):
    """Sparse MoE FFN — drop-in replacement for SwiGLUFeedForward.

    Returns (output, load_balance_loss) instead of just output.
    TransformerBlock handles unpacking.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int = 8,
        top_k: int = 2,
        router_jitter_noise: float = 0.01,
        bias: bool = False,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_size = hidden_size

        self.router = MoERouter(hidden_size, num_experts, top_k, router_jitter_noise)
        self.experts = nn.ModuleList(
            [ExpertFFN(hidden_size, intermediate_size, bias=bias) for _ in range(num_experts)]
        )

    def forward(
        self, hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sparse forward pass.

        Args:
            hidden_states: [batch, seq_len, hidden_size]

        Returns:
            output:             [batch, seq_len, hidden_size]
            load_balance_loss:  scalar tensor
        """
        batch, seq_len, hidden_size = hidden_states.shape
        # Flatten to [T, H]
        flat = hidden_states.view(-1, hidden_size)  # [T, H]

        router_weights, selected_experts, aux_loss = self.router(flat)
        # router_weights:   [T, top_k]
        # selected_experts: [T, top_k]

        # Compute expert outputs for each (token, expert-slot) pair
        output = torch.zeros_like(flat)

        for k in range(self.top_k):
            expert_indices = selected_experts[:, k]   # [T]
            expert_weights = router_weights[:, k]     # [T]

            for expert_id in range(self.num_experts):
                # Tokens routed to this expert in slot k
                mask = expert_indices == expert_id    # [T] bool
                if not mask.any():
                    continue
                token_inputs = flat[mask]             # [n_tokens, H]
                expert_out = self.experts[expert_id](token_inputs)  # [n_tokens, H]
                # Weight by router probability and accumulate
                output[mask] += expert_weights[mask].unsqueeze(-1) * expert_out

        output = output.view(batch, seq_len, hidden_size)
        return output, aux_loss


# ---------------------------------------------------------------------------
# convert_to_moe  — dense → sparse upgrade path
# ---------------------------------------------------------------------------

def convert_to_moe(model: nn.Module, config: MoEConfig) -> nn.Module:
    """Replace SwiGLUFeedForward with MoEFeedForward in selected layers.

    Dense-to-sparse upgrade: copies the original FFN weights into expert 0;
    remaining experts are randomly initialised.  This preserves existing
    knowledge in at least one expert from day one.
    """
    from src.model.feedforward import SwiGLUFeedForward

    for name, module in model.named_modules():
        # Only wrap TransformerBlock.mlp if it's a plain SwiGLUFeedForward
        if not isinstance(module, SwiGLUFeedForward):
            continue

        # Infer layer index from module name (e.g. "model.layers.5.mlp")
        parts = name.split(".")
        layer_idx = None
        for p in parts:
            if p.isdigit():
                layer_idx = int(p)
                break

        if layer_idx is None or not config.is_moe_layer(layer_idx):
            continue

        # Build MoE layer with same dimensions
        hidden_size = module.gate_proj.in_features
        intermediate_size = module.gate_proj.out_features
        bias = module.gate_proj.bias is not None

        moe_layer = MoEFeedForward(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=config.num_experts,
            top_k=config.top_k,
            router_jitter_noise=config.router_jitter_noise,
            bias=bias,
        )

        # Copy dense weights (and biases if present) into expert 0
        with torch.no_grad():
            moe_layer.experts[0].gate_proj.weight.copy_(module.gate_proj.weight)
            moe_layer.experts[0].up_proj.weight.copy_(module.up_proj.weight)
            moe_layer.experts[0].down_proj.weight.copy_(module.down_proj.weight)
            if module.gate_proj.bias is not None:
                moe_layer.experts[0].gate_proj.bias.copy_(module.gate_proj.bias)
            if module.up_proj.bias is not None:
                moe_layer.experts[0].up_proj.bias.copy_(module.up_proj.bias)
            if module.down_proj.bias is not None:
                moe_layer.experts[0].down_proj.bias.copy_(module.down_proj.bias)

        # Replace the module in its parent
        parent_name, _, attr = name.rpartition(".")
        parent = model
        for part in parent_name.split("."):
            parent = getattr(parent, part)
        setattr(parent, attr, moe_layer)
        print(f"  Converted layer {layer_idx} FFN → MoE ({config.num_experts} experts, top-{config.top_k})")

    return model
