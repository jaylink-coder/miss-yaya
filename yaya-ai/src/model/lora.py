"""LoRA — Low-Rank Adaptation for parameter-efficient fine-tuning.

Injects trainable low-rank matrices (A, B) alongside frozen base weights.
For weight W ∈ R^{m×n}: output = x @ W.T + scaling * x @ lora_A.T @ lora_B.T
where scaling = alpha / r.  Only lora_A and lora_B are trained.

Reference: https://arxiv.org/abs/2106.09685
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class LoRAConfig:
    rank: int = 16
    alpha: float = 32.0
    dropout: float = 0.05
    # Module attribute names to wrap — must be plain nn.Linear leaves
    target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "down_proj"]
    )
    # Layer indices to adapt: "all" or list of ints, e.g. [0, 2, 4]
    layers_to_adapt: Union[str, List[int]] = "all"


# ---------------------------------------------------------------------------
# LoRALinear
# ---------------------------------------------------------------------------

class LoRALinear(nn.Module):
    """Drop-in replacement for nn.Linear with a low-rank adapter.

    The original weight is kept frozen.  Only lora_A and lora_B are trained.
    At export time call merge_weights() to fold the adapter back into W.
    """

    def __init__(
        self,
        linear: nn.Linear,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.rank = rank
        self.scaling = alpha / rank

        # Frozen base weight
        self.weight = linear.weight          # shared reference — do NOT copy
        self.bias = linear.bias              # may be None
        self.weight.requires_grad_(False)
        if self.bias is not None:
            self.bias.requires_grad_(False)

        # Trainable adapter — init: A~Kaiming, B=0  →  initial delta = 0
        self.lora_A = nn.Parameter(torch.empty(rank, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.linear(x, self.weight, self.bias)
        lora_delta = self.scaling * F.linear(F.linear(self.dropout(x), self.lora_A), self.lora_B)
        return base + lora_delta

    def merge_weights(self) -> nn.Linear:
        """Return a plain nn.Linear with adapter folded into W (for export)."""
        merged_weight = self.weight.data + self.scaling * (self.lora_B @ self.lora_A)
        new_linear = nn.Linear(self.in_features, self.out_features, bias=self.bias is not None)
        new_linear.weight = nn.Parameter(merged_weight)
        if self.bias is not None:
            new_linear.bias = nn.Parameter(self.bias.data.clone())
        return new_linear

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"rank={self.rank}, scaling={self.scaling:.4f}"
        )


# ---------------------------------------------------------------------------
# inject_lora / merge_lora / lora_state_dict
# ---------------------------------------------------------------------------

def inject_lora(model: nn.Module, config: LoRAConfig) -> nn.Module:
    """Walk the model and replace targeted nn.Linear leaves with LoRALinear.

    After injection:
    - Base weights are frozen (requires_grad=False).
    - Only lora_A / lora_B are trainable.
    - lm_head is always skipped to preserve tied-embedding integrity.
    """
    # Resolve which transformer layer indices to adapt
    adapt_all = config.layers_to_adapt == "all"
    if not adapt_all:
        layer_set = set(config.layers_to_adapt)

    def _in_adapted_layer(module_path: str) -> bool:
        if adapt_all:
            return True
        # Path looks like "model.layers.3.self_attn.q_proj"
        for part in module_path.split("."):
            if part.isdigit() and int(part) in layer_set:
                return True
        return False

    # Detect tied weights by data pointer — any Linear whose weight is shared
    # with another parameter (e.g., lm_head <-> embed_tokens) must be skipped
    # to preserve the tie. This is robust to naming convention changes.
    seen_ptrs: set[int] = set()
    tied_ptrs: set[int] = set()
    for param in model.parameters():
        ptr = param.data_ptr()
        if ptr in seen_ptrs:
            tied_ptrs.add(ptr)
        seen_ptrs.add(ptr)

    for name, module in list(model.named_modules()):
        # Check if the leaf attribute name is a target
        parent_name, _, attr_name = name.rpartition(".")
        if attr_name not in config.target_modules:
            continue
        if not isinstance(module, nn.Linear):
            continue
        # Skip any Linear whose weight is tied to another parameter
        if module.weight.data_ptr() in tied_ptrs:
            continue
        if not _in_adapted_layer(name):
            continue

        # Swap in LoRALinear
        parent = model
        if parent_name:
            for part in parent_name.split("."):
                parent = getattr(parent, part)
        lora_layer = LoRALinear(module, rank=config.rank, alpha=config.alpha, dropout=config.dropout)
        setattr(parent, attr_name, lora_layer)

    # Freeze everything that is NOT a LoRA adapter parameter
    for name, param in model.named_parameters():
        if "lora_A" not in name and "lora_B" not in name:
            param.requires_grad_(False)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(
        f"LoRA injected — trainable params: {trainable:,} / {total:,} "
        f"({100 * trainable / max(total, 1):.2f}%)"
    )
    return model


def merge_lora(model: nn.Module) -> nn.Module:
    """Replace every LoRALinear with its weight-merged nn.Linear.

    Call this before exporting a checkpoint to produce a standard model
    with no adapter wrappers and all base weights unfrozen.
    """
    for name, module in list(model.named_modules()):
        parent_name, _, attr_name = name.rpartition(".")
        if not isinstance(module, LoRALinear):
            continue
        parent = model
        if parent_name:
            for part in parent_name.split("."):
                parent = getattr(parent, part)
        setattr(parent, attr_name, module.merge_weights())
        print(f"  Merged LoRA: {name}")

    # Unfreeze all weights after merge
    for param in model.parameters():
        param.requires_grad_(True)
    return model


def lora_state_dict(model: nn.Module) -> dict:
    """Return only the lora_A / lora_B parameters for lightweight adapter saving."""
    return {
        k: v
        for k, v in model.state_dict().items()
        if "lora_A" in k or "lora_B" in k
    }
