"""SwiGLU Feed-Forward Network.

SwiGLU is a gated linear unit with SiLU (Swish) activation, shown to
outperform ReLU and GELU in transformer language models.
Reference: https://arxiv.org/abs/2002.05202
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLUFeedForward(nn.Module):
    """SwiGLU-based feed-forward network.

    Architecture:
        Input [B, S, H]
        -> Gate projection: Linear(H -> intermediate_size)
        -> Up projection:   Linear(H -> intermediate_size)
        -> SwiGLU:          SiLU(gate) * up
        -> Down projection: Linear(intermediate_size -> H)
        -> Output [B, S, H]

    The gate and up projections can be fused into a single linear layer
    for efficiency, but are kept separate here for clarity.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        bias: bool = False,
    ):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor [batch, seq_len, hidden_size]

        Returns:
            Output tensor [batch, seq_len, hidden_size]
        """
        # SwiGLU: SiLU(gate(x)) * up(x), then project back down
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
