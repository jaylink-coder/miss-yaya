"""RMSNorm — Root Mean Square Layer Normalization.

Faster than LayerNorm (no mean subtraction), more stable training.
Used as pre-normalization in every transformer block.
Reference: https://arxiv.org/abs/1910.07467
"""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Normalizes inputs by their RMS value, then scales by a learned parameter.
    Formula: output = (x / RMS(x)) * weight
    where RMS(x) = sqrt(mean(x^2) + eps)
    """

    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight
