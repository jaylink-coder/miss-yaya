"""KV-Cache for efficient autoregressive generation.

During inference, caches key and value tensors from previous tokens
so each new token only requires computing attention for one position
instead of the full sequence.
"""

import torch
from typing import List, Optional, Tuple


class KVCache:
    """Key-Value cache for efficient transformer inference.

    Stores computed key and value tensors for all layers,
    enabling O(1) per-token computation during generation
    instead of O(n) recomputation.
    """

    def __init__(
        self,
        num_layers: int,
        max_batch_size: int = 1,
        max_seq_length: int = 4096,
        num_kv_heads: int = 4,
        head_dim: int = 128,
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
    ):
        """Initialize empty KV cache.

        Args:
            num_layers: Number of transformer layers.
            max_batch_size: Maximum batch size.
            max_seq_length: Maximum sequence length to cache.
            num_kv_heads: Number of key-value heads (GQA).
            head_dim: Dimension per attention head.
            dtype: Data type for cache tensors.
            device: Device to store cache on.
        """
        self.num_layers = num_layers
        self.max_batch_size = max_batch_size
        self.max_seq_length = max_seq_length
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = device

        # Pre-allocate cache tensors for each layer
        # Shape: [batch, num_kv_heads, max_seq_length, head_dim]
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []

        for _ in range(num_layers):
            self.key_cache.append(
                torch.zeros(
                    max_batch_size, num_kv_heads, max_seq_length, head_dim,
                    dtype=dtype, device=device,
                )
            )
            self.value_cache.append(
                torch.zeros(
                    max_batch_size, num_kv_heads, max_seq_length, head_dim,
                    dtype=dtype, device=device,
                )
            )

        # Current sequence length in cache
        self.seq_length = 0

    def update(
        self,
        layer_idx: int,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update cache for a specific layer and return full cached KV.

        Args:
            layer_idx: Which transformer layer.
            key: New key tensor [batch, num_kv_heads, new_seq_len, head_dim]
            value: New value tensor [batch, num_kv_heads, new_seq_len, head_dim]

        Returns:
            Tuple of full (cached_key, cached_value) including new tokens.
        """
        new_seq_len = key.shape[2]

        # Write new KV into cache
        start = self.seq_length
        end = self.seq_length + new_seq_len
        self.key_cache[layer_idx][:, :, start:end, :] = key
        self.value_cache[layer_idx][:, :, start:end, :] = value

        # Return full cached KV up to current position
        return (
            self.key_cache[layer_idx][:, :, :end, :],
            self.value_cache[layer_idx][:, :, :end, :],
        )

    def advance(self, num_tokens: int = 1):
        """Advance the sequence position counter after processing tokens."""
        self.seq_length += num_tokens

    def get(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get current cached KV for a layer.

        Returns:
            Tuple of (key_cache, value_cache) up to current seq_length.
        """
        return (
            self.key_cache[layer_idx][:, :, :self.seq_length, :],
            self.value_cache[layer_idx][:, :, :self.seq_length, :],
        )

    def get_as_list(self) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Get all layer caches as a list of (K, V) tuples.

        Compatible with model forward pass past_key_values argument.
        """
        return [self.get(i) for i in range(self.num_layers)]

    def reset(self):
        """Clear the cache for a new generation."""
        self.seq_length = 0
        for i in range(self.num_layers):
            self.key_cache[i].zero_()
            self.value_cache[i].zero_()

    @property
    def memory_usage_mb(self) -> float:
        """Estimate memory usage of the cache in MB."""
        bytes_per_element = 2 if self.dtype in (torch.bfloat16, torch.float16) else 4
        total_elements = (
            2 * self.num_layers * self.max_batch_size
            * self.num_kv_heads * self.max_seq_length * self.head_dim
        )
        return total_elements * bytes_per_element / (1024 * 1024)
