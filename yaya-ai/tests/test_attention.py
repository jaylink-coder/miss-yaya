"""Tests for attention mechanisms."""

import pytest
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model.attention import GroupedQueryAttention
from src.model.embeddings import RotaryPositionalEmbedding, apply_rotary_pos_emb, YayaEmbeddings


class TestRoPE:
    def test_output_shape(self):
        rope = RotaryPositionalEmbedding(head_dim=32, max_position_embeddings=128)
        x = torch.randn(2, 4, 16, 32)  # [batch, heads, seq, head_dim]
        cos, sin = rope(x)
        assert cos.shape[-1] == 32
        assert sin.shape[-1] == 32

    def test_apply_rotary(self):
        rope = RotaryPositionalEmbedding(head_dim=32, max_position_embeddings=128)
        q = torch.randn(2, 4, 16, 32)
        k = torch.randn(2, 2, 16, 32)
        cos, sin = rope(q)
        q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)
        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape

    def test_cache_extension(self):
        rope = RotaryPositionalEmbedding(head_dim=32, max_position_embeddings=64)
        # Request longer sequence than initial cache
        x = torch.randn(1, 4, 128, 32)
        cos, sin = rope(x)
        assert cos.shape[-2] >= 128


class TestGQA:
    def test_output_shape(self):
        attn = GroupedQueryAttention(
            hidden_size=64,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            max_position_embeddings=128,
        )
        x = torch.randn(2, 16, 64)
        out, cache = attn(x)
        assert out.shape == (2, 16, 64)
        assert cache is None

    def test_gqa_with_cache(self):
        attn = GroupedQueryAttention(
            hidden_size=64,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            max_position_embeddings=128,
        )
        x = torch.randn(2, 16, 64)
        out, cache = attn(x, use_cache=True)
        assert cache is not None
        k, v = cache
        assert k.shape[1] == 2  # num_kv_heads
        assert k.shape[2] == 16  # seq_len

    def test_gqa_incremental_decoding(self):
        attn = GroupedQueryAttention(
            hidden_size=64,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            max_position_embeddings=128,
        )
        # Prefill
        x = torch.randn(1, 8, 64)
        _, cache = attn(x, use_cache=True)

        # Decode one token
        x_new = torch.randn(1, 1, 64)
        pos_ids = torch.tensor([[8]])
        out, new_cache = attn(x_new, past_key_value=cache, position_ids=pos_ids, use_cache=True)
        assert out.shape == (1, 1, 64)
        assert new_cache[0].shape[2] == 9  # 8 + 1

    def test_mha_mode(self):
        """When num_kv_heads == num_heads, GQA reduces to standard MHA."""
        attn = GroupedQueryAttention(
            hidden_size=64,
            num_attention_heads=4,
            num_key_value_heads=4,
            head_dim=16,
            max_position_embeddings=128,
        )
        x = torch.randn(2, 16, 64)
        out, _ = attn(x)
        assert out.shape == (2, 16, 64)

    def test_mqa_mode(self):
        """When num_kv_heads == 1, GQA reduces to Multi-Query Attention."""
        attn = GroupedQueryAttention(
            hidden_size=64,
            num_attention_heads=4,
            num_key_value_heads=1,
            head_dim=16,
            max_position_embeddings=128,
        )
        x = torch.randn(2, 16, 64)
        out, cache = attn(x, use_cache=True)
        assert out.shape == (2, 16, 64)
        assert cache[0].shape[1] == 1  # Single KV head


class TestEmbeddings:
    def test_token_embeddings(self):
        emb = YayaEmbeddings(vocab_size=256, hidden_size=64)
        ids = torch.randint(0, 256, (2, 16))
        out = emb(ids)
        assert out.shape == (2, 16, 64)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
