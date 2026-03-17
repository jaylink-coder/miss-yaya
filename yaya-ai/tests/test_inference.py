"""Tests for inference components."""

import pytest
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import ModelConfig
from src.model.yaya_model import YayaForCausalLM
from src.inference.kv_cache import KVCache
from src.inference.quantization import (
    quantize_tensor_int8,
    dequantize_tensor_int8,
    QuantizationConfig,
    SimpleQuantizer,
)
from src.evaluation.metrics import accuracy, f1_score, exact_match, perplexity


def get_test_config():
    return ModelConfig(
        model_name="yaya-test",
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
    )


class TestKVCache:
    def test_initialization(self):
        cache = KVCache(
            num_layers=2,
            max_batch_size=1,
            max_seq_length=64,
            num_kv_heads=2,
            head_dim=16,
            device="cpu",
        )
        assert cache.seq_length == 0
        assert len(cache.key_cache) == 2

    def test_update_and_get(self):
        cache = KVCache(
            num_layers=2,
            max_batch_size=1,
            max_seq_length=64,
            num_kv_heads=2,
            head_dim=16,
            device="cpu",
        )
        new_k = torch.randn(1, 2, 8, 16)
        new_v = torch.randn(1, 2, 8, 16)
        full_k, full_v = cache.update(0, new_k, new_v)
        assert full_k.shape == (1, 2, 8, 16)
        cache.advance(8)
        assert cache.seq_length == 8

    def test_incremental_update(self):
        cache = KVCache(
            num_layers=1,
            max_batch_size=1,
            max_seq_length=64,
            num_kv_heads=2,
            head_dim=16,
            device="cpu",
        )
        # First chunk
        k1 = torch.randn(1, 2, 4, 16)
        v1 = torch.randn(1, 2, 4, 16)
        cache.update(0, k1, v1)
        cache.advance(4)

        # Second chunk
        k2 = torch.randn(1, 2, 1, 16)
        v2 = torch.randn(1, 2, 1, 16)
        full_k, full_v = cache.update(0, k2, v2)
        assert full_k.shape == (1, 2, 5, 16)

    def test_reset(self):
        cache = KVCache(
            num_layers=1,
            max_batch_size=1,
            max_seq_length=64,
            num_kv_heads=2,
            head_dim=16,
            device="cpu",
        )
        cache.advance(10)
        cache.reset()
        assert cache.seq_length == 0

    def test_memory_estimate(self):
        cache = KVCache(
            num_layers=32,
            max_batch_size=1,
            max_seq_length=4096,
            num_kv_heads=8,
            head_dim=128,
            dtype=torch.bfloat16,
            device="cpu",
        )
        mb = cache.memory_usage_mb
        assert mb > 0


class TestQuantization:
    def test_int8_symmetric(self):
        tensor = torch.randn(64, 128)
        result = quantize_tensor_int8(tensor, symmetric=True)
        assert result["quantized"].dtype == torch.int8
        assert result["scale"].item() > 0

    def test_int8_roundtrip(self):
        tensor = torch.randn(64, 128)
        result = quantize_tensor_int8(tensor, symmetric=True)
        reconstructed = dequantize_tensor_int8(
            result["quantized"], result["scale"], result["zero_point"]
        )
        # Quantization error should be small
        error = (tensor - reconstructed).abs().mean()
        assert error < 0.1

    def test_quantizer_size_estimate(self):
        config = get_test_config()
        model = YayaForCausalLM(config)
        quantizer = SimpleQuantizer(QuantizationConfig(bits=8))
        sizes = quantizer.estimate_model_size(model)
        assert sizes["original_mb"] > 0
        assert sizes["compression_ratio"] >= 1.0


class TestMetrics:
    def test_accuracy(self):
        assert accuracy([1, 2, 3], [1, 2, 3]) == 1.0
        assert accuracy([1, 2, 3], [1, 2, 4]) == pytest.approx(2 / 3)

    def test_f1_score(self):
        assert f1_score("the cat sat", "the cat sat") == 1.0
        assert f1_score("hello", "world") == 0.0

    def test_exact_match(self):
        assert exact_match("Hello", "hello") == 1.0
        assert exact_match("Hello", "hello", normalize=False) == 0.0

    def test_perplexity(self):
        assert perplexity(0.0) == pytest.approx(1.0)
        assert perplexity(1.0) == pytest.approx(2.718, rel=0.01)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
