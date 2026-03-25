"""Tests for the Yaya model architecture."""

import pytest
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import ModelConfig, VisionConfig
from src.model.yaya_model import YayaModel, YayaForCausalLM
from src.model.normalization import RMSNorm
from src.model.feedforward import SwiGLUFeedForward
from src.model.transformer import TransformerBlock


def get_test_config() -> ModelConfig:
    """Create a small model config for testing."""
    return ModelConfig(
        model_name="yaya-test",
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        rope_theta=10000.0,
        rms_norm_eps=1e-5,
        tie_word_embeddings=True,
        initializer_range=0.02,
    )


class TestRMSNorm:
    def test_output_shape(self):
        norm = RMSNorm(64)
        x = torch.randn(2, 16, 64)
        out = norm(x)
        assert out.shape == x.shape

    def test_normalization(self):
        norm = RMSNorm(64)
        x = torch.randn(2, 16, 64) * 100  # Large values
        out = norm(x)
        # Output should be roughly unit scale
        assert out.abs().mean() < 10


class TestSwiGLU:
    def test_output_shape(self):
        ffn = SwiGLUFeedForward(64, 128)
        x = torch.randn(2, 16, 64)
        out = ffn(x)
        assert out.shape == x.shape

    def test_no_bias(self):
        ffn = SwiGLUFeedForward(64, 128, bias=False)
        for name, param in ffn.named_parameters():
            assert "bias" not in name or param is None


class TestTransformerBlock:
    def test_output_shape(self):
        config = get_test_config()
        block = TransformerBlock(config, layer_idx=0)
        x = torch.randn(2, 16, 64)
        out, cache, aux = block(x)
        assert out.shape == x.shape
        assert cache is None
        assert aux is None  # Dense layer: no MoE aux loss

    def test_with_cache(self):
        config = get_test_config()
        block = TransformerBlock(config, layer_idx=0)
        x = torch.randn(2, 16, 64)
        out, cache, aux = block(x, use_cache=True)
        assert out.shape == x.shape
        assert cache is not None
        assert len(cache) == 2  # (key, value)
        assert aux is None  # Dense layer: no MoE aux loss


class TestYayaModel:
    def test_base_model_output_shape(self):
        config = get_test_config()
        model = YayaModel(config)
        input_ids = torch.randint(0, 256, (2, 16))
        hidden_states, _, moe_aux = model(input_ids)
        assert hidden_states.shape == (2, 16, 64)
        assert moe_aux is None  # Dense model: no MoE aux loss

    def test_causal_lm_output(self):
        config = get_test_config()
        model = YayaForCausalLM(config)
        input_ids = torch.randint(0, 256, (2, 16))
        outputs = model(input_ids=input_ids)
        assert "logits" in outputs
        assert outputs["logits"].shape == (2, 16, 256)

    def test_causal_lm_loss(self):
        config = get_test_config()
        model = YayaForCausalLM(config)
        input_ids = torch.randint(0, 256, (2, 16))
        labels = torch.randint(0, 256, (2, 16))
        outputs = model(input_ids=input_ids, labels=labels)
        assert "loss" in outputs
        assert outputs["loss"].dim() == 0  # Scalar
        assert outputs["loss"].item() > 0

    def test_kv_cache_generation(self):
        config = get_test_config()
        model = YayaForCausalLM(config)
        model.eval()

        # Prefill
        input_ids = torch.randint(0, 256, (1, 8))
        with torch.no_grad():
            outputs = model(input_ids=input_ids, use_cache=True)

        assert "past_key_values" in outputs
        past_kv = outputs["past_key_values"]
        assert len(past_kv) == config.num_hidden_layers

        # Decode one token
        next_token = torch.randint(0, 256, (1, 1))
        with torch.no_grad():
            outputs2 = model(
                input_ids=next_token,
                past_key_values=past_kv,
                use_cache=True,
            )
        assert outputs2["logits"].shape == (1, 1, 256)

    def test_parameter_count(self):
        config = get_test_config()
        model = YayaForCausalLM(config)
        n_params = model.get_num_params(non_embedding=False)
        assert n_params > 0
        summary = model.generate_summary()
        assert "yaya-test" in summary

    def test_gradient_checkpointing(self):
        config = get_test_config()
        model = YayaForCausalLM(config)
        model.model.enable_gradient_checkpointing()
        assert model.model.gradient_checkpointing is True

        input_ids = torch.randint(0, 256, (2, 16))
        labels = torch.randint(0, 256, (2, 16))
        outputs = model(input_ids=input_ids, labels=labels)
        outputs["loss"].backward()

        # Check gradients exist
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
