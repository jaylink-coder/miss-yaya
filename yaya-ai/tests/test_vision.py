"""Tests for vision encoder and multimodal fusion."""

import pytest
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import ModelConfig, VisionConfig
from src.model.vision_encoder import (
    PatchEmbedding,
    VisionAttention,
    VisionMLP,
    VisionTransformerBlock,
    VisionEncoder,
)
from src.model.multimodal import VisionProjector, YayaMultimodalModel


def get_vision_config() -> VisionConfig:
    return VisionConfig(
        enabled=True,
        image_size=64,
        patch_size=16,
        vision_hidden_size=32,
        vision_layers=2,
        vision_heads=4,
    )


def get_multimodal_config() -> ModelConfig:
    return ModelConfig(
        model_name="yaya-mm-test",
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        vision=get_vision_config(),
    )


class TestPatchEmbedding:
    def test_output_shape(self):
        pe = PatchEmbedding(image_size=64, patch_size=16, hidden_size=32)
        x = torch.randn(2, 3, 64, 64)
        out = pe(x)
        # 64/16 = 4 patches per side, 4*4 = 16 patches
        assert out.shape == (2, 16, 32)

    def test_different_patch_sizes(self):
        pe = PatchEmbedding(image_size=64, patch_size=8, hidden_size=32)
        x = torch.randn(1, 3, 64, 64)
        out = pe(x)
        # 64/8 = 8, 8*8 = 64 patches
        assert out.shape == (1, 64, 32)

    def test_position_embeddings(self):
        pe = PatchEmbedding(image_size=64, patch_size=16, hidden_size=32)
        x = torch.randn(2, 3, 64, 64)
        out = pe(x)
        # All patch positions should have embeddings
        assert out.shape == (2, 16, 32)


class TestVisionAttention:
    def test_output_shape(self):
        attn = VisionAttention(hidden_size=32, num_heads=4)
        x = torch.randn(2, 16, 32)
        out = attn(x)
        assert out.shape == (2, 16, 32)

    def test_single_head(self):
        attn = VisionAttention(hidden_size=32, num_heads=1)
        x = torch.randn(1, 16, 32)
        out = attn(x)
        assert out.shape == (1, 16, 32)


class TestVisionMLP:
    def test_output_shape(self):
        mlp = VisionMLP(hidden_size=32, intermediate_size=64)
        x = torch.randn(2, 16, 32)
        out = mlp(x)
        assert out.shape == (2, 16, 32)


class TestVisionTransformerBlock:
    def test_output_shape(self):
        block = VisionTransformerBlock(hidden_size=32, num_heads=4)
        x = torch.randn(2, 16, 32)
        out = block(x)
        assert out.shape == (2, 16, 32)

    def test_residual_connection(self):
        block = VisionTransformerBlock(hidden_size=32, num_heads=4)
        x = torch.randn(2, 16, 32)
        out = block(x)
        # Output shouldn't be identical to input (transformations applied)
        assert not torch.allclose(x, out)


class TestVisionEncoder:
    def test_output_shape(self):
        config = get_vision_config()
        encoder = VisionEncoder(config)
        x = torch.randn(2, 3, 64, 64)
        out = encoder(x)
        assert out.shape[0] == 2
        assert out.shape[2] == 32  # vision_hidden_size

    def test_parameter_count(self):
        config = get_vision_config()
        encoder = VisionEncoder(config)
        n_params = sum(p.numel() for p in encoder.parameters())
        assert n_params > 0

    def test_gradient_flow(self):
        config = get_vision_config()
        encoder = VisionEncoder(config)
        x = torch.randn(1, 3, 64, 64, requires_grad=True)
        out = encoder(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None


class TestVisionProjector:
    def test_output_shape(self):
        proj = VisionProjector(vision_hidden_size=32, llm_hidden_size=64)
        x = torch.randn(2, 16, 32)
        out = proj(x)
        assert out.shape == (2, 16, 64)

    def test_projects_to_text_dim(self):
        proj = VisionProjector(vision_hidden_size=128, llm_hidden_size=256)
        x = torch.randn(1, 10, 128)
        out = proj(x)
        assert out.shape == (1, 10, 256)


class TestYayaMultimodalModel:
    def test_text_only_forward(self):
        config = get_multimodal_config()
        model = YayaMultimodalModel(config)
        input_ids = torch.randint(0, 256, (2, 16))
        outputs = model(input_ids=input_ids)
        assert "logits" in outputs
        assert outputs["logits"].shape == (2, 16, 256)

    def test_text_with_loss(self):
        config = get_multimodal_config()
        model = YayaMultimodalModel(config)
        input_ids = torch.randint(0, 256, (2, 16))
        labels = torch.randint(0, 256, (2, 16))
        outputs = model(input_ids=input_ids, labels=labels)
        assert "loss" in outputs
        assert outputs["loss"].dim() == 0

    def test_multimodal_forward(self):
        config = get_multimodal_config()
        model = YayaMultimodalModel(config)
        input_ids = torch.randint(0, 256, (2, 16))
        pixel_values = torch.randn(2, 3, 64, 64)
        outputs = model(input_ids=input_ids, pixel_values=pixel_values)
        assert "logits" in outputs

    def test_multimodal_gradient_flow(self):
        config = get_multimodal_config()
        model = YayaMultimodalModel(config)
        input_ids = torch.randint(0, 256, (2, 16))
        pixel_values = torch.randn(2, 3, 64, 64)
        # Don't pass labels — sequence length changes with image tokens
        outputs = model(input_ids=input_ids, pixel_values=pixel_values)
        # Compute loss manually from logits
        loss = outputs["logits"].sum()
        loss.backward()
        # Vision encoder should receive gradients
        vision_grad_found = False
        for name, param in model.named_parameters():
            if "vision" in name and param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                vision_grad_found = True
                break
        assert vision_grad_found, "No vision parameters found"

    def test_freeze_vision(self):
        config = get_multimodal_config()
        model = YayaMultimodalModel(config)
        model.freeze_vision_encoder()
        for name, param in model.vision_encoder.named_parameters():
            assert not param.requires_grad, f"{name} should be frozen"

    def test_unfreeze_vision(self):
        config = get_multimodal_config()
        model = YayaMultimodalModel(config)
        model.freeze_vision_encoder()
        model.unfreeze_vision_encoder()
        for name, param in model.vision_encoder.named_parameters():
            assert param.requires_grad, f"{name} should be unfrozen"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
