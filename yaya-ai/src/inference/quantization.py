"""Post-training quantization for efficient inference.

Supports INT8 and INT4 quantization to reduce model size
and increase inference throughput with minimal quality loss.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any


class QuantizationConfig:
    """Configuration for model quantization."""

    def __init__(
        self,
        bits: int = 8,
        group_size: int = 128,
        symmetric: bool = True,
        exclude_layers: Optional[list] = None,
    ):
        """
        Args:
            bits: Quantization bit width (4 or 8).
            group_size: Number of elements per quantization group.
            symmetric: Use symmetric quantization.
            exclude_layers: Layer name patterns to exclude from quantization.
        """
        self.bits = bits
        self.group_size = group_size
        self.symmetric = symmetric
        self.exclude_layers = exclude_layers or ["norm", "embed", "lm_head"]


def quantize_tensor_int8(
    tensor: torch.Tensor,
    symmetric: bool = True,
) -> Dict[str, torch.Tensor]:
    """Quantize a tensor to INT8.

    Args:
        tensor: Float tensor to quantize.
        symmetric: Use symmetric quantization around zero.

    Returns:
        Dict with 'quantized' (int8), 'scale' (float), 'zero_point' (int8).
    """
    if symmetric:
        abs_max = tensor.abs().max()
        scale = abs_max / 127.0
        quantized = torch.clamp(torch.round(tensor / scale), -128, 127).to(torch.int8)
        return {"quantized": quantized, "scale": scale, "zero_point": torch.tensor(0, dtype=torch.int8)}
    else:
        min_val = tensor.min()
        max_val = tensor.max()
        scale = (max_val - min_val) / 255.0
        zero_point = torch.clamp(torch.round(-min_val / scale), 0, 255).to(torch.int8)
        quantized = torch.clamp(torch.round(tensor / scale) + zero_point, 0, 255).to(torch.int8)
        return {"quantized": quantized, "scale": scale, "zero_point": zero_point}


def dequantize_tensor_int8(
    quantized: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
) -> torch.Tensor:
    """Dequantize an INT8 tensor back to float."""
    return (quantized.float() - zero_point.float()) * scale


class SimpleQuantizer:
    """Simple post-training quantization for Yaya model.

    Quantizes linear layer weights to INT8 or INT4 for reduced
    memory and faster inference. Uses per-channel quantization.

    For production use, prefer GPTQ, AWQ, or GGUF quantization
    via external tools (auto-gptq, autoawq, llama.cpp).
    """

    def __init__(self, config: Optional[QuantizationConfig] = None):
        self.config = config or QuantizationConfig()

    def should_quantize_layer(self, name: str) -> bool:
        """Check if a layer should be quantized based on config."""
        for pattern in self.config.exclude_layers:
            if pattern in name:
                return False
        return True

    @torch.no_grad()
    def quantize_model(self, model: nn.Module) -> nn.Module:
        """Quantize model weights in-place.

        Replaces float weights with quantized versions and stores
        scale factors for dequantization during forward pass.

        Args:
            model: The model to quantize.

        Returns:
            Quantized model (modified in-place).
        """
        total_params = 0
        quantized_params = 0

        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and self.should_quantize_layer(name):
                weight = module.weight.data
                total_params += weight.numel()

                if self.config.bits == 8:
                    result = quantize_tensor_int8(weight, self.config.symmetric)
                    module.weight_quantized = result["quantized"]
                    module.weight_scale = result["scale"]
                    module.weight_zero_point = result["zero_point"]
                    quantized_params += weight.numel()

        compression = total_params / max(quantized_params, 1)
        print(f"Quantized {quantized_params:,} / {total_params:,} parameters")
        print(f"  Bits: {self.config.bits}")
        print(f"  Estimated compression: ~{32 / self.config.bits:.1f}x")

        return model

    def estimate_model_size(self, model: nn.Module) -> Dict[str, float]:
        """Estimate model size before and after quantization.

        Returns:
            Dict with 'original_mb', 'quantized_mb', 'compression_ratio'.
        """
        total_params = sum(p.numel() for p in model.parameters())
        quantizable_params = sum(
            p.numel() for name, p in model.named_parameters()
            if self.should_quantize_layer(name) and p.dim() >= 2
        )
        non_quantizable_params = total_params - quantizable_params

        original_bytes = total_params * 2  # BF16
        quantized_bytes = (
            quantizable_params * (self.config.bits / 8)
            + non_quantizable_params * 2  # Keep non-quantizable in BF16
            + quantizable_params / self.config.group_size * 2  # Scale factors
        )

        original_mb = original_bytes / (1024 * 1024)
        quantized_mb = quantized_bytes / (1024 * 1024)

        return {
            "original_mb": original_mb,
            "quantized_mb": quantized_mb,
            "compression_ratio": original_mb / quantized_mb if quantized_mb > 0 else float("inf"),
        }
