"""Model export utilities for deployment formats.

Supports exporting Yaya model weights to:
- SafeTensors (fast, safe loading)
- GGUF (llama.cpp compatible quantized format)
- ONNX (cross-platform inference)
"""

import os
import json
from typing import Optional, Dict, Any

import torch
import torch.nn as nn


def export_safetensors(
    model: nn.Module,
    output_dir: str,
    model_config: Optional[Dict[str, Any]] = None,
    shard_size_bytes: int = 5 * 1024 * 1024 * 1024,
):
    """Export model weights to SafeTensors format.

    Args:
        model: The model to export.
        output_dir: Output directory.
        model_config: Optional config dict to save alongside weights.
        shard_size_bytes: Max bytes per shard file (default 5GB).
    """
    try:
        from safetensors.torch import save_file
    except ImportError:
        raise ImportError("safetensors required: pip install safetensors")

    os.makedirs(output_dir, exist_ok=True)
    state_dict = model.state_dict()

    total_bytes = sum(t.numel() * t.element_size() for t in state_dict.values())
    num_shards = max(1, (total_bytes + shard_size_bytes - 1) // shard_size_bytes)

    if num_shards == 1:
        save_file(state_dict, os.path.join(output_dir, "model.safetensors"))
        print(f"Exported: model.safetensors ({total_bytes / 1e9:.2f} GB)")
    else:
        keys = list(state_dict.keys())
        shard_idx = 0
        current_shard = {}
        current_bytes = 0
        index = {"metadata": {"total_size": total_bytes}, "weight_map": {}}

        for key in keys:
            tensor = state_dict[key]
            tensor_bytes = tensor.numel() * tensor.element_size()

            if current_bytes + tensor_bytes > shard_size_bytes and current_shard:
                shard_name = f"model-{shard_idx:05d}-of-{num_shards:05d}.safetensors"
                save_file(current_shard, os.path.join(output_dir, shard_name))
                shard_idx += 1
                current_shard = {}
                current_bytes = 0

            current_shard[key] = tensor
            current_bytes += tensor_bytes
            shard_name = f"model-{shard_idx:05d}-of-{num_shards:05d}.safetensors"
            index["weight_map"][key] = shard_name

        if current_shard:
            shard_name = f"model-{shard_idx:05d}-of-{num_shards:05d}.safetensors"
            save_file(current_shard, os.path.join(output_dir, shard_name))

        with open(os.path.join(output_dir, "model.safetensors.index.json"), "w") as f:
            json.dump(index, f, indent=2)

    if model_config:
        with open(os.path.join(output_dir, "config.json"), "w") as f:
            json.dump(model_config, f, indent=2)

    print(f"SafeTensors export complete: {output_dir}")


def export_onnx(
    model: nn.Module,
    output_path: str,
    vocab_size: int,
    max_seq_length: int = 2048,
    opset_version: int = 17,
):
    """Export model to ONNX format for cross-platform inference.

    Args:
        model: The model to export (should be in eval mode).
        output_path: Path for the .onnx file.
        vocab_size: Vocabulary size for dummy input.
        max_seq_length: Max sequence length for dynamic axes.
        opset_version: ONNX opset version.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    model.eval()
    device = next(model.parameters()).device

    dummy_input_ids = torch.randint(0, vocab_size, (1, 64), device=device)

    dynamic_axes = {
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "logits": {0: "batch_size", 1: "sequence_length"},
    }

    print(f"Exporting to ONNX: {output_path}")
    torch.onnx.export(
        model,
        (dummy_input_ids,),
        output_path,
        input_names=["input_ids"],
        output_names=["logits"],
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
        do_constant_folding=True,
    )
    print(f"ONNX export complete: {output_path}")
    print(f"  File size: {os.path.getsize(output_path) / 1e6:.1f} MB")


def prepare_gguf_metadata(
    model_config: Dict[str, Any],
    tokenizer_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Prepare metadata dict for GGUF export.

    GGUF is the format used by llama.cpp for efficient CPU/GPU inference.
    Actual GGUF conversion requires the llama.cpp convert script.

    Args:
        model_config: Model configuration dict.
        tokenizer_path: Path to tokenizer model file.

    Returns:
        Dict with GGUF-compatible metadata.
    """
    metadata = {
        "general.architecture": "llama",
        "general.name": model_config.get("model_name", "yaya"),
        "general.file_type": 1,
        "llama.context_length": model_config.get("max_position_embeddings", 4096),
        "llama.embedding_length": model_config.get("hidden_size", 2048),
        "llama.block_count": model_config.get("num_hidden_layers", 24),
        "llama.feed_forward_length": model_config.get("intermediate_size", 5632),
        "llama.attention.head_count": model_config.get("num_attention_heads", 16),
        "llama.attention.head_count_kv": model_config.get("num_key_value_heads", 4),
        "llama.rope.freq_base": model_config.get("rope_theta", 10000.0),
        "llama.attention.layer_norm_rms_epsilon": model_config.get("rms_norm_eps", 1e-5),
        "tokenizer.ggml.model": "llama",
    }
    if tokenizer_path:
        metadata["tokenizer.ggml.tokens_path"] = tokenizer_path
    return metadata


def export_for_llama_cpp(
    model: nn.Module,
    output_dir: str,
    model_config: Dict[str, Any],
    tokenizer_path: Optional[str] = None,
):
    """Prepare model for llama.cpp GGUF conversion.

    Saves weights and metadata. After running this, use llama.cpp's convert tool:
        python convert_hf_to_gguf.py <output_dir> --outtype f16

    Args:
        model: The model to export.
        output_dir: Output directory.
        model_config: Model configuration dict.
        tokenizer_path: Path to tokenizer file.
    """
    os.makedirs(output_dir, exist_ok=True)

    state_dict = model.state_dict()
    torch.save(state_dict, os.path.join(output_dir, "pytorch_model.bin"))

    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(model_config, f, indent=2)

    metadata = prepare_gguf_metadata(model_config, tokenizer_path)
    with open(os.path.join(output_dir, "gguf_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    total_mb = sum(t.numel() * t.element_size() for t in state_dict.values()) / (1024 * 1024)
    print(f"Exported for llama.cpp: {output_dir} ({total_mb:.1f} MB)")
    print("  Next: use llama.cpp convert_hf_to_gguf.py to create GGUF file")
