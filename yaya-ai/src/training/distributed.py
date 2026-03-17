"""Distributed training setup and utilities.

Supports:
- PyTorch DDP (DistributedDataParallel)
- PyTorch FSDP (Fully Sharded Data Parallel)
- DeepSpeed ZeRO (Stage 1, 2, 3)
"""

import os
import torch
import torch.distributed as dist
from typing import Optional, Dict, Any

from src.utils.config import TrainingConfig


def setup_distributed() -> Dict[str, Any]:
    """Initialize distributed training environment.

    Auto-detects backend and configuration from environment variables
    set by torchrun or DeepSpeed launcher.

    Returns:
        Dict with 'rank', 'local_rank', 'world_size', 'device'.
    """
    # Check if distributed environment is set up
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
    elif "SLURM_PROCID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        local_rank = int(os.environ.get("SLURM_LOCALID", 0))
        world_size = int(os.environ.get("SLURM_NTASKS", 1))
    else:
        # Single process
        rank = 0
        local_rank = 0
        world_size = 1

    # Set device
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    # Initialize process group
    if world_size > 1 and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(
            backend=backend,
            rank=rank,
            world_size=world_size,
        )
        print(f"Distributed initialized: rank={rank}, world_size={world_size}, backend={backend}")

    return {
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "device": device,
    }


def cleanup_distributed():
    """Clean up distributed training resources."""
    if dist.is_initialized():
        dist.destroy_process_group()


def wrap_model_ddp(
    model: torch.nn.Module,
    device_id: int,
    find_unused_parameters: bool = False,
) -> torch.nn.Module:
    """Wrap model with DistributedDataParallel."""
    from torch.nn.parallel import DistributedDataParallel as DDP

    return DDP(
        model,
        device_ids=[device_id],
        output_device=device_id,
        find_unused_parameters=find_unused_parameters,
    )


def wrap_model_fsdp(
    model: torch.nn.Module,
    mixed_precision: bool = True,
) -> torch.nn.Module:
    """Wrap model with Fully Sharded Data Parallel (FSDP).

    FSDP shards model parameters, gradients, and optimizer states
    across all GPUs for memory efficiency.
    """
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import MixedPrecision, ShardingStrategy

    mp_policy = None
    if mixed_precision:
        mp_policy = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.bfloat16,
        )

    return FSDP(
        model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=mp_policy,
        use_orig_params=True,
    )


def get_deepspeed_config(
    config: TrainingConfig,
    zero_stage: int = 2,
) -> Dict[str, Any]:
    """Generate DeepSpeed configuration dict.

    Args:
        config: Training configuration.
        zero_stage: ZeRO optimization stage (1, 2, or 3).

    Returns:
        DeepSpeed config dictionary.
    """
    ds_config = {
        "train_batch_size": config.per_device_batch_size * config.gradient_accumulation_steps,
        "train_micro_batch_size_per_gpu": config.per_device_batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "gradient_clipping": config.max_grad_norm,

        "bf16": {
            "enabled": config.dtype == "bfloat16",
        },
        "fp16": {
            "enabled": config.dtype == "float16",
            "loss_scale": 0,
            "loss_scale_window": 1000,
        },

        "zero_optimization": {
            "stage": zero_stage,
            "overlap_comm": True,
            "contiguous_gradients": True,
            "reduce_bucket_size": 5e8,
            "allgather_bucket_size": 5e8,
        },

        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": config.learning_rate,
                "betas": [config.adam_beta1, config.adam_beta2],
                "eps": config.adam_epsilon,
                "weight_decay": config.weight_decay,
            },
        },

        "scheduler": {
            "type": "WarmupCosineWithMinLR" if config.lr_scheduler == "cosine" else "WarmupLR",
            "params": {
                "warmup_min_lr": config.learning_rate * config.min_lr_ratio,
                "warmup_max_lr": config.learning_rate,
                "warmup_num_steps": config.warmup_steps,
                "total_num_steps": config.max_steps,
            },
        },

        "activation_checkpointing": {
            "partition_activations": config.gradient_checkpointing,
            "contiguous_memory_optimization": False,
        },

        "wall_clock_breakdown": False,
        "steps_per_print": config.log_steps,
    }

    # ZeRO-3 specific settings
    if zero_stage == 3:
        ds_config["zero_optimization"]["stage3_max_live_parameters"] = 1e9
        ds_config["zero_optimization"]["stage3_max_reuse_distance"] = 1e9
        ds_config["zero_optimization"]["stage3_prefetch_bucket_size"] = 5e8
        ds_config["zero_optimization"]["stage3_param_persistence_threshold"] = 1e6

        if config.cpu_offload:
            ds_config["zero_optimization"]["offload_optimizer"] = {
                "device": "cpu",
                "pin_memory": True,
            }
            ds_config["zero_optimization"]["offload_param"] = {
                "device": "cpu",
                "pin_memory": True,
            }

    return ds_config


def is_main_process() -> bool:
    """Check if current process is the main process (rank 0)."""
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def barrier():
    """Synchronize all processes."""
    if dist.is_initialized():
        dist.barrier()


def all_reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    """All-reduce a tensor and compute mean across processes."""
    if not dist.is_initialized():
        return tensor
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= dist.get_world_size()
    return tensor
