"""Distributed computing helper utilities."""

import os
import torch
import torch.distributed as dist
from typing import Optional


def get_rank() -> int:
    """Get current process rank."""
    if dist.is_initialized():
        return dist.get_rank()
    return 0


def get_world_size() -> int:
    """Get total number of processes."""
    if dist.is_initialized():
        return dist.get_world_size()
    return 1


def is_main() -> bool:
    """Check if this is the main process (rank 0)."""
    return get_rank() == 0


def print_rank0(*args, **kwargs):
    """Print only from rank 0."""
    if is_main():
        print(*args, **kwargs)


def sync_tensor(tensor: torch.Tensor, op: str = "sum") -> torch.Tensor:
    """Synchronize a tensor across all processes.

    Args:
        tensor: Tensor to synchronize.
        op: Reduction operation ('sum', 'mean', 'max', 'min').

    Returns:
        Reduced tensor.
    """
    if not dist.is_initialized() or get_world_size() == 1:
        return tensor

    ops = {
        "sum": dist.ReduceOp.SUM,
        "max": dist.ReduceOp.MAX,
        "min": dist.ReduceOp.MIN,
    }

    reduce_op = ops.get(op, dist.ReduceOp.SUM)
    dist.all_reduce(tensor, op=reduce_op)

    if op == "mean":
        tensor = tensor / get_world_size()

    return tensor


def broadcast_object(obj, src: int = 0):
    """Broadcast a Python object from src rank to all ranks."""
    if not dist.is_initialized() or get_world_size() == 1:
        return obj

    object_list = [obj]
    dist.broadcast_object_list(object_list, src=src)
    return object_list[0]
