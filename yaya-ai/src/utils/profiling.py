"""Performance profiling tools for training and inference.

Measures throughput, memory usage, and computation time
to help optimize training and identify bottlenecks.
"""

import time
import torch
from typing import Optional, Dict
from contextlib import contextmanager


class Timer:
    """Simple timer for measuring code execution time."""

    def __init__(self, name: str = ""):
        self.name = name
        self.start_time = None
        self.elapsed = 0.0

    def start(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.start_time = time.perf_counter()

    def stop(self) -> float:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.elapsed = time.perf_counter() - self.start_time
        return self.elapsed

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()


@contextmanager
def cuda_memory_tracker(label: str = ""):
    """Context manager to track CUDA memory allocation."""
    if not torch.cuda.is_available():
        yield {}
        return

    torch.cuda.reset_peak_memory_stats()
    mem_before = torch.cuda.memory_allocated()

    stats = {}
    yield stats

    mem_after = torch.cuda.memory_allocated()
    mem_peak = torch.cuda.max_memory_allocated()

    stats["allocated_before_mb"] = mem_before / (1024 * 1024)
    stats["allocated_after_mb"] = mem_after / (1024 * 1024)
    stats["peak_mb"] = mem_peak / (1024 * 1024)
    stats["delta_mb"] = (mem_after - mem_before) / (1024 * 1024)

    if label:
        print(f"[Memory {label}] Before: {stats['allocated_before_mb']:.1f}MB, "
              f"After: {stats['allocated_after_mb']:.1f}MB, "
              f"Peak: {stats['peak_mb']:.1f}MB, "
              f"Delta: {stats['delta_mb']:.1f}MB")


def get_gpu_memory_info() -> Dict[str, float]:
    """Get current GPU memory usage information."""
    if not torch.cuda.is_available():
        return {}

    return {
        "allocated_mb": torch.cuda.memory_allocated() / (1024 * 1024),
        "reserved_mb": torch.cuda.memory_reserved() / (1024 * 1024),
        "max_allocated_mb": torch.cuda.max_memory_allocated() / (1024 * 1024),
        "total_mb": torch.cuda.get_device_properties(0).total_mem / (1024 * 1024),
    }


def estimate_model_memory(
    num_params: int,
    dtype_bytes: int = 2,
    optimizer_states: int = 2,
    gradient: bool = True,
) -> Dict[str, float]:
    """Estimate memory requirements for model training.

    Args:
        num_params: Number of model parameters.
        dtype_bytes: Bytes per parameter (2 for BF16, 4 for FP32).
        optimizer_states: Number of optimizer states per parameter (2 for AdamW).
        gradient: Whether to account for gradient memory.

    Returns:
        Dict with memory estimates in GB.
    """
    param_memory = num_params * dtype_bytes
    grad_memory = num_params * dtype_bytes if gradient else 0
    # AdamW stores 2 states (momentum + variance) in FP32
    optimizer_memory = num_params * 4 * optimizer_states

    total = param_memory + grad_memory + optimizer_memory

    return {
        "params_gb": param_memory / (1024 ** 3),
        "gradients_gb": grad_memory / (1024 ** 3),
        "optimizer_gb": optimizer_memory / (1024 ** 3),
        "total_gb": total / (1024 ** 3),
    }
