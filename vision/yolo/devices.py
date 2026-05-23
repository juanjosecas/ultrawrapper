"""Device detection and GPU memory management.

Supports CUDA and CPU; MPS is not prioritised.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DeviceInfo:
    device: str
    cuda_available: bool
    cuda_device_count: int
    cuda_device_name: Optional[str]
    cuda_total_memory_mb: Optional[float]
    cuda_free_memory_mb: Optional[float]
    cpu_count: int
    ram_total_mb: float
    ram_available_mb: float


def detect_device() -> str:
    """Return the best available device string ('cuda:0' or 'cpu')."""
    import torch

    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def get_device_info() -> DeviceInfo:
    """Return detailed device information."""
    import torch
    import psutil

    cuda_available = torch.cuda.is_available()
    cuda_count = torch.cuda.device_count() if cuda_available else 0
    cuda_name: Optional[str] = None
    cuda_total: Optional[float] = None
    cuda_free: Optional[float] = None

    if cuda_available:
        cuda_name = torch.cuda.get_device_name(0)
        mem = torch.cuda.mem_get_info(0)
        cuda_free = mem[0] / 1024 ** 2
        cuda_total = mem[1] / 1024 ** 2

    vm = psutil.virtual_memory()
    return DeviceInfo(
        device=detect_device(),
        cuda_available=cuda_available,
        cuda_device_count=cuda_count,
        cuda_device_name=cuda_name,
        cuda_total_memory_mb=cuda_total,
        cuda_free_memory_mb=cuda_free,
        cpu_count=psutil.cpu_count(logical=True) or 1,
        ram_total_mb=vm.total / 1024 ** 2,
        ram_available_mb=vm.available / 1024 ** 2,
    )


def get_gpu_memory() -> dict[str, float]:
    """Return current GPU memory stats (MB) for device 0.

    Returns empty dict when CUDA is not available.
    """
    import torch

    if not torch.cuda.is_available():
        return {}

    free, total = torch.cuda.mem_get_info(0)
    used = total - free
    return {
        "total_mb": total / 1024 ** 2,
        "used_mb": used / 1024 ** 2,
        "free_mb": free / 1024 ** 2,
    }


def clear_gpu_memory() -> None:
    """Release cached GPU memory and run Python GC."""
    import gc
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def estimate_optimal_batch_size(
    model_size_mb: float = 100.0,
    safety_factor: float = 0.7,
    min_batch: int = 1,
    max_batch: int = 64,
) -> int:
    """Estimate a safe batch size based on available GPU (or RAM) memory.

    Parameters
    ----------
    model_size_mb:
        Approximate memory footprint of one inference pass (MB).
    safety_factor:
        Fraction of available memory to use (0–1).
    min_batch / max_batch:
        Hard limits on the returned batch size.
    """
    import torch
    import psutil

    if torch.cuda.is_available():
        free_mb, _ = torch.cuda.mem_get_info(0)
        available_mb = (free_mb / 1024 ** 2) * safety_factor
    else:
        vm = psutil.virtual_memory()
        available_mb = (vm.available / 1024 ** 2) * safety_factor

    batch = max(min_batch, int(available_mb // model_size_mb))
    return min(batch, max_batch)
