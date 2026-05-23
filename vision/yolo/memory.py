"""Memory monitoring utilities – RAM and VRAM.

Uses psutil for RAM and pynvml for VRAM when available.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional


@dataclass
class MemorySnapshot:
    timestamp: float
    ram_used_mb: float
    ram_total_mb: float
    vram_used_mb: Optional[float]
    vram_total_mb: Optional[float]


def get_memory_snapshot() -> MemorySnapshot:
    """Capture current RAM and VRAM usage."""
    import psutil

    vm = psutil.virtual_memory()
    vram_used: Optional[float] = None
    vram_total: Optional[float] = None

    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        vram_used = mem.used / 1024 ** 2
        vram_total = mem.total / 1024 ** 2
    except Exception:
        pass

    return MemorySnapshot(
        timestamp=time.time(),
        ram_used_mb=vm.used / 1024 ** 2,
        ram_total_mb=vm.total / 1024 ** 2,
        vram_used_mb=vram_used,
        vram_total_mb=vram_total,
    )


def log_memory_usage(label: str = "") -> MemorySnapshot:
    """Print and return a memory snapshot."""
    snap = get_memory_snapshot()
    parts = [f"RAM {snap.ram_used_mb:.0f}/{snap.ram_total_mb:.0f} MB"]
    if snap.vram_used_mb is not None:
        parts.append(f"VRAM {snap.vram_used_mb:.0f}/{snap.vram_total_mb:.0f} MB")
    prefix = f"[{label}] " if label else ""
    print(f"{prefix}Memory: {' | '.join(parts)}")
    return snap


class MemoryTracker:
    """Accumulate memory snapshots over time."""

    def __init__(self) -> None:
        self._snapshots: list[MemorySnapshot] = []

    def record(self, label: str = "") -> MemorySnapshot:
        snap = log_memory_usage(label)
        self._snapshots.append(snap)
        return snap

    def to_dataframe(self):
        """Return snapshots as a pandas DataFrame."""
        import pandas as pd

        return pd.DataFrame(
            [
                {
                    "timestamp": s.timestamp,
                    "ram_used_mb": s.ram_used_mb,
                    "ram_total_mb": s.ram_total_mb,
                    "vram_used_mb": s.vram_used_mb,
                    "vram_total_mb": s.vram_total_mb,
                }
                for s in self._snapshots
            ]
        )
