"""Inference, training, and exported-model benchmark utilities."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from vision.yolo.devices import detect_device, get_gpu_memory
from vision.yolo.memory import get_memory_snapshot
from vision.yolo.utils import load_model


def benchmark_inference(
    model_path: str,
    imgsz: int = 640,
    batch_size: int = 1,
    n_warmup: int = 10,
    n_runs: int = 100,
    device: Optional[str] = None,
    save_to: Optional[str | Path] = None,
) -> pd.DataFrame:
    """Benchmark raw inference latency and throughput.

    Parameters
    ----------
    n_warmup:
        Inference passes to discard before timing.
    n_runs:
        Number of timed passes.

    Returns
    -------
    pd.DataFrame with latency statistics, FPS, and memory metrics.
    """
    device = device or detect_device()
    model = load_model(model_path)
    dummy = _make_dummy_input(imgsz, batch_size)

    # Warm-up
    for _ in range(n_warmup):
        model.predict(dummy, device=device, verbose=False)

    snap_before = get_memory_snapshot()
    latencies: list[float] = []

    for _ in range(n_runs):
        t0 = time.perf_counter()
        model.predict(dummy, device=device, verbose=False)
        latencies.append((time.perf_counter() - t0) * 1000)

    snap_after = get_memory_snapshot()
    arr = np.array(latencies)
    fps = 1000.0 / arr.mean() * batch_size

    result = {
        "model": model_path,
        "device": device,
        "imgsz": imgsz,
        "batch_size": batch_size,
        "n_runs": n_runs,
        "latency_mean_ms": float(arr.mean()),
        "latency_std_ms": float(arr.std()),
        "latency_p50_ms": float(np.percentile(arr, 50)),
        "latency_p95_ms": float(np.percentile(arr, 95)),
        "latency_p99_ms": float(np.percentile(arr, 99)),
        "fps": float(fps),
        "ram_used_mb": snap_after.ram_used_mb,
        "vram_used_mb": snap_after.vram_used_mb,
    }

    df = pd.DataFrame([result])
    if save_to:
        from vision.yolo.serialization import save_dataframe

        save_dataframe(df, save_to)

    return df


def benchmark_training(
    model_path: str,
    data: str,
    epochs: int = 5,
    imgsz: int = 640,
    batch: int = 16,
    device: Optional[str] = None,
    save_to: Optional[str | Path] = None,
) -> pd.DataFrame:
    """Run a short training cycle and record throughput + memory usage."""
    device = device or detect_device()
    snap_before = get_memory_snapshot()
    t0 = time.perf_counter()

    from vision.yolo.train import train_model

    result = train_model(
        model_path, data=data, epochs=epochs, imgsz=imgsz, batch=batch, device=device
    )

    elapsed = time.perf_counter() - t0
    snap_after = get_memory_snapshot()

    row = {
        "model": model_path,
        "device": device,
        "epochs": epochs,
        "batch": batch,
        "elapsed_s": elapsed,
        "epochs_per_s": epochs / elapsed,
        "ram_used_mb": snap_after.ram_used_mb,
        "vram_used_mb": snap_after.vram_used_mb,
        "results_dir": result["results_dir"],
    }

    df = pd.DataFrame([row])
    if save_to:
        from vision.yolo.serialization import save_dataframe

        save_dataframe(df, save_to)

    return df


def benchmark_exported_model(
    exported_path: str,
    imgsz: int = 640,
    batch_size: int = 1,
    n_warmup: int = 5,
    n_runs: int = 50,
    device: Optional[str] = None,
    save_to: Optional[str | Path] = None,
) -> pd.DataFrame:
    """Benchmark a previously exported model (ONNX, OpenVINO, etc.)."""
    return benchmark_inference(
        exported_path,
        imgsz=imgsz,
        batch_size=batch_size,
        n_warmup=n_warmup,
        n_runs=n_runs,
        device=device,
        save_to=save_to,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _make_dummy_input(imgsz: int, batch_size: int) -> list:
    return [np.zeros((imgsz, imgsz, 3), dtype=np.uint8)] * batch_size
