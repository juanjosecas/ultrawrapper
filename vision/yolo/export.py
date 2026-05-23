"""Model export and export benchmarking."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from vision.yolo.devices import detect_device
from vision.yolo.utils import load_model


_SUPPORTED_FORMATS = frozenset(
    ["onnx", "openvino", "tensorrt", "coreml", "tflite", "torchscript"]
)


def export_model(
    model_path: str,
    fmt: str = "onnx",
    imgsz: int = 640,
    device: Optional[str] = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Export a YOLO model to the requested format.

    Parameters
    ----------
    fmt:
        Target format: ``'onnx'``, ``'openvino'``, ``'tensorrt'``, ``'coreml'``,
        ``'tflite'``, ``'torchscript'``.

    Returns
    -------
    dict with ``exported_path`` and ``format``.
    """
    if fmt not in _SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported format {fmt!r}. Choose from {sorted(_SUPPORTED_FORMATS)}.")

    device = device or detect_device()
    model = load_model(model_path)
    exported = model.export(format=fmt, imgsz=imgsz, device=device, **kwargs)

    return {
        "exported_path": str(exported),
        "format": fmt,
    }


def benchmark_export(
    model_path: str,
    fmt: str = "onnx",
    imgsz: int = 640,
    n_warmup: int = 5,
    n_runs: int = 50,
    device: Optional[str] = None,
    batch_size: int = 1,
    save_to: Optional[str | Path] = None,
) -> pd.DataFrame:
    """Export a model and benchmark its inference latency.

    Parameters
    ----------
    n_warmup:
        Number of warm-up inferences before timing.
    n_runs:
        Number of timed inference runs.

    Returns
    -------
    pd.DataFrame with benchmark metrics per run.
    """
    import numpy as np

    device = device or detect_device()

    # Export first
    exported = export_model(model_path, fmt=fmt, imgsz=imgsz, device=device)
    exported_path = exported["exported_path"]

    # Load exported model
    model = load_model(exported_path)

    dummy_input = _make_dummy_input(imgsz, batch_size)

    # Warm-up
    for _ in range(n_warmup):
        model.predict(dummy_input, device=device, verbose=False)

    # Timed runs
    latencies: list[float] = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        model.predict(dummy_input, device=device, verbose=False)
        latencies.append((time.perf_counter() - t0) * 1000)

    latencies_arr = np.array(latencies)
    fps = 1000.0 / latencies_arr.mean() * batch_size

    result = {
        "format": fmt,
        "imgsz": imgsz,
        "batch_size": batch_size,
        "device": device,
        "exported_path": exported_path,
        "latency_mean_ms": float(latencies_arr.mean()),
        "latency_p50_ms": float(np.percentile(latencies_arr, 50)),
        "latency_p95_ms": float(np.percentile(latencies_arr, 95)),
        "latency_p99_ms": float(np.percentile(latencies_arr, 99)),
        "fps": float(fps),
    }

    _add_file_size(result, exported_path)

    df = pd.DataFrame([result])

    if save_to:
        from vision.yolo.serialization import save_dataframe

        save_dataframe(df, save_to)

    return df


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _make_dummy_input(imgsz: int, batch_size: int):
    import numpy as np

    return [np.zeros((imgsz, imgsz, 3), dtype=np.uint8)] * batch_size


def _add_file_size(result: dict, path: str) -> None:
    try:
        size_mb = Path(path).stat().st_size / 1024 ** 2
        result["model_size_mb"] = round(size_mb, 2)
    except OSError:
        result["model_size_mb"] = None
