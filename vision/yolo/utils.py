"""Tensor conversion utilities and general helpers."""

from __future__ import annotations

from typing import Any


def tensor_to_numpy(x: Any):
    """Convert a tensor (or array-like) to a numpy ndarray.

    Handles CUDA tensors, regular tensors, and plain array-likes.
    """
    import numpy as np

    try:
        import torch

        if isinstance(x, torch.Tensor):
            return x.detach().cpu().contiguous().numpy()
    except ImportError:
        pass

    return np.asarray(x)


def tensor_to_cpu(x: Any) -> Any:
    """Move a tensor to CPU if it lives on GPU; otherwise return unchanged."""
    try:
        import torch

        if isinstance(x, torch.Tensor):
            return x.detach().cpu()
    except ImportError:
        pass
    return x


def tensor_to_dataframe(x: Any, columns: list[str] | None = None):
    """Convert a 2-D tensor or array to a pandas DataFrame."""
    import pandas as pd

    arr = tensor_to_numpy(x)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return pd.DataFrame(arr, columns=columns)


def to_serializable(obj: Any) -> Any:
    """Recursively convert tensors and numpy objects to JSON-serialisable types."""
    import numpy as np

    try:
        import torch

        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().tolist()
    except ImportError:
        pass

    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_serializable(v) for v in obj]
    return obj


def ensure_rgb(image) -> "np.ndarray":  # type: ignore[name-defined]
    """Convert a BGR (cv2) image to RGB numpy array."""
    import cv2
    import numpy as np

    if isinstance(image, np.ndarray) and image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def xywh_to_xyxy(xywh) -> list[float]:
    """Convert [x_center, y_center, w, h] → [x1, y1, x2, y2]."""
    x, y, w, h = xywh
    return [x - w / 2, y - h / 2, x + w / 2, y + h / 2]


def xyxy_to_xywh(xyxy) -> list[float]:
    """Convert [x1, y1, x2, y2] → [x_center, y_center, w, h]."""
    x1, y1, x2, y2 = xyxy
    return [(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1]


def load_model(model_path: str):
    """Load a YOLO model lazily."""
    from ultralytics import YOLO

    return YOLO(model_path)
