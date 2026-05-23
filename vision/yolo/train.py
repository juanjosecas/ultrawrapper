"""Training, resuming, and validation helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import pandas as pd

from vision.yolo.devices import detect_device
from vision.yolo.utils import load_model


def train_model(
    model_path: str,
    data: str,
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = 16,
    device: Optional[str] = None,
    project: str = "runs/train",
    name: str = "exp",
    save_metrics: bool = True,
    metrics_fmt: str = "parquet",
    **kwargs: Any,
) -> dict[str, Any]:
    """Train a YOLO model and return a results summary dict.

    Parameters
    ----------
    model_path:
        Path to base weights (e.g. ``'yolo11n.pt'``).
    data:
        Path to a YOLO ``data.yaml`` file.
    epochs / imgsz / batch:
        Standard Ultralytics training parameters.
    save_metrics:
        When True, saves per-epoch CSV/parquet alongside Ultralytics output.
    metrics_fmt:
        ``'parquet'``, ``'feather'``, or ``'csv'``.

    Returns
    -------
    dict with keys: ``results_dir``, ``best_weights``, ``last_weights``,
    ``metrics`` (DataFrame).
    """
    device = device or detect_device()
    model = load_model(model_path)

    results = model.train(
        data=data,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=project,
        name=name,
        **kwargs,
    )

    save_dir = Path(results.save_dir) if hasattr(results, "save_dir") else Path(project) / name
    metrics_df = _load_results_csv(save_dir)

    if save_metrics and not metrics_df.empty:
        from vision.yolo.serialization import save_dataframe

        save_dataframe(metrics_df, save_dir / f"metrics.{metrics_fmt}", fmt=metrics_fmt)

    return {
        "results_dir": str(save_dir),
        "best_weights": str(save_dir / "weights" / "best.pt"),
        "last_weights": str(save_dir / "weights" / "last.pt"),
        "metrics": metrics_df,
    }


def resume_training(
    weights: str,
    **kwargs: Any,
) -> dict[str, Any]:
    """Resume an interrupted training run from the given weights file."""
    from ultralytics import YOLO

    model = YOLO(weights)
    results = model.train(resume=True, **kwargs)
    save_dir = Path(results.save_dir) if hasattr(results, "save_dir") else Path(weights).parent
    metrics_df = _load_results_csv(save_dir)

    return {
        "results_dir": str(save_dir),
        "best_weights": str(save_dir / "weights" / "best.pt"),
        "last_weights": str(save_dir / "weights" / "last.pt"),
        "metrics": metrics_df,
    }


def validate_model(
    model_path: str,
    data: str,
    imgsz: int = 640,
    batch: int = 16,
    device: Optional[str] = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Validate a YOLO model and return validation metrics as a DataFrame."""
    device = device or detect_device()
    model = load_model(model_path)
    metrics = model.val(data=data, imgsz=imgsz, batch=batch, device=device, **kwargs)

    row: dict[str, Any] = {}
    if hasattr(metrics, "results_dict"):
        row = {k: float(v) for k, v in metrics.results_dict.items()}
    elif hasattr(metrics, "box"):
        b = metrics.box
        row = {
            "map50": float(b.map50),
            "map50_95": float(b.map),
            "precision": float(b.mp),
            "recall": float(b.mr),
        }

    return pd.DataFrame([row]) if row else pd.DataFrame()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_results_csv(save_dir: Path) -> pd.DataFrame:
    csv_path = save_dir / "results.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return pd.DataFrame()
