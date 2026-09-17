"""Feature extraction for downstream machine learning from YOLO inference tables."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def frame_features(detections: pd.DataFrame) -> pd.DataFrame:
    """Summarize detections into one numeric row per frame."""
    required = {"frame", "confidence", "xmin", "ymin", "xmax", "ymax"}
    missing = required - set(detections.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if detections.empty:
        return pd.DataFrame(
            columns=[
                "frame",
                "n_detections",
                "mean_confidence",
                "max_confidence",
                "mean_box_area",
                "total_box_area",
                "n_classes",
            ]
        )

    data = detections.copy()
    data["box_area"] = (data["xmax"] - data["xmin"]).clip(lower=0) * (
        data["ymax"] - data["ymin"]
    ).clip(lower=0)

    rows = []
    for frame, group in data.groupby("frame", sort=True):
        rows.append(
            {
                "frame": int(frame),
                "n_detections": int(len(group)),
                "mean_confidence": float(group["confidence"].mean()),
                "max_confidence": float(group["confidence"].max()),
                "mean_box_area": float(group["box_area"].mean()),
                "total_box_area": float(group["box_area"].sum()),
                "n_classes": int(group["class_id"].nunique()) if "class_id" in group else 0,
            }
        )
    return pd.DataFrame(rows)


def track_features(detections: pd.DataFrame, fps: float | None = None) -> pd.DataFrame:
    """Summarize tracked detections into one row per ``track_id``.

    Distance is measured between successive box centers in pixels. If ``fps`` is
    provided, mean and maximum speeds are returned in pixels/second; otherwise
    they are pixels/frame.
    """
    required = {"frame", "track_id", "xmin", "ymin", "xmax", "ymax", "confidence"}
    missing = required - set(detections.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if fps is not None and fps <= 0:
        raise ValueError("fps must be > 0")

    data = detections.dropna(subset=["track_id"]).copy()
    if data.empty:
        return pd.DataFrame()

    data["center_x"] = (data["xmin"] + data["xmax"]) / 2.0
    data["center_y"] = (data["ymin"] + data["ymax"]) / 2.0

    rows = []
    for track_id, group in data.groupby("track_id", sort=True):
        group = group.sort_values("frame")
        dx = group["center_x"].diff().to_numpy(dtype=float)
        dy = group["center_y"].diff().to_numpy(dtype=float)
        frame_delta = group["frame"].diff().to_numpy(dtype=float)
        distance = np.sqrt(dx**2 + dy**2)

        valid = np.isfinite(distance) & np.isfinite(frame_delta) & (frame_delta > 0)
        step_speed = distance[valid] / frame_delta[valid]
        if fps is not None:
            step_speed = step_speed * fps

        rows.append(
            {
                "track_id": int(track_id),
                "n_observations": int(len(group)),
                "first_frame": int(group["frame"].min()),
                "last_frame": int(group["frame"].max()),
                "frame_span": int(group["frame"].max() - group["frame"].min()),
                "mean_confidence": float(group["confidence"].mean()),
                "total_distance": float(np.nansum(distance)),
                "mean_speed": float(np.nanmean(step_speed)) if len(step_speed) else 0.0,
                "max_speed": float(np.nanmax(step_speed)) if len(step_speed) else 0.0,
            }
        )
    return pd.DataFrame(rows)


def merge_labels(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    on: str | Sequence[str],
    how: str = "inner",
) -> pd.DataFrame:
    """Merge feature rows with experimental metadata or labels."""
    keys = [on] if isinstance(on, str) else list(on)
    missing_features = set(keys) - set(features.columns)
    missing_labels = set(keys) - set(labels.columns)
    if missing_features:
        raise ValueError(f"Missing feature keys: {sorted(missing_features)}")
    if missing_labels:
        raise ValueError(f"Missing label keys: {sorted(missing_labels)}")
    return features.merge(labels, on=keys, how=how)


def prepare_xy(
    table: pd.DataFrame,
    target: str,
    drop: Sequence[str] | None = None,
    numeric_only: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    """Split a feature table into ``X`` and ``y`` for scikit-learn-style models."""
    if target not in table.columns:
        raise ValueError(f"Target column not found: {target}")

    drop_columns = list(drop or [])
    y = table[target].copy()
    X = table.drop(columns=[target] + drop_columns, errors="ignore")
    if numeric_only:
        X = X.select_dtypes(include=["number", "bool"])
    return X, y


def shap_values(model, X: pd.DataFrame):
    """Return SHAP values for a downstream tabular model.

    SHAP stays decoupled from private Ultralytics layers. Install it through the
    ``explain`` extra when needed.
    """
    try:
        import shap
    except ImportError as exc:
        raise ImportError("Install SHAP with: pip install -e '.[explain]'") from exc

    explainer = shap.Explainer(model, X)
    return explainer(X)
