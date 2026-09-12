"""Utilities to turn YOLO prediction tables into machine-learning features."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def frame_features(
    predictions: pd.DataFrame,
    include_class_counts: bool = True,
    total_frames: Optional[int] = None,
) -> pd.DataFrame:
    """Aggregate detection-level predictions into one row per video frame.

    When ``total_frames`` is provided, frames with no detections are restored
    with ``detection_count = 0``. This is important for temporal ML because
    otherwise absence of detections disappears from the dataset entirely.
    """
    required = {"frame", "confidence", "xmin", "ymin", "xmax", "ymax"}
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    data = predictions.copy()
    data["bbox_width"] = data["xmax"] - data["xmin"]
    data["bbox_height"] = data["ymax"] - data["ymin"]
    data["bbox_area"] = data["bbox_width"] * data["bbox_height"]
    data["center_x"] = (data["xmin"] + data["xmax"]) / 2.0
    data["center_y"] = (data["ymin"] + data["ymax"]) / 2.0

    features = data.groupby("frame").agg(
        detection_count=("confidence", "size"),
        confidence_mean=("confidence", "mean"),
        confidence_max=("confidence", "max"),
        confidence_std=("confidence", "std"),
        bbox_area_mean=("bbox_area", "mean"),
        bbox_area_max=("bbox_area", "max"),
        center_x_mean=("center_x", "mean"),
        center_y_mean=("center_y", "mean"),
    )

    features = features.reset_index()
    features["confidence_std"] = features["confidence_std"].fillna(0.0)

    if "timestamp" in data.columns:
        timestamps = data.groupby("frame")["timestamp"].first().reset_index()
        features = features.merge(timestamps, on="frame", how="left")

    if include_class_counts and "class_name" in data.columns:
        class_counts = pd.crosstab(data["frame"], data["class_name"])
        class_counts.columns = [f"count_{name}" for name in class_counts.columns]
        class_counts = class_counts.reset_index()
        features = features.merge(class_counts, on="frame", how="left")

    if total_frames is not None:
        if total_frames < 0:
            raise ValueError("total_frames must be >= 0")

        all_frames = pd.DataFrame({"frame": range(total_frames)})
        features = all_frames.merge(features, on="frame", how="left")
        features["detection_count"] = features["detection_count"].fillna(0).astype(int)

        class_columns = [c for c in features.columns if c.startswith("count_")]
        for column in class_columns:
            features[column] = features[column].fillna(0).astype(int)

    return features.sort_values("frame").reset_index(drop=True)


def track_features(predictions: pd.DataFrame) -> pd.DataFrame:
    """Aggregate tracked detections into one row per ``track_id``."""
    required = {"track_id", "frame", "confidence", "xmin", "ymin", "xmax", "ymax"}
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    data = predictions.dropna(subset=["track_id"]).copy()
    if data.empty:
        return pd.DataFrame()

    data["center_x"] = (data["xmin"] + data["xmax"]) / 2.0
    data["center_y"] = (data["ymin"] + data["ymax"]) / 2.0
    data["bbox_area"] = (
        (data["xmax"] - data["xmin"])
        * (data["ymax"] - data["ymin"])
    )
    data = data.sort_values(["track_id", "frame"])

    data["dx"] = data.groupby("track_id")["center_x"].diff()
    data["dy"] = data.groupby("track_id")["center_y"].diff()
    data["step_distance"] = np.sqrt(data["dx"] ** 2 + data["dy"] ** 2)

    if "timestamp" in data.columns:
        data["dt"] = data.groupby("track_id")["timestamp"].diff()
        data["speed"] = data["step_distance"] / data["dt"].replace(0, np.nan)
    else:
        data["speed"] = np.nan

    features = data.groupby("track_id").agg(
        first_frame=("frame", "min"),
        last_frame=("frame", "max"),
        n_frames=("frame", "nunique"),
        confidence_mean=("confidence", "mean"),
        confidence_max=("confidence", "max"),
        bbox_area_mean=("bbox_area", "mean"),
        distance_total=("step_distance", "sum"),
        speed_mean=("speed", "mean"),
        speed_max=("speed", "max"),
    ).reset_index()

    if "class_name" in data.columns:
        classes = data.groupby("track_id")["class_name"].agg(_most_common_value)
        classes = classes.reset_index()
        features = features.merge(classes, on="track_id", how="left")

    features["track_span_frames"] = (
        features["last_frame"] - features["first_frame"] + 1
    )
    return features


def prepare_xy(
    features: pd.DataFrame,
    target: str,
    drop_columns: Optional[list[str]] = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Return predictors ``X`` and target ``y`` without hidden preprocessing."""
    if target not in features.columns:
        raise ValueError(f"Target column not found: {target}")

    drop_columns = drop_columns or []
    columns_to_drop = [target] + [c for c in drop_columns if c in features.columns]
    X = features.drop(columns=columns_to_drop)
    y = features[target].copy()
    return X, y


def merge_labels(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    on: str = "frame",
    how: str = "inner",
) -> pd.DataFrame:
    """Attach experimental or manually curated labels to inference features."""
    if on not in features.columns:
        raise ValueError(f"Join column not found in features: {on}")
    if on not in labels.columns:
        raise ValueError(f"Join column not found in labels: {on}")
    return features.merge(labels, on=on, how=how)


def shap_values(
    model,
    X: pd.DataFrame,
    max_samples: int = 500,
):
    """Compute SHAP values for a downstream tabular model.

    This is intended for models trained on features produced by
    :func:`frame_features` or :func:`track_features`, not for explaining YOLO
    pixels directly.
    """
    try:
        import shap
    except ImportError as exc:
        raise ImportError(
            "Install SHAP with: pip install -e '.[explain]'"
        ) from exc

    data = X.iloc[:max_samples].copy()
    explainer = shap.Explainer(model, data)
    return explainer(data)


def _most_common_value(values: pd.Series):
    mode = values.mode()
    if not mode.empty:
        return mode.iloc[0]
    return values.iloc[0]
