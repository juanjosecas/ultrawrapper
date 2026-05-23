"""Metrics computation for detection, segmentation, and pose tasks."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def compute_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    """Compute IoU between two [x1, y1, x2, y2] boxes."""
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b

    inter_x1 = max(xa1, xb1)
    inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2)
    inter_y2 = min(ya2, yb2)

    inter_area = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
    area_a = (xa2 - xa1) * (ya2 - ya1)
    area_b = (xb2 - xb1) * (yb2 - yb1)
    union = area_a + area_b - inter_area

    return inter_area / union if union > 0 else 0.0


def compute_ap(precision: np.ndarray, recall: np.ndarray) -> float:
    """Compute Average Precision using the 11-point interpolation."""
    ap = 0.0
    for thr in np.linspace(0, 1, 11):
        prec_at_rec = precision[recall >= thr]
        ap += prec_at_rec.max() if prec_at_rec.size > 0 else 0.0
    return ap / 11.0


def precision_recall_from_df(
    df: pd.DataFrame,
    iou_threshold: float = 0.5,
    confidence_col: str = "confidence",
) -> pd.DataFrame:
    """Compute precision and recall at various confidence thresholds.

    Assumes ``df`` contains predicted detections with ground-truth match info
    (column ``is_tp``: 1 for true positive, 0 for false positive).

    If ``is_tp`` is not present this function returns an empty DataFrame –
    callers should pre-label detections before calling this function.
    """
    if "is_tp" not in df.columns:
        return pd.DataFrame(columns=["threshold", "precision", "recall", "f1"])

    df = df.sort_values(confidence_col, ascending=False).reset_index(drop=True)
    tp_cumsum = df["is_tp"].cumsum()
    fp_cumsum = (1 - df["is_tp"]).cumsum()
    total_gt = df["is_tp"].sum()

    precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-9)
    recall = tp_cumsum / (total_gt + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)

    return pd.DataFrame(
        {
            "threshold": df[confidence_col].values,
            "precision": precision.values,
            "recall": recall.values,
            "f1": f1.values,
        }
    )


def mean_average_precision(
    pr_df: pd.DataFrame,
    class_col: str = "class_name",
) -> float:
    """Compute mAP@0.5 from a per-detection PR DataFrame with a ``class_name`` column."""
    if pr_df.empty:
        return 0.0

    classes = pr_df[class_col].unique() if class_col in pr_df.columns else [None]
    aps: list[float] = []
    for cls in classes:
        sub = pr_df[pr_df[class_col] == cls] if cls is not None else pr_df
        sub = sub.sort_values("threshold", ascending=False)
        ap = compute_ap(sub["precision"].values, sub["recall"].values)
        aps.append(ap)
    return float(np.mean(aps))


def detections_per_class(df: pd.DataFrame) -> pd.DataFrame:
    """Return count of detections grouped by class."""
    return (
        df.groupby("class_name")
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )


def confidence_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Descriptive statistics of confidence scores per class."""
    return (
        df.groupby("class_name")["confidence"]
        .describe()
        .reset_index()
    )
