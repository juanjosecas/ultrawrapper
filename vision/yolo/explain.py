"""Lightweight explainability helpers for detection outputs."""

from __future__ import annotations

from collections.abc import Callable

import cv2
import numpy as np
import pandas as pd


def detection_density_map(
    detections: pd.DataFrame,
    image_shape: tuple[int, int] | tuple[int, int, int],
    sigma: float = 25.0,
    weight_by_confidence: bool = True,
) -> np.ndarray:
    """Build a normalized density map from detection centers.

    The map is intentionally model-independent. Each bounding-box center contributes
    either one unit or its confidence value before Gaussian smoothing.
    """
    height, width = image_shape[:2]
    density = np.zeros((height, width), dtype=np.float32)

    if detections.empty:
        return density

    required = {"xmin", "ymin", "xmax", "ymax"}
    missing = required - set(detections.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    for row in detections.itertuples(index=False):
        x = int(round((float(row.xmin) + float(row.xmax)) / 2.0))
        y = int(round((float(row.ymin) + float(row.ymax)) / 2.0))
        if 0 <= x < width and 0 <= y < height:
            weight = 1.0
            if weight_by_confidence and hasattr(row, "confidence"):
                weight = float(row.confidence)
            density[y, x] += weight

    if sigma > 0:
        density = cv2.GaussianBlur(density, (0, 0), sigmaX=sigma, sigmaY=sigma)

    maximum = float(density.max())
    if maximum > 0:
        density /= maximum
    return density


def overlay_heatmap(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.45,
) -> np.ndarray:
    """Overlay a normalized heatmap on a BGR image."""
    if image.shape[:2] != heatmap.shape[:2]:
        raise ValueError("image and heatmap must have the same height and width")
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be between 0 and 1")

    normalized = np.clip(heatmap, 0.0, 1.0)
    colored = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    return cv2.addWeighted(image, 1.0 - alpha, colored, alpha, 0.0)


def occlusion_sensitivity(
    image: np.ndarray,
    predictor: Callable[[np.ndarray], pd.DataFrame],
    patch_size: int = 64,
    stride: int | None = None,
    fill_value: int | tuple[int, int, int] = 0,
) -> np.ndarray:
    """Estimate image regions supporting detections by model-agnostic occlusion.

    ``predictor`` must accept one BGR image and return a DataFrame containing a
    ``confidence`` column. The score is the sum of detection confidences. Regions
    whose occlusion decreases that score receive larger sensitivity values.
    """
    if patch_size < 1:
        raise ValueError("patch_size must be >= 1")
    if stride is None:
        stride = patch_size
    if stride < 1:
        raise ValueError("stride must be >= 1")

    baseline = predictor(image)
    baseline_score = _confidence_score(baseline)

    height, width = image.shape[:2]
    sensitivity = np.zeros((height, width), dtype=np.float32)
    counts = np.zeros((height, width), dtype=np.float32)

    for y0 in range(0, height, stride):
        y1 = min(y0 + patch_size, height)
        for x0 in range(0, width, stride):
            x1 = min(x0 + patch_size, width)
            occluded = image.copy()
            occluded[y0:y1, x0:x1] = fill_value
            score = _confidence_score(predictor(occluded))
            drop = max(baseline_score - score, 0.0)
            sensitivity[y0:y1, x0:x1] += drop
            counts[y0:y1, x0:x1] += 1.0

    np.divide(sensitivity, counts, out=sensitivity, where=counts > 0)
    maximum = float(sensitivity.max())
    if maximum > 0:
        sensitivity /= maximum
    return sensitivity


def _confidence_score(detections: pd.DataFrame) -> float:
    if detections is None or detections.empty:
        return 0.0
    if "confidence" not in detections.columns:
        raise ValueError("predictor output must contain a confidence column")
    return float(detections["confidence"].sum())
