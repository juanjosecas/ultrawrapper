"""Explainability helpers for YOLO detections.

These functions deliberately avoid depending on Ultralytics internal layers.
They operate either on the stable prediction DataFrame or by repeatedly calling
``predict_image`` with controlled occlusions.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def detection_density_map(
    predictions: pd.DataFrame,
    image_width: int,
    image_height: int,
    bins: int = 64,
    weight_by_confidence: bool = True,
    class_name: Optional[str] = None,
) -> np.ndarray:
    """Build a 2D density map from bounding-box centers.

    Each detection contributes one point at the center of its bounding box.
    By default each point is weighted by the detection confidence.
    """
    required = {"xmin", "ymin", "xmax", "ymax"}
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    data = predictions.copy()
    if class_name is not None and "class_name" in data.columns:
        data = data[data["class_name"] == class_name]

    if data.empty:
        return np.zeros((bins, bins), dtype=float)

    x = (data["xmin"].to_numpy(float) + data["xmax"].to_numpy(float)) / 2.0
    y = (data["ymin"].to_numpy(float) + data["ymax"].to_numpy(float)) / 2.0

    weights = None
    if weight_by_confidence and "confidence" in data.columns:
        weights = data["confidence"].to_numpy(float)

    density, _, _ = np.histogram2d(
        y,
        x,
        bins=(bins, bins),
        range=((0, image_height), (0, image_width)),
        weights=weights,
    )
    return density


def plot_density_map(
    density: np.ndarray,
    image: Optional[np.ndarray] = None,
    alpha: float = 0.55,
    save_to: Optional[str] = None,
    show: bool = True,
) -> None:
    """Plot a density map alone or over an image."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))

    if image is not None:
        if image.ndim == 3 and image.shape[2] == 3:
            ax.imshow(image[..., ::-1])
        else:
            ax.imshow(image)

    ax.imshow(
        density,
        cmap="magma",
        alpha=alpha if image is not None else 1.0,
        interpolation="bilinear",
        extent=(0, image.shape[1], image.shape[0], 0) if image is not None else None,
        aspect="auto",
    )
    ax.set_title("Detection density")
    ax.set_axis_off()
    fig.tight_layout()

    if save_to:
        fig.savefig(save_to, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def occlusion_sensitivity(
    model_path: str,
    image: str | np.ndarray,
    target_class: Optional[str] = None,
    grid_size: int = 8,
    confidence: float = 0.25,
    occlusion_value: int = 127,
    device: Optional[str] = None,
) -> np.ndarray:
    """Estimate which image regions support a detection using occlusion.

    The image is divided into a regular grid. Each cell is hidden once and the
    drop in the best detection confidence is recorded. Larger positive values
    indicate regions whose removal weakens the target detection.

    This method is slower than gradient-based saliency but is model-agnostic and
    does not depend on private Ultralytics layer names.
    """
    import cv2

    from vision.yolo.infer import predict_image

    if isinstance(image, str):
        image_array = cv2.imread(image)
        if image_array is None:
            raise ValueError(f"Could not read image: {image}")
    else:
        image_array = np.asarray(image).copy()

    baseline = predict_image(
        model_path,
        image_array,
        confidence=confidence,
        device=device,
    )
    baseline_score = _best_detection_score(baseline, target_class)

    height, width = image_array.shape[:2]
    heatmap = np.zeros((grid_size, grid_size), dtype=float)

    for row in range(grid_size):
        y1 = int(row * height / grid_size)
        y2 = int((row + 1) * height / grid_size)

        for col in range(grid_size):
            x1 = int(col * width / grid_size)
            x2 = int((col + 1) * width / grid_size)

            occluded = image_array.copy()
            occluded[y1:y2, x1:x2] = occlusion_value

            predictions = predict_image(
                model_path,
                occluded,
                confidence=confidence,
                device=device,
            )
            score = _best_detection_score(predictions, target_class)
            heatmap[row, col] = baseline_score - score

    return heatmap


def _best_detection_score(
    predictions: pd.DataFrame,
    target_class: Optional[str],
) -> float:
    if predictions.empty or "confidence" not in predictions.columns:
        return 0.0

    data = predictions
    if target_class is not None and "class_name" in data.columns:
        data = data[data["class_name"] == target_class]

    if data.empty:
        return 0.0
    return float(data["confidence"].max())
