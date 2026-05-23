"""Matplotlib plotting helpers for YOLO DataFrame outputs.

The public functions plot directly by default and can also save the image with
``save_to``.  They intentionally consume plain DataFrames instead of returning
Ultralytics plotting objects.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Polygon, Rectangle

COCO_KEYPOINT_SKELETON: list[tuple[int, int]] = [
    (15, 13),
    (13, 11),
    (16, 14),
    (14, 12),
    (11, 12),
    (5, 11),
    (6, 12),
    (5, 6),
    (5, 7),
    (6, 8),
    (7, 9),
    (8, 10),
    (1, 2),
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (3, 5),
    (4, 6),
]


def _finish_plot(
    fig: "plt.Figure",
    save_to: Optional[str | Path] = None,
    show: bool = True,
    dpi: int = 150,
) -> None:
    if save_to:
        output_path = Path(save_to)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()


def _empty_axes_message(ax: "plt.Axes", message: str) -> None:
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes)
    ax.set_axis_off()


def _plot_color(index: int) -> tuple[float, float, float, float]:
    return plt.get_cmap("tab20")(index % 20)


def _row_label(row: pd.Series, show_confidence: bool = True) -> str:
    name = str(row.get("class_name", row.get("class_id", "")))
    if show_confidence and "confidence" in row and pd.notna(row["confidence"]):
        return f"{name} {float(row['confidence']):.2f}"
    return name


def _load_image_array(
    image: str | Path | np.ndarray,
    array_color_order: str = "rgb",
) -> np.ndarray:
    if isinstance(image, np.ndarray):
        arr = np.asarray(image)
        if arr.ndim == 2:
            return arr
        if arr.ndim == 3 and arr.shape[2] == 3 and array_color_order.lower() == "bgr":
            return arr[..., ::-1]
        return arr

    import cv2

    image_str = str(image)
    if image_str.startswith(("http://", "https://")):
        import urllib.request

        with urllib.request.urlopen(image_str) as response:
            data = np.frombuffer(response.read(), dtype=np.uint8)
        arr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    else:
        arr = cv2.imread(image_str, cv2.IMREAD_COLOR)

    if arr is None:
        raise ValueError(f"Could not read image: {image}")
    return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)


def _filter_detections(
    df: pd.DataFrame,
    frame: Optional[int] = None,
    confidence_threshold: Optional[float] = None,
    max_detections: Optional[int] = None,
) -> pd.DataFrame:
    data = df.copy()
    if frame is not None and "frame" in data.columns:
        data = data[data["frame"] == frame]
    if confidence_threshold is not None and "confidence" in data.columns:
        confidence = pd.to_numeric(data["confidence"], errors="coerce")
        data = data[confidence >= confidence_threshold]
    if max_detections is not None:
        data = data.head(max_detections)
    return data.reset_index(drop=True)


def _valid_polygon(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, float) and pd.isna(value):
        return False
    if not isinstance(value, (Sequence, np.ndarray)):
        return False
    return len(value) >= 3


def _keypoints_array(value: Any) -> np.ndarray:
    if value is None:
        return np.empty((0, 3), dtype=float)
    if isinstance(value, float) and pd.isna(value):
        return np.empty((0, 3), dtype=float)

    arr = np.asarray(value, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 2:
        return np.empty((0, 3), dtype=float)
    if arr.shape[1] == 2:
        visibility = np.ones((arr.shape[0], 1), dtype=float)
        arr = np.hstack([arr, visibility])
    return arr[:, :3]


def _draw_polygon(
    ax: "plt.Axes",
    polygon: Any,
    color: tuple[float, float, float, float],
    alpha: float = 0.25,
) -> None:
    if not _valid_polygon(polygon):
        return
    points = np.asarray(polygon, dtype=float)
    if points.ndim != 2 or points.shape[1] < 2:
        return
    ax.add_patch(
        Polygon(
            points[:, :2],
            closed=True,
            facecolor=color,
            edgecolor=color,
            linewidth=1.5,
            alpha=alpha,
        )
    )


def _draw_box(
    ax: "plt.Axes",
    row: pd.Series,
    color: tuple[float, float, float, float],
    linewidth: float = 2.0,
) -> None:
    required = ["xmin", "ymin", "xmax", "ymax"]
    if any(col not in row or pd.isna(row[col]) for col in required):
        return
    xmin, ymin, xmax, ymax = (float(row[col]) for col in required)
    ax.add_patch(
        Rectangle(
            (xmin, ymin),
            xmax - xmin,
            ymax - ymin,
            fill=False,
            edgecolor=color,
            linewidth=linewidth,
        )
    )


def _draw_label(
    ax: "plt.Axes",
    row: pd.Series,
    color: tuple[float, float, float, float],
    show_confidence: bool = True,
) -> None:
    if "xmin" not in row or "ymin" not in row or pd.isna(row["xmin"]) or pd.isna(row["ymin"]):
        return
    text = _row_label(row, show_confidence=show_confidence)
    ax.text(
        float(row["xmin"]),
        max(0.0, float(row["ymin"]) - 3.0),
        text,
        color="white",
        fontsize=9,
        va="bottom",
        bbox={"facecolor": color, "edgecolor": color, "alpha": 0.85, "pad": 2},
    )


def _draw_keypoints(
    ax: "plt.Axes",
    keypoints: Any,
    color: tuple[float, float, float, float],
    keypoint_threshold: float = 0.25,
    skeleton: Optional[Sequence[tuple[int, int]]] = None,
    point_size: float = 24,
    linewidth: float = 2.0,
) -> None:
    arr = _keypoints_array(keypoints)
    if arr.size == 0:
        return

    visible = arr[:, 2] >= keypoint_threshold
    skeleton = skeleton if skeleton is not None else COCO_KEYPOINT_SKELETON
    for start, end in skeleton:
        if start >= len(arr) or end >= len(arr):
            continue
        if not (visible[start] and visible[end]):
            continue
        ax.plot(
            [arr[start, 0], arr[end, 0]],
            [arr[start, 1], arr[end, 1]],
            color=color,
            linewidth=linewidth,
            alpha=0.85,
        )

    visible_points = arr[visible]
    if len(visible_points):
        ax.scatter(
            visible_points[:, 0],
            visible_points[:, 1],
            s=point_size,
            color=color,
            edgecolors="white",
            linewidths=0.6,
            zorder=5,
        )


def _setup_detection_axes(
    ax: "plt.Axes",
    image: Optional[str | Path | np.ndarray],
    df: pd.DataFrame,
    image_width: Optional[int],
    image_height: Optional[int],
    array_color_order: str,
) -> None:
    if image is not None:
        img = _load_image_array(image, array_color_order=array_color_order)
        ax.imshow(img)
        ax.set_xlim(0, img.shape[1])
        ax.set_ylim(img.shape[0], 0)
    else:
        xmax = pd.to_numeric(df["xmax"], errors="coerce").max() if "xmax" in df else np.nan
        ymax = pd.to_numeric(df["ymax"], errors="coerce").max() if "ymax" in df else np.nan
        width = image_width or int(max(float(xmax) if pd.notna(xmax) else 640.0, 1.0))
        height = image_height or int(max(float(ymax) if pd.notna(ymax) else 640.0, 1.0))
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)
        ax.set_facecolor("#f7f7f7")
    ax.set_axis_off()


def plot_image_detections(
    image: Optional[str | Path | np.ndarray],
    detections_df: pd.DataFrame,
    frame: Optional[int] = None,
    confidence_threshold: Optional[float] = None,
    max_detections: Optional[int] = None,
    draw_boxes: bool = True,
    draw_polygons: bool = True,
    draw_keypoints: bool = True,
    draw_labels: bool = True,
    show_confidence: bool = True,
    keypoint_threshold: float = 0.25,
    skeleton: Optional[Sequence[tuple[int, int]]] = None,
    image_width: Optional[int] = None,
    image_height: Optional[int] = None,
    array_color_order: str = "rgb",
    title: Optional[str] = None,
    figsize: tuple[float, float] = (10, 8),
    save_to: Optional[str | Path] = None,
    show: bool = True,
    dpi: int = 150,
) -> None:
    """Plot YOLO detections directly on an image or on a blank pixel canvas.

    ``detections_df`` can contain boxes, segmentation polygons and pose
    keypoints in the columns produced by ``ultralytics_result_to_dataframe``.
    """
    data = _filter_detections(
        detections_df,
        frame=frame,
        confidence_threshold=confidence_threshold,
        max_detections=max_detections,
    )

    fig, ax = plt.subplots(figsize=figsize)
    _setup_detection_axes(ax, image, data, image_width, image_height, array_color_order)
    if title:
        ax.set_title(title)

    if data.empty:
        _empty_axes_message(ax, "No detections")
        fig.tight_layout()
        _finish_plot(fig, save_to=save_to, show=show, dpi=dpi)
        return

    for idx, row in data.iterrows():
        if "class_id" in row and pd.notna(row["class_id"]):
            color_idx = int(row["class_id"])
        else:
            color_idx = idx
        color = _plot_color(color_idx)
        if draw_polygons and "polygon" in row:
            _draw_polygon(ax, row["polygon"], color)
        if draw_boxes:
            _draw_box(ax, row, color)
        if draw_keypoints and "keypoints" in row:
            _draw_keypoints(
                ax,
                row["keypoints"],
                color,
                keypoint_threshold=keypoint_threshold,
                skeleton=skeleton,
            )
        if draw_labels:
            _draw_label(ax, row, color, show_confidence=show_confidence)

    fig.tight_layout()
    _finish_plot(fig, save_to=save_to, show=show, dpi=dpi)


def plot_bounding_boxes(
    image: Optional[str | Path | np.ndarray],
    detections_df: pd.DataFrame,
    **kwargs: Any,
) -> None:
    """Plot only bounding boxes and labels."""
    plot_image_detections(
        image,
        detections_df,
        draw_boxes=True,
        draw_polygons=False,
        draw_keypoints=False,
        **kwargs,
    )


def plot_segmentation_masks(
    image: Optional[str | Path | np.ndarray],
    detections_df: pd.DataFrame,
    draw_boxes: bool = True,
    **kwargs: Any,
) -> None:
    """Plot segmentation polygons and optional boxes/labels."""
    plot_image_detections(
        image,
        detections_df,
        draw_boxes=draw_boxes,
        draw_polygons=True,
        draw_keypoints=False,
        **kwargs,
    )


def plot_pose_keypoints(
    image: Optional[str | Path | np.ndarray],
    detections_df: pd.DataFrame,
    draw_boxes: bool = True,
    **kwargs: Any,
) -> None:
    """Plot pose keypoints with the COCO skeleton by default."""
    plot_image_detections(
        image,
        detections_df,
        draw_boxes=draw_boxes,
        draw_polygons=False,
        draw_keypoints=True,
        **kwargs,
    )


def plot_training_metrics(
    metrics_df: pd.DataFrame,
    columns: Optional[list[str]] = None,
    title: str = "Training Metrics",
    figsize: tuple[float, float] = (12, 6),
    save_to: Optional[str | Path] = None,
    show: bool = True,
    dpi: int = 150,
) -> None:
    """Plot selected training metrics over epochs."""
    if columns is None:
        numeric = metrics_df.select_dtypes(include="number").columns.tolist()
        columns = [c for c in numeric if "epoch" not in c.lower()]

    fig, axes = plt.subplots(1, max(len(columns), 1), figsize=figsize, squeeze=False)
    if not columns:
        _empty_axes_message(axes[0, 0], "No numeric metrics")
    else:
        for ax, col in zip(axes[0], columns):
            if col in metrics_df.columns:
                ax.plot(metrics_df.index, metrics_df[col])
                ax.set_title(col)
                ax.set_xlabel("Epoch")
                ax.grid(True, alpha=0.3)
            else:
                _empty_axes_message(ax, f"Missing column: {col}")

    fig.suptitle(title)
    fig.tight_layout()
    _finish_plot(fig, save_to=save_to, show=show, dpi=dpi)


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str],
    title: str = "Confusion Matrix",
    cmap: str = "Blues",
    figsize: tuple[float, float] = (8, 6),
    save_to: Optional[str | Path] = None,
    show: bool = True,
    dpi: int = 150,
) -> None:
    """Plot a confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    fig.colorbar(im, ax=ax)
    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    ax.set_title(title)

    thresh = cm.max() / 2.0 if cm.size else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                f"{cm[i, j]}",
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    _finish_plot(fig, save_to=save_to, show=show, dpi=dpi)


def plot_precision_recall(
    precision: Sequence[float],
    recall: Sequence[float],
    title: str = "Precision-Recall Curve",
    figsize: tuple[float, float] = (7, 5),
    save_to: Optional[str | Path] = None,
    show: bool = True,
    dpi: int = 150,
) -> None:
    """Plot a precision-recall curve."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(recall, precision, marker=".", linewidth=1.5)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _finish_plot(fig, save_to=save_to, show=show, dpi=dpi)


def plot_gpu_usage(
    memory_df: pd.DataFrame,
    figsize: tuple[float, float] = (10, 4),
    save_to: Optional[str | Path] = None,
    show: bool = True,
    dpi: int = 150,
) -> None:
    """Plot GPU/CPU memory over time."""
    fig, ax = plt.subplots(figsize=figsize)
    if "vram_used_mb" in memory_df.columns and memory_df["vram_used_mb"].notna().any():
        ax.plot(memory_df["timestamp"], memory_df["vram_used_mb"], label="VRAM used (MB)")
    if "ram_used_mb" in memory_df.columns:
        ax.plot(
            memory_df["timestamp"],
            memory_df["ram_used_mb"],
            label="RAM used (MB)",
            linestyle="--",
        )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Memory (MB)")
    ax.set_title("Memory Usage Over Time")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _finish_plot(fig, save_to=save_to, show=show, dpi=dpi)


def plot_tracking_trajectories(
    tracks_df: pd.DataFrame,
    image_width: int = 640,
    image_height: int = 640,
    max_tracks: int = 50,
    title: str = "Tracking Trajectories",
    figsize: tuple[float, float] = (8, 8),
    save_to: Optional[str | Path] = None,
    show: bool = True,
    dpi: int = 150,
) -> None:
    """Plot spatial trajectories of tracked objects."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, image_width)
    ax.set_ylim(image_height, 0)

    if tracks_df.empty or "track_id" not in tracks_df.columns:
        _empty_axes_message(ax, "No tracks")
    else:
        track_ids = tracks_df["track_id"].dropna().unique()
        if len(track_ids) > max_tracks:
            track_ids = track_ids[:max_tracks]

        for i, tid in enumerate(track_ids):
            sub = tracks_df[tracks_df["track_id"] == tid].sort_values("frame")
            if sub.empty:
                continue
            x = (sub["xmin"] + sub["xmax"]) / 2
            y = (sub["ymin"] + sub["ymax"]) / 2
            color = _plot_color(i)
            ax.plot(x, y, color=color, linewidth=1.0, alpha=0.8)
            ax.scatter(x.iloc[-1], y.iloc[-1], color=color, s=20, zorder=5)

        ax.set_title(title)
        ax.set_xlabel("x (px)")
        ax.set_ylabel("y (px)")

    fig.tight_layout()
    _finish_plot(fig, save_to=save_to, show=show, dpi=dpi)


def plot_class_distribution(
    df: pd.DataFrame,
    column: str = "class_name",
    title: str = "Class Distribution",
    figsize: tuple[float, float] = (8, 5),
    save_to: Optional[str | Path] = None,
    show: bool = True,
    dpi: int = 150,
) -> None:
    """Plot class counts as a horizontal bar chart."""
    fig, ax = plt.subplots(figsize=figsize)
    if df.empty or column not in df.columns:
        _empty_axes_message(ax, f"No data in column: {column}")
    else:
        counts = df[column].value_counts()
        ax.barh(counts.index.astype(str), counts.values)
        ax.set_title(title)
        ax.set_xlabel("Count")
        ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    _finish_plot(fig, save_to=save_to, show=show, dpi=dpi)


def plot_video_statistics(
    df: pd.DataFrame,
    figsize: tuple[float, float] = (12, 8),
    save_to: Optional[str | Path] = None,
    show: bool = True,
    dpi: int = 150,
) -> None:
    """Plot a summary dashboard for video prediction results."""
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    if df.empty:
        for ax in axes.ravel():
            _empty_axes_message(ax, "No detections")
        fig.tight_layout()
        _finish_plot(fig, save_to=save_to, show=show, dpi=dpi)
        return

    det_per_frame = df.groupby("frame").size() if "frame" in df.columns else pd.Series(dtype=int)
    axes[0, 0].plot(det_per_frame.index, det_per_frame.values)
    axes[0, 0].set_title("Detections per Frame")
    axes[0, 0].set_xlabel("Frame")
    axes[0, 0].set_ylabel("Count")
    axes[0, 0].grid(True, alpha=0.3)

    if "confidence" in df.columns:
        axes[0, 1].hist(df["confidence"].dropna(), bins=30, edgecolor="black")
    axes[0, 1].set_title("Confidence Distribution")
    axes[0, 1].set_xlabel("Confidence")
    axes[0, 1].set_ylabel("Frequency")
    axes[0, 1].grid(True, alpha=0.3)

    if "class_name" in df.columns:
        counts = df["class_name"].value_counts()
        axes[1, 0].barh(counts.index.astype(str), counts.values)
    axes[1, 0].set_title("Class Distribution")
    axes[1, 0].set_xlabel("Count")
    axes[1, 0].grid(True, axis="x", alpha=0.3)

    required_bbox_cols = {"xmin", "ymin", "xmax", "ymax"}
    if required_bbox_cols.issubset(df.columns):
        bbox_area = (df["xmax"] - df["xmin"]) * (df["ymax"] - df["ymin"])
        axes[1, 1].hist(bbox_area.dropna(), bins=30, edgecolor="black")
    axes[1, 1].set_title("BBox Area Distribution")
    axes[1, 1].set_xlabel("Area (px^2)")
    axes[1, 1].set_ylabel("Frequency")
    axes[1, 1].grid(True, alpha=0.3)

    fig.tight_layout()
    _finish_plot(fig, save_to=save_to, show=show, dpi=dpi)
