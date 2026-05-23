"""Matplotlib-based scientific plotting functions.

No default Ultralytics plots are used unless explicitly requested.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd


matplotlib.use("Agg")  # non-interactive backend; switch to 'TkAgg' for display


# ---------------------------------------------------------------------------
# Training metrics
# ---------------------------------------------------------------------------

def plot_training_metrics(
    metrics_df: pd.DataFrame,
    columns: Optional[list[str]] = None,
    title: str = "Training Metrics",
    figsize: tuple = (12, 6),
    save_to: Optional[str | Path] = None,
) -> "plt.Figure":
    """Line plot of training metrics over epochs."""
    if columns is None:
        # Auto-detect numeric columns (drop index / epoch)
        numeric = metrics_df.select_dtypes(include="number").columns.tolist()
        columns = [c for c in numeric if "epoch" not in c.lower()]

    fig, axes = plt.subplots(1, len(columns), figsize=figsize, squeeze=False)
    for ax, col in zip(axes[0], columns):
        if col in metrics_df.columns:
            ax.plot(metrics_df.index, metrics_df[col])
            ax.set_title(col)
            ax.set_xlabel("Epoch")
            ax.grid(True, alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()
    if save_to:
        fig.savefig(save_to, dpi=150)
    return fig


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    cm: "np.ndarray",  # type: ignore[name-defined]
    class_names: list[str],
    title: str = "Confusion Matrix",
    cmap: str = "Blues",
    figsize: tuple = (8, 6),
    save_to: Optional[str | Path] = None,
) -> "plt.Figure":
    """Heatmap of a confusion matrix."""
    import numpy as np

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

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, f"{cm[i, j]}",
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    if save_to:
        fig.savefig(save_to, dpi=150)
    return fig


# ---------------------------------------------------------------------------
# Precision–Recall
# ---------------------------------------------------------------------------

def plot_precision_recall(
    precision: Sequence[float],
    recall: Sequence[float],
    title: str = "Precision–Recall Curve",
    figsize: tuple = (7, 5),
    save_to: Optional[str | Path] = None,
) -> "plt.Figure":
    """PR curve."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(recall, precision, marker=".", linewidth=1.5)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save_to:
        fig.savefig(save_to, dpi=150)
    return fig


# ---------------------------------------------------------------------------
# GPU usage
# ---------------------------------------------------------------------------

def plot_gpu_usage(
    memory_df: pd.DataFrame,
    figsize: tuple = (10, 4),
    save_to: Optional[str | Path] = None,
) -> "plt.Figure":
    """Line plot of GPU/CPU memory over time."""
    fig, ax = plt.subplots(figsize=figsize)
    if "vram_used_mb" in memory_df.columns and memory_df["vram_used_mb"].notna().any():
        ax.plot(memory_df["timestamp"], memory_df["vram_used_mb"], label="VRAM used (MB)")
    if "ram_used_mb" in memory_df.columns:
        ax.plot(memory_df["timestamp"], memory_df["ram_used_mb"], label="RAM used (MB)", linestyle="--")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Memory (MB)")
    ax.set_title("Memory Usage Over Time")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save_to:
        fig.savefig(save_to, dpi=150)
    return fig


# ---------------------------------------------------------------------------
# Tracking trajectories
# ---------------------------------------------------------------------------

def plot_tracking_trajectories(
    tracks_df: pd.DataFrame,
    image_width: int = 640,
    image_height: int = 640,
    max_tracks: int = 50,
    title: str = "Tracking Trajectories",
    figsize: tuple = (8, 8),
    save_to: Optional[str | Path] = None,
) -> "plt.Figure":
    """Plot spatial trajectories of tracked objects."""
    import numpy as np

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, image_width)
    ax.set_ylim(image_height, 0)  # image coordinates: y=0 at top

    track_ids = tracks_df["track_id"].dropna().unique()
    if len(track_ids) > max_tracks:
        track_ids = track_ids[:max_tracks]

    cmap = plt.get_cmap("tab20")
    for i, tid in enumerate(track_ids):
        sub = tracks_df[tracks_df["track_id"] == tid].sort_values("frame")
        x = (sub["xmin"] + sub["xmax"]) / 2
        y = (sub["ymin"] + sub["ymax"]) / 2
        color = cmap(i % 20)
        ax.plot(x, y, color=color, linewidth=1.0, alpha=0.8)
        ax.scatter(x.iloc[-1], y.iloc[-1], color=color, s=20, zorder=5)

    ax.set_title(title)
    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px)")
    fig.tight_layout()
    if save_to:
        fig.savefig(save_to, dpi=150)
    return fig


# ---------------------------------------------------------------------------
# Class distribution
# ---------------------------------------------------------------------------

def plot_class_distribution(
    df: pd.DataFrame,
    column: str = "class_name",
    title: str = "Class Distribution",
    figsize: tuple = (8, 5),
    save_to: Optional[str | Path] = None,
) -> "plt.Figure":
    """Horizontal bar chart of class counts."""
    counts = df[column].value_counts().sort_values()
    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(counts.index, counts.values)
    ax.set_xlabel("Count")
    ax.set_title(title)
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    if save_to:
        fig.savefig(save_to, dpi=150)
    return fig


# ---------------------------------------------------------------------------
# Video statistics
# ---------------------------------------------------------------------------

def plot_video_statistics(
    df: pd.DataFrame,
    figsize: tuple = (12, 8),
    save_to: Optional[str | Path] = None,
) -> "plt.Figure":
    """Summary dashboard for video prediction results."""
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # 1. Detections per frame
    det_per_frame = df.groupby("frame").size()
    axes[0, 0].plot(det_per_frame.index, det_per_frame.values)
    axes[0, 0].set_title("Detections per Frame")
    axes[0, 0].set_xlabel("Frame")
    axes[0, 0].set_ylabel("Count")
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Confidence distribution
    axes[0, 1].hist(df["confidence"], bins=30, edgecolor="black")
    axes[0, 1].set_title("Confidence Distribution")
    axes[0, 1].set_xlabel("Confidence")
    axes[0, 1].set_ylabel("Frequency")
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Class distribution
    counts = df["class_name"].value_counts()
    axes[1, 0].barh(counts.index, counts.values)
    axes[1, 0].set_title("Class Distribution")
    axes[1, 0].set_xlabel("Count")
    axes[1, 0].grid(True, axis="x", alpha=0.3)

    # 4. Bounding box area distribution
    df = df.copy()
    df["bbox_area"] = (df["xmax"] - df["xmin"]) * (df["ymax"] - df["ymin"])
    axes[1, 1].hist(df["bbox_area"], bins=30, edgecolor="black")
    axes[1, 1].set_title("BBox Area Distribution")
    axes[1, 1].set_xlabel("Area (px²)")
    axes[1, 1].set_ylabel("Frequency")
    axes[1, 1].grid(True, alpha=0.3)

    fig.tight_layout()
    if save_to:
        fig.savefig(save_to, dpi=150)
    return fig
