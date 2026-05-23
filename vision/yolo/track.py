"""Tracking support – ByteTrack, BoT-SORT, DataFrame output."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from vision.yolo.dataframe import ultralytics_result_to_dataframe, concat_results
from vision.yolo.devices import detect_device
from vision.yolo.utils import load_model


def track_video(
    model_path: str,
    video_path: str | Path,
    tracker: str = "bytetrack.yaml",
    confidence: float = 0.25,
    iou: float = 0.45,
    device: Optional[str] = None,
    batch_size: int = 8,
    save_to: Optional[str | Path] = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Run multi-object tracking on a video using ByteTrack or BoT-SORT.

    Parameters
    ----------
    tracker:
        Tracker config file – ``'bytetrack.yaml'`` or ``'botsort.yaml'``.
    save_to:
        Optional path for incremental parquet output.

    Returns
    -------
    pd.DataFrame with tracking columns (track_id, frame, …).
    """
    import cv2

    device = device or detect_device()
    model = load_model(model_path)

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    all_dfs: list[pd.DataFrame] = []
    frame_idx = 0
    batch_frames: list = []
    batch_indices: list[int] = []
    batch_ts: list[float] = []

    def _process_batch() -> None:
        results = model.track(
            batch_frames,
            tracker=tracker,
            conf=confidence,
            iou=iou,
            device=device,
            persist=True,
            verbose=False,
            **kwargs,
        )
        chunk_dfs: list[pd.DataFrame] = []
        for result, idx, ts in zip(results, batch_indices, batch_ts):
            df = ultralytics_result_to_dataframe(result, frame_idx=idx, timestamp=ts)
            chunk_dfs.append(df)
        chunk = concat_results(chunk_dfs)
        all_dfs.append(chunk)

        if save_to:
            from vision.yolo.serialization import append_to_parquet

            append_to_parquet(chunk, save_to)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            batch_frames.append(frame)
            batch_indices.append(frame_idx)
            batch_ts.append(frame_idx / fps)
            frame_idx += 1

            if len(batch_frames) == batch_size:
                _process_batch()
                batch_frames, batch_indices, batch_ts = [], [], []

        if batch_frames:
            _process_batch()
    finally:
        cap.release()

    return concat_results(all_dfs)


def build_tracks_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Enrich a raw tracking DataFrame with computed kinematic fields.

    Adds: ``x_center``, ``y_center``, ``velocity``, ``acceleration``,
    ``trajectory_length``, ``time_alive``.
    """
    df = df.copy()
    df["x_center"] = (df["xmin"] + df["xmax"]) / 2
    df["y_center"] = (df["ymin"] + df["ymax"]) / 2

    df = df.sort_values(["track_id", "frame"]).reset_index(drop=True)

    # Per-track kinematics using pandas groupby + diff
    df["dx"] = df.groupby("track_id")["x_center"].diff()
    df["dy"] = df.groupby("track_id")["y_center"].diff()
    df["dt"] = df.groupby("track_id")["timestamp"].diff().fillna(1 / 30)
    df["velocity"] = np.sqrt(df["dx"] ** 2 + df["dy"] ** 2) / df["dt"].replace(0, np.nan)
    df["acceleration"] = df.groupby("track_id")["velocity"].diff() / df["dt"].replace(0, np.nan)

    # Cumulative distance per track
    df["step_dist"] = np.sqrt(df["dx"].fillna(0) ** 2 + df["dy"].fillna(0) ** 2)
    df["trajectory_length"] = df.groupby("track_id")["step_dist"].cumsum()

    # Time alive
    df["time_alive"] = df.groupby("track_id")["timestamp"].transform(
        lambda s: s - s.min()
    )

    df.drop(columns=["dx", "dy", "dt", "step_dist"], inplace=True)
    return df


def compute_track_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Return per-track summary statistics."""
    df = build_tracks_dataframe(df)
    stats = (
        df.groupby("track_id")
        .agg(
            class_name=("class_name", "first"),
            frames=("frame", "count"),
            max_confidence=("confidence", "max"),
            mean_confidence=("confidence", "mean"),
            max_velocity=("velocity", "max"),
            mean_velocity=("velocity", "mean"),
            trajectory_length=("trajectory_length", "max"),
            time_alive=("time_alive", "max"),
        )
        .reset_index()
    )
    return stats


def smooth_tracks(
    df: pd.DataFrame,
    window: int = 5,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Apply a rolling mean to spatial columns within each track."""
    df = df.copy()
    cols = columns or ["x_center", "y_center", "xmin", "ymin", "xmax", "ymax"]
    existing = [c for c in cols if c in df.columns]
    for col in existing:
        df[col] = df.groupby("track_id")[col].transform(
            lambda s: s.rolling(window, min_periods=1, center=True).mean()
        )
    return df


def filter_short_tracks(df: pd.DataFrame, min_frames: int = 5) -> pd.DataFrame:
    """Remove tracks that appear in fewer than ``min_frames`` frames."""
    counts = df.groupby("track_id")["frame"].count()
    valid = counts[counts >= min_frames].index
    return df[df["track_id"].isin(valid)].reset_index(drop=True)
