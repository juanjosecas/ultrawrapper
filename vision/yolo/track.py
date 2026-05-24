"""Tracking support – ByteTrack, BoT-SORT, DataFrame output."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from vision.yolo.dataframe import concat_results, ultralytics_result_to_dataframe
from vision.yolo.devices import detect_device
from vision.yolo.metrics import compute_iou
from vision.yolo.utils import load_model

DEFAULT_TRACKER_CONFIGS: dict[str, dict[str, Any]] = {
    "bytetrack": {
        "tracker_type": "bytetrack",
        "track_high_thresh": 0.25,
        "track_low_thresh": 0.1,
        "new_track_thresh": 0.25,
        "track_buffer": 30,
        "match_thresh": 0.8,
        "fuse_score": True,
    },
    "botsort": {
        "tracker_type": "botsort",
        "track_high_thresh": 0.25,
        "track_low_thresh": 0.1,
        "new_track_thresh": 0.25,
        "track_buffer": 30,
        "match_thresh": 0.8,
        "fuse_score": True,
        "gmc_method": "sparseOptFlow",
        "proximity_thresh": 0.5,
        "appearance_thresh": 0.25,
        "with_reid": False,
    },
}


def make_tracker_config(
    base_tracker: str | Path = "bytetrack.yaml",
    overrides: Optional[dict[str, Any]] = None,
    save_to: Optional[str | Path] = None,
) -> Path:
    """Create a tracker YAML file from a base tracker plus overrides.

    ``base_tracker`` can be ``'bytetrack.yaml'``, ``'botsort.yaml'`` or a path
    to an existing YAML file. ``overrides`` updates the loaded configuration.
    """
    import yaml

    config = _load_tracker_config(base_tracker)
    if overrides:
        config = _deep_update(config, overrides)

    if save_to is None:
        tracker_name = str(base_tracker).replace("\\", "/").split("/")[-1]
        save_to = Path.cwd() / f"custom_{tracker_name}"
    output_path = Path(save_to)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as fh:
        yaml.safe_dump(config, fh, sort_keys=False)
    return output_path


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


def _load_tracker_config(base_tracker: str | Path) -> dict[str, Any]:
    import yaml

    path = Path(base_tracker)
    if path.exists():
        with open(path) as fh:
            data = yaml.safe_load(fh) or {}
        return dict(data)

    tracker_key = path.stem.lower()
    if tracker_key in DEFAULT_TRACKER_CONFIGS:
        return dict(DEFAULT_TRACKER_CONFIGS[tracker_key])

    ultralytics_path = _find_ultralytics_tracker_yaml(path.name)
    if ultralytics_path is not None:
        with open(ultralytics_path) as fh:
            data = yaml.safe_load(fh) or {}
        return dict(data)

    raise FileNotFoundError(
        f"Could not find tracker config {base_tracker!r}. "
        "Use 'bytetrack.yaml', 'botsort.yaml', or an existing YAML path."
    )


def _find_ultralytics_tracker_yaml(filename: str) -> Optional[Path]:
    try:
        import ultralytics
    except ImportError:
        return None

    root = Path(ultralytics.__file__).parent
    for candidate in root.rglob(filename):
        return candidate
    return None


def _deep_update(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    updated = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(updated.get(key), dict):
            updated[key] = _deep_update(updated[key], value)
        else:
            updated[key] = value
    return updated


def track_detections_dataframe(
    detections_df: pd.DataFrame,
    iou_threshold: float = 0.3,
    max_frame_gap: int = 1,
    same_class_only: bool = True,
    confidence_threshold: Optional[float] = None,
    track_id_start: int = 1,
    overwrite_track_id: bool = True,
    save_to: Optional[str | Path] = None,
    save_fmt: str = "parquet",
) -> pd.DataFrame:
    """Assign track IDs to an existing video-detection DataFrame.

    This is a lightweight post-processing tracker for detections that were
    already generated with ``predict_video``. It links boxes between frames by
    greedy IoU matching, optionally constrained to the same class.

    It does not rerun YOLO and is not ByteTrack/BOT-SORT; use ``track_video``
    when you need the model-backed Ultralytics trackers.
    """
    required = {"frame", "xmin", "ymin", "xmax", "ymax"}
    missing = required.difference(detections_df.columns)
    if missing:
        raise ValueError(f"detections_df is missing required columns: {sorted(missing)}")

    df = detections_df.copy()
    if df.empty:
        if "track_id" not in df.columns:
            df["track_id"] = pd.Series(dtype="Int64")
        return df

    if confidence_threshold is not None and "confidence" in df.columns:
        confidence = pd.to_numeric(df["confidence"], errors="coerce")
        df = df[confidence >= confidence_threshold].copy()

    if not overwrite_track_id and "track_id" in df.columns and df["track_id"].notna().any():
        return df

    original_order = "__original_order"
    df[original_order] = np.arange(len(df))
    sort_cols = ["frame"]
    ascending = [True]
    if "confidence" in df.columns:
        sort_cols.append("confidence")
        ascending.append(False)
    df = df.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)
    df["track_id"] = pd.NA

    active_tracks: dict[int, dict[str, Any]] = {}
    next_track_id = track_id_start

    for frame in sorted(df["frame"].dropna().unique()):
        frame_mask = df["frame"] == frame
        frame_indices = df.index[frame_mask].tolist()

        candidate_pairs: list[tuple[float, int, int]] = []
        for row_index in frame_indices:
            row = df.loc[row_index]
            row_box = _row_box(row)
            for track_id, track in active_tracks.items():
                frame_gap = int(frame) - int(track["frame"])
                if frame_gap <= 0 or frame_gap > max_frame_gap:
                    continue
                if same_class_only and not _same_class(row, track["row"]):
                    continue
                iou = compute_iou(row_box, track["box"])
                if iou >= iou_threshold:
                    candidate_pairs.append((iou, row_index, track_id))

        used_rows: set[int] = set()
        used_tracks: set[int] = set()
        for _, row_index, track_id in sorted(candidate_pairs, reverse=True):
            if row_index in used_rows or track_id in used_tracks:
                continue
            df.at[row_index, "track_id"] = track_id
            used_rows.add(row_index)
            used_tracks.add(track_id)

        for row_index in frame_indices:
            if pd.isna(df.at[row_index, "track_id"]):
                df.at[row_index, "track_id"] = next_track_id
                next_track_id += 1

            track_id = int(df.at[row_index, "track_id"])
            row = df.loc[row_index]
            active_tracks[track_id] = {
                "frame": int(frame),
                "box": _row_box(row),
                "row": row,
            }

        stale_ids = [
            track_id
            for track_id, track in active_tracks.items()
            if int(frame) - int(track["frame"]) > max_frame_gap
        ]
        for track_id in stale_ids:
            active_tracks.pop(track_id, None)

    df["track_id"] = df["track_id"].astype("Int64")
    df = df.sort_values(original_order).drop(columns=[original_order]).reset_index(drop=True)

    if save_to:
        from vision.yolo.serialization import save_dataframe

        save_dataframe(df, save_to, fmt=save_fmt)

    return df


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


def _row_box(row: pd.Series) -> list[float]:
    return [float(row["xmin"]), float(row["ymin"]), float(row["xmax"]), float(row["ymax"])]


def _same_class(row: pd.Series, previous_row: pd.Series) -> bool:
    if "class_id" in row and "class_id" in previous_row:
        if pd.notna(row["class_id"]) and pd.notna(previous_row["class_id"]):
            return int(row["class_id"]) == int(previous_row["class_id"])
    if "class_name" in row and "class_name" in previous_row:
        if pd.notna(row["class_name"]) and pd.notna(previous_row["class_name"]):
            return str(row["class_name"]) == str(previous_row["class_name"])
    return True


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
