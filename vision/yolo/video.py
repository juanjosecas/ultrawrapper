"""Streaming video frame generator and prediction writer.

Videos are NEVER loaded fully into memory – frames are yielded one at a time
or in batches.
"""

from __future__ import annotations

from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Generator, Optional

import numpy as np
import pandas as pd


def video_frame_generator(
    video_path: str | Path,
    batch_size: int = 1,
    max_frames: Optional[int] = None,
    start_frame: int = 0,
) -> Generator[tuple[list, list[int], list[float]], None, None]:
    """Yield (frames, frame_indices, timestamps) batches from a video.

    Frames are BGR numpy arrays (as returned by OpenCV).

    Parameters
    ----------
    batch_size:
        Number of frames per yielded batch.
    max_frames:
        Stop after this many frames (``None`` = entire video).
    start_frame:
        Skip this many frames at the beginning.
    """
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # Seek to start
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frame_idx = start_frame
    total = 0
    batch_frames: list = []
    batch_indices: list[int] = []
    batch_ts: list[float] = []

    try:
        while True:
            if max_frames is not None and total >= max_frames:
                break

            ret, frame = cap.read()
            if not ret:
                break

            batch_frames.append(frame)
            batch_indices.append(frame_idx)
            batch_ts.append(frame_idx / fps)
            frame_idx += 1
            total += 1

            if len(batch_frames) == batch_size:
                yield batch_frames, batch_indices, batch_ts
                batch_frames, batch_indices, batch_ts = [], [], []

        if batch_frames:
            yield batch_frames, batch_indices, batch_ts
    finally:
        cap.release()


def write_video_predictions(
    model_path: str,
    video_path: str | Path,
    output_path: str | Path,
    tracker: Optional[str] = None,
    confidence: float = 0.25,
    iou: float = 0.45,
    batch_size: int = 8,
    fmt: str = "parquet",
    device: Optional[str] = None,
) -> Path:
    """Run inference (or tracking) on a video and write predictions incrementally.

    Parameters
    ----------
    tracker:
        Pass a tracker config (e.g. ``'bytetrack.yaml'``) to enable tracking.
        When ``None``, plain detection is used.
    fmt:
        Output format: ``'parquet'``, ``'feather'``, or ``'csv'``.

    Returns
    -------
    Path to the saved prediction file.
    """
    from vision.yolo.dataframe import concat_results, ultralytics_result_to_dataframe
    from vision.yolo.devices import detect_device
    from vision.yolo.serialization import append_to_parquet, save_dataframe
    from vision.yolo.utils import load_model

    device = device or detect_device()
    model = load_model(model_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Remove existing file so we start fresh
    if output_path.exists():
        output_path.unlink()

    # For non-parquet formats accumulate all chunks and write once at the end
    # to avoid repeated full-file reads.
    non_parquet_chunks: list[pd.DataFrame] = []

    for frames, indices, timestamps in video_frame_generator(video_path, batch_size=batch_size):
        if tracker:
            results = model.track(
                frames,
                tracker=tracker,
                conf=confidence,
                iou=iou,
                device=device,
                persist=True,
                verbose=False,
            )
        else:
            results = model.predict(
                frames, conf=confidence, iou=iou, device=device, verbose=False
            )

        chunk_dfs = [
            ultralytics_result_to_dataframe(r, frame_idx=idx, timestamp=ts)
            for r, idx, ts in zip(results, indices, timestamps)
        ]
        chunk = concat_results(chunk_dfs)

        if fmt == "parquet":
            append_to_parquet(chunk, output_path)
        else:
            non_parquet_chunks.append(chunk)

    if fmt != "parquet" and non_parquet_chunks:
        save_dataframe(concat_results(non_parquet_chunks), output_path, fmt=fmt)

    return output_path


def draw_predictions_on_frame(
    frame: "np.ndarray",
    detections_df: pd.DataFrame,
    frame_idx: Optional[int] = None,
    color_by: str = "class",
    confidence_range: tuple[float, float] = (0.0, 1.0),
    draw_boxes: bool = True,
    draw_labels: bool = True,
    draw_polygons: bool = True,
    draw_keypoints: bool = True,
    draw_tails: bool = False,
    tail_history: Optional[dict[Any, deque[tuple[int, int]]]] = None,
    tail_length: int = 30,
    keypoint_threshold: float = 0.25,
    box_thickness: int = 2,
    label_scale: float = 0.5,
    label_thickness: int = 1,
) -> "np.ndarray":
    """Draw DataFrame predictions on one BGR frame.

    Parameters
    ----------
    color_by:
        ``'class'``, ``'track_id'`` or ``'confidence'``.
    draw_tails:
        Draw trailing paths for tracked objects. Requires ``track_id`` values.
    tail_history:
        Mutable history reused across video frames.
    """
    import cv2

    annotated = frame.copy()
    data = detections_df
    if frame_idx is not None and "frame" in data.columns:
        data = data[data["frame"] == frame_idx]

    if data.empty:
        return annotated

    if tail_history is None:
        tail_history = defaultdict(lambda: deque(maxlen=tail_length))

    for row_idx, row in data.reset_index(drop=True).iterrows():
        color = _prediction_color(
            row,
            row_idx=row_idx,
            color_by=color_by,
            confidence_range=confidence_range,
        )

        if draw_polygons and _has_points(row.get("polygon")):
            pts = np.asarray(row["polygon"], dtype=np.int32)
            if pts.ndim == 2 and pts.shape[1] >= 2:
                cv2.polylines(
                    annotated,
                    [pts[:, :2]],
                    isClosed=True,
                    color=color,
                    thickness=box_thickness,
                )
                overlay = annotated.copy()
                cv2.fillPoly(overlay, [pts[:, :2]], color=color)
                annotated = cv2.addWeighted(overlay, 0.25, annotated, 0.75, 0)

        if draw_boxes and _has_bbox(row):
            x1, y1, x2, y2 = _bbox_int(row)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, box_thickness)

        if draw_keypoints and _has_points(row.get("keypoints")):
            _draw_keypoints_cv2(
                annotated,
                row["keypoints"],
                color=color,
                keypoint_threshold=keypoint_threshold,
                thickness=max(1, box_thickness),
            )

        if draw_tails and "track_id" in row and pd.notna(row["track_id"]) and _has_bbox(row):
            track_id = int(row["track_id"])
            x1, y1, x2, y2 = _bbox_int(row)
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            tail_history[track_id].append(center)
            points = list(tail_history[track_id])
            for start, end in zip(points[:-1], points[1:]):
                cv2.line(annotated, start, end, color, max(1, box_thickness))

        if draw_labels and _has_bbox(row):
            _draw_label_cv2(
                annotated,
                row,
                color=color,
                label_scale=label_scale,
                label_thickness=label_thickness,
            )

    return annotated


def write_annotated_video(
    model_path: Optional[str],
    video_path: str | Path,
    output_path: str | Path,
    predictions_df: Optional[pd.DataFrame] = None,
    tracker: Optional[str] = None,
    confidence: float = 0.25,
    iou: float = 0.45,
    device: Optional[str] = None,
    batch_size: int = 8,
    max_frames: Optional[int] = None,
    start_frame: int = 0,
    color_by: str = "class",
    draw_boxes: bool = True,
    draw_labels: bool = True,
    draw_polygons: bool = True,
    draw_keypoints: bool = True,
    draw_tails: bool = False,
    tail_length: int = 30,
    keypoint_threshold: float = 0.25,
    box_thickness: int = 2,
    label_scale: float = 0.5,
    label_thickness: int = 1,
    save_predictions_to: Optional[str | Path] = None,
    save_predictions_fmt: str = "parquet",
    return_predictions: bool = False,
    codec: str = "mp4v",
    **kwargs: Any,
) -> Path | tuple[Path, pd.DataFrame]:
    """Load a video, overlay YOLO predictions, and write an annotated video.

    Pass ``predictions_df`` to draw existing predictions, or pass ``model_path``
    to run prediction/tracking while streaming the video.
    """
    import cv2

    from vision.yolo.dataframe import concat_results, ultralytics_result_to_dataframe
    from vision.yolo.devices import detect_device
    from vision.yolo.serialization import save_dataframe
    from vision.yolo.utils import load_model

    if predictions_df is None and not model_path:
        raise ValueError("model_path is required when predictions_df is not provided")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*codec),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        raise ValueError(f"Could not open output video for writing: {output_path}")

    model = None
    if predictions_df is None:
        device = device or detect_device()
        model = load_model(str(model_path))

    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    all_dfs: list[pd.DataFrame] = []
    tail_history: dict[Any, deque[tuple[int, int]]] = defaultdict(
        lambda: deque(maxlen=tail_length)
    )
    frame_idx = start_frame
    total = 0
    batch_frames: list[np.ndarray] = []
    batch_indices: list[int] = []
    batch_timestamps: list[float] = []

    def _draw_and_write(frames: list[np.ndarray], frame_dfs: list[pd.DataFrame]) -> None:
        for frame, frame_df in zip(frames, frame_dfs):
            annotated = draw_predictions_on_frame(
                frame,
                frame_df,
                color_by=color_by,
                draw_boxes=draw_boxes,
                draw_labels=draw_labels,
                draw_polygons=draw_polygons,
                draw_keypoints=draw_keypoints,
                draw_tails=draw_tails,
                tail_history=tail_history,
                tail_length=tail_length,
                keypoint_threshold=keypoint_threshold,
                box_thickness=box_thickness,
                label_scale=label_scale,
                label_thickness=label_thickness,
            )
            writer.write(annotated)

    def _process_model_batch() -> None:
        if not batch_frames or model is None:
            return
        if tracker:
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
        else:
            results = model.predict(
                batch_frames,
                conf=confidence,
                iou=iou,
                device=device,
                verbose=False,
                **kwargs,
            )
        frame_dfs = [
            ultralytics_result_to_dataframe(result, frame_idx=idx, timestamp=ts)
            for result, idx, ts in zip(results, batch_indices, batch_timestamps)
        ]
        all_dfs.extend(frame_dfs)
        _draw_and_write(batch_frames, frame_dfs)

    try:
        if predictions_df is not None:
            while True:
                if max_frames is not None and total >= max_frames:
                    break
                ret, frame = cap.read()
                if not ret:
                    break
                frame_df = predictions_df[predictions_df["frame"] == frame_idx]
                _draw_and_write([frame], [frame_df])
                frame_idx += 1
                total += 1
            all_dfs.append(predictions_df)
        else:
            while True:
                if max_frames is not None and total >= max_frames:
                    break
                ret, frame = cap.read()
                if not ret:
                    break
                batch_frames.append(frame)
                batch_indices.append(frame_idx)
                batch_timestamps.append(frame_idx / fps)
                frame_idx += 1
                total += 1

                if len(batch_frames) == batch_size:
                    _process_model_batch()
                    batch_frames, batch_indices, batch_timestamps = [], [], []

            if batch_frames:
                _process_model_batch()
    finally:
        cap.release()
        writer.release()

    predictions = concat_results(all_dfs)
    if save_predictions_to:
        save_dataframe(predictions, save_predictions_to, fmt=save_predictions_fmt)

    if return_predictions:
        return output_path, predictions
    return output_path


def write_annotated_video_from_dataframe(
    video_path: str | Path,
    predictions: pd.DataFrame | str | Path,
    output_path: str | Path,
    max_frames: Optional[int] = None,
    start_frame: int = 0,
    color_by: str = "class",
    draw_boxes: bool = True,
    draw_labels: bool = True,
    draw_polygons: bool = True,
    draw_keypoints: bool = True,
    draw_tails: bool = False,
    tail_length: int = 30,
    keypoint_threshold: float = 0.25,
    box_thickness: int = 2,
    label_scale: float = 0.5,
    label_thickness: int = 1,
    codec: str = "mp4v",
    return_predictions: bool = False,
) -> Path | tuple[Path, pd.DataFrame]:
    """Annotate a video using predictions already stored in a DataFrame.

    ``predictions`` can be the DataFrame returned by ``predict_video`` or a
    parquet/feather/csv path saved by the prediction step. Parquet or feather
    are preferred when the DataFrame contains list columns such as polygons or
    keypoints.
    """
    from vision.yolo.serialization import load_dataframe

    if isinstance(predictions, pd.DataFrame):
        predictions_df = predictions
    else:
        predictions_df = load_dataframe(predictions)

    if "frame" not in predictions_df.columns:
        raise ValueError("predictions must contain a 'frame' column")

    return write_annotated_video(
        model_path=None,
        video_path=video_path,
        output_path=output_path,
        predictions_df=predictions_df,
        max_frames=max_frames,
        start_frame=start_frame,
        color_by=color_by,
        draw_boxes=draw_boxes,
        draw_labels=draw_labels,
        draw_polygons=draw_polygons,
        draw_keypoints=draw_keypoints,
        draw_tails=draw_tails,
        tail_length=tail_length,
        keypoint_threshold=keypoint_threshold,
        box_thickness=box_thickness,
        label_scale=label_scale,
        label_thickness=label_thickness,
        codec=codec,
        return_predictions=return_predictions,
    )


def _prediction_color(
    row: pd.Series,
    row_idx: int,
    color_by: str,
    confidence_range: tuple[float, float],
) -> tuple[int, int, int]:
    if color_by == "confidence":
        conf = float(row.get("confidence", 0.0) or 0.0)
        low, high = confidence_range
        if high <= low:
            high = low + 1.0
        value = max(0.0, min(1.0, (conf - low) / (high - low)))
        return (0, int(255 * value), int(255 * (1.0 - value)))

    if color_by == "track_id" and "track_id" in row and pd.notna(row["track_id"]):
        return _palette_color(int(row["track_id"]))

    if color_by == "class" and "class_id" in row and pd.notna(row["class_id"]):
        return _palette_color(int(row["class_id"]))

    return _palette_color(row_idx)


def _palette_color(index: int) -> tuple[int, int, int]:
    palette = [
        (255, 99, 71),
        (60, 179, 113),
        (65, 105, 225),
        (255, 165, 0),
        (186, 85, 211),
        (0, 206, 209),
        (220, 20, 60),
        (154, 205, 50),
        (255, 105, 180),
        (70, 130, 180),
    ]
    rgb = palette[index % len(palette)]
    return (rgb[2], rgb[1], rgb[0])


def _has_bbox(row: pd.Series) -> bool:
    required = ["xmin", "ymin", "xmax", "ymax"]
    return all(col in row and pd.notna(row[col]) for col in required)


def _bbox_int(row: pd.Series) -> tuple[int, int, int, int]:
    return (
        int(round(float(row["xmin"]))),
        int(round(float(row["ymin"]))),
        int(round(float(row["xmax"]))),
        int(round(float(row["ymax"]))),
    )


def _has_points(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, float) and pd.isna(value):
        return False
    try:
        return len(value) > 0
    except TypeError:
        return False


def _draw_label_cv2(
    frame: "np.ndarray",
    row: pd.Series,
    color: tuple[int, int, int],
    label_scale: float,
    label_thickness: int,
) -> None:
    import cv2

    x1, y1, _, _ = _bbox_int(row)
    name = str(row.get("class_name", row.get("class_id", "")))
    if "track_id" in row and pd.notna(row["track_id"]):
        name = f"{name} id:{int(row['track_id'])}"
    if "confidence" in row and pd.notna(row["confidence"]):
        name = f"{name} {float(row['confidence']):.2f}"

    (text_w, text_h), baseline = cv2.getTextSize(
        name,
        cv2.FONT_HERSHEY_SIMPLEX,
        label_scale,
        label_thickness,
    )
    y_text = max(y1 - 5, text_h + baseline + 2)
    cv2.rectangle(
        frame,
        (x1, y_text - text_h - baseline - 4),
        (x1 + text_w + 4, y_text + baseline),
        color,
        thickness=-1,
    )
    cv2.putText(
        frame,
        name,
        (x1 + 2, y_text - 3),
        cv2.FONT_HERSHEY_SIMPLEX,
        label_scale,
        (255, 255, 255),
        label_thickness,
        cv2.LINE_AA,
    )


def _draw_keypoints_cv2(
    frame: "np.ndarray",
    keypoints: Any,
    color: tuple[int, int, int],
    keypoint_threshold: float,
    thickness: int,
) -> None:
    import cv2

    arr = np.asarray(keypoints, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 2:
        return
    if arr.shape[1] == 2:
        visibility = np.ones((arr.shape[0], 1), dtype=float)
        arr = np.hstack([arr, visibility])

    visible = arr[:, 2] >= keypoint_threshold
    skeleton = [
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
    for start, end in skeleton:
        if start >= len(arr) or end >= len(arr):
            continue
        if not (visible[start] and visible[end]):
            continue
        p1 = (int(round(arr[start, 0])), int(round(arr[start, 1])))
        p2 = (int(round(arr[end, 0])), int(round(arr[end, 1])))
        cv2.line(frame, p1, p2, color, thickness)

    for x, y, score in arr:
        if score < keypoint_threshold:
            continue
        cv2.circle(frame, (int(round(x)), int(round(y))), 3, color, thickness=-1)
        cv2.circle(frame, (int(round(x)), int(round(y))), 4, (255, 255, 255), thickness=1)


def get_video_info(video_path: str | Path) -> dict:
    """Return basic metadata about a video file."""
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    info = {
        "path": str(video_path),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "duration_s": None,
    }
    if info["fps"]:
        info["duration_s"] = info["frame_count"] / info["fps"]
    cap.release()
    return info
