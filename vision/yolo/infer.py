"""Inference functions for detection, segmentation, and pose estimation."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Generator, Optional

import numpy as np
import pandas as pd

from vision.yolo.dataframe import concat_results, ultralytics_result_to_dataframe
from vision.yolo.devices import detect_device
from vision.yolo.utils import load_model


def predict_image(
    model_path: str,
    image: "str | Path | np.ndarray",  # type: ignore[name-defined]
    confidence: float = 0.25,
    iou: float = 0.45,
    device: Optional[str] = None,
    task: str = "detect",
    **kwargs: Any,
) -> pd.DataFrame:
    """Run inference on a single image.

    Parameters
    ----------
    model_path:
        Path to a ``.pt`` weights file or model name (e.g. ``'yolo11n.pt'``).
    image:
        File path or numpy array (BGR or RGB).
    confidence / iou:
        Ultralytics conf and iou thresholds.
    device:
        Device string; auto-detected when ``None``.
    task:
        ``'detect'``, ``'segment'``, or ``'pose'``.

    Returns
    -------
    pd.DataFrame with one row per detection.
    """
    device = device or detect_device()
    model = load_model(model_path)
    results = model.predict(
        image, conf=confidence, iou=iou, device=device, verbose=False, **kwargs
    )
    return ultralytics_result_to_dataframe(results[0], frame_idx=0)


def predict_images(
    model_path: str,
    images: list,
    confidence: float = 0.25,
    iou: float = 0.45,
    batch_size: int = 8,
    device: Optional[str] = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Run inference on a list of images, returning a single concatenated DataFrame."""
    device = device or detect_device()
    model = load_model(model_path)
    all_dfs: list[pd.DataFrame] = []

    for start in range(0, len(images), batch_size):
        batch = images[start : start + batch_size]
        results = model.predict(
            batch, conf=confidence, iou=iou, device=device, verbose=False, **kwargs
        )
        for idx, result in enumerate(results):
            df = ultralytics_result_to_dataframe(result, frame_idx=start + idx)
            all_dfs.append(df)

    return concat_results(all_dfs)


def predict_video(
    model_path: str,
    video_path: str | Path,
    confidence: float = 0.25,
    iou: float = 0.45,
    device: Optional[str] = None,
    batch_size: int = 8,
    tracker: Optional[str] = None,
    tracker_config: Optional[dict[str, Any]] = None,
    tracker_config_save_to: Optional[str | Path] = None,
    tracking_persist: bool = True,
    save_to: Optional[str | Path] = None,
    save_fmt: str = "parquet",
    imgsz: int = 640,
    **kwargs: Any,
) -> pd.DataFrame:
    """Run inference or tracking on a video file, frame by frame.

    Frames are processed in batches; the video is never fully loaded into
    memory at once.

    Parameters
    ----------
    tracker:
        Optional Ultralytics tracker config, e.g. ``'bytetrack.yaml'`` or
        ``'botsort.yaml'``. When provided, ``model.track`` is used instead of
        ``model.predict`` and the output DataFrame can include ``track_id``.
    tracker_config:
        Optional YAML overrides for the tracker, e.g.
        ``{'track_high_thresh': 0.4, 'track_buffer': 60}``. When provided,
        a custom tracker YAML is generated and used for this run.
    tracker_config_save_to:
        Optional path where the generated tracker YAML should be saved.
    tracking_persist:
        Reuse tracker state between batches. Keep ``True`` for normal videos.
    save_to:
        Optional path to persist results incrementally (parquet/feather/csv).
    """
    import cv2
    import time
    from vision.yolo.serialization import append_to_parquet, save_dataframe

    # get information about the video for progress tracking
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    dimensions = (
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
    cap.release()



    device = device or detect_device()
    print(f"Inference on {device}")
    start_time = time.time()

    model = load_model(model_path)
    all_dfs: list[pd.DataFrame] = []
    temp_dir: Optional[TemporaryDirectory] = None
    tracker_for_run = tracker

    if tracker_config:
        from vision.yolo.track import make_tracker_config

        base_tracker = tracker or "bytetrack.yaml"
        if tracker_config_save_to is None:
            temp_dir = TemporaryDirectory()
            tracker_config_save_to = Path(temp_dir.name) / f"custom_{Path(base_tracker).name}"
        tracker_for_run = str(
            make_tracker_config(
                base_tracker=base_tracker,
                overrides=tracker_config,
                save_to=tracker_config_save_to,
            )
        )

    try:
        for batch_frames, batch_indices, batch_timestamps in _batch_video(
            video_path, batch_size
        ):
            if tracker_for_run:
                results = model.track(
                    batch_frames,
                    tracker=tracker_for_run,
                    conf=confidence,
                    iou=iou,
                    device=device,
                    persist=tracking_persist,
                    verbose=False,
                    imgsz=imgsz,
                    **kwargs,
                )
            else:
                results = model.predict(
                    batch_frames,
                    conf=confidence,
                    iou=iou,
                    device=device,
                    verbose=False,
                    imgsz=imgsz,
                    **kwargs,
                )
            batch_dfs: list[pd.DataFrame] = []
            for result, idx, ts in zip(results, batch_indices, batch_timestamps):
                df = ultralytics_result_to_dataframe(result, frame_idx=idx, timestamp=ts)
                batch_dfs.append(df)

            chunk = concat_results(batch_dfs)

            if save_to and save_fmt == "parquet":
                append_to_parquet(chunk, save_to)

            all_dfs.append(chunk)
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()

    final = concat_results(all_dfs)

    if save_to and save_fmt != "parquet":
        save_dataframe(final, save_to, fmt=save_fmt)
    end_time = time.time()

    inference_time = end_time - start_time

    # calculate if the inference was faster than real-time
    if total_frames > 0 and timestamp > 0:
        video_duration = timestamp  # in seconds
        print(f"Video duration: {video_duration:.2f} seconds")
        print(f"Inference time: {inference_time:.2f} seconds")
        if inference_time < video_duration:
            print("Inference was faster than real-time!")
        else:
            print("Inference was slower than real-time.")


    print(f"Total inference time: {end_time - start_time:.2f} seconds")

    return final


def predict_directory(
    model_path: str,
    directory: str | Path,
    pattern: str = "*.jpg",
    confidence: float = 0.25,
    iou: float = 0.45,
    batch_size: int = 8,
    device: Optional[str] = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Run inference on all matching images inside a directory."""
    paths = sorted(Path(directory).glob(pattern))
    if not paths:
        import pandas as pd
        return pd.DataFrame()

    return predict_images(
        model_path,
        [str(p) for p in paths],
        confidence=confidence,
        iou=iou,
        batch_size=batch_size,
        device=device,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _batch_video(
    video_path: str | Path,
    batch_size: int,
) -> Generator[tuple[list, list[int], list[float]], None, None]:
    """Yield (frames, indices, timestamps) in batches without loading full video."""
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_idx = 0
    batch_frames: list = []
    batch_indices: list[int] = []
    batch_ts: list[float] = []

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
                yield batch_frames, batch_indices, batch_ts
                batch_frames, batch_indices, batch_ts = [], [], []

        if batch_frames:
            yield batch_frames, batch_indices, batch_ts
    finally:
        cap.release()
