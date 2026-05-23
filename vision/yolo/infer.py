"""Inference functions for detection, segmentation, and pose estimation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Generator, Optional

import pandas as pd

from vision.yolo.dataframe import ultralytics_result_to_dataframe, concat_results
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
    save_to: Optional[str | Path] = None,
    save_fmt: str = "parquet",
    **kwargs: Any,
) -> pd.DataFrame:
    """Run inference on a video file, frame by frame.

    Frames are processed in batches; the video is never fully loaded into
    memory at once.

    Parameters
    ----------
    save_to:
        Optional path to persist results incrementally (parquet/feather/csv).
    """
    from vision.yolo.video import video_frame_generator
    from vision.yolo.serialization import append_to_parquet, save_dataframe

    device = device or detect_device()
    model = load_model(model_path)
    all_dfs: list[pd.DataFrame] = []

    for batch_frames, batch_indices, batch_timestamps in _batch_video(
        video_path, batch_size
    ):
        results = model.predict(
            batch_frames, conf=confidence, iou=iou, device=device, verbose=False, **kwargs
        )
        batch_dfs: list[pd.DataFrame] = []
        for result, idx, ts in zip(results, batch_indices, batch_timestamps):
            df = ultralytics_result_to_dataframe(result, frame_idx=idx, timestamp=ts)
            batch_dfs.append(df)

        chunk = concat_results(batch_dfs)

        if save_to and save_fmt == "parquet":
            append_to_parquet(chunk, save_to)

        all_dfs.append(chunk)

    final = concat_results(all_dfs)

    if save_to and save_fmt != "parquet":
        save_dataframe(final, save_to, fmt=save_fmt)

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
