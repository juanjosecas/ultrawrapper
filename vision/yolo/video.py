"""Streaming video frame generator and prediction writer.

Videos are NEVER loaded fully into memory – frames are yielded one at a time
or in batches.
"""

from __future__ import annotations

from pathlib import Path
from typing import Generator, Optional


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
    from vision.yolo.dataframe import ultralytics_result_to_dataframe, concat_results
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
