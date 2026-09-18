"""Video and image-sequence I/O without invoking an ffmpeg subprocess."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
import os

import cv2
import numpy as np


def video_info(video_path: str) -> dict[str, float | int]:
    """Return basic metadata reported by OpenCV for a video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    info = {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": float(cap.get(cv2.CAP_PROP_FPS)),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "duration": 0.0,
    }
    cap.release()

    if info["fps"] > 0:
        info["duration"] = info["frame_count"] / info["fps"]
    return info


def read_frames(
    video_path: str,
    start: int = 0,
    stop: int | None = None,
    step: int = 1,
) -> Iterator[tuple[int, np.ndarray]]:
    """Yield ``(frame_index, frame)`` pairs without loading the full video."""
    if start < 0:
        raise ValueError("start must be >= 0")
    if step < 1:
        raise ValueError("step must be >= 1")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    frame_idx = start

    try:
        while stop is None or frame_idx < stop:
            ok, frame = cap.read()
            if not ok:
                break

            if (frame_idx - start) % step == 0:
                yield frame_idx, frame
            frame_idx += 1
    finally:
        cap.release()


def write_frames(
    frames: Iterable[np.ndarray],
    output_path: str,
    fps: float,
    codec: str = "mp4v",
) -> str:
    """Write an iterable of BGR frames to a video, streaming one frame at a time."""
    if fps <= 0:
        raise ValueError("fps must be > 0")
    if len(codec) != 4:
        raise ValueError("codec must contain exactly four characters")

    iterator = iter(frames)
    try:
        first = next(iterator)
    except StopIteration as exc:
        raise ValueError("frames is empty") from exc

    if first.ndim != 3 or first.shape[2] != 3:
        raise ValueError("frames must be BGR images with shape (height, width, 3)")

    height, width = first.shape[:2]
    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*codec),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise ValueError(f"Could not create video: {output_path}")

    try:
        writer.write(first)
        for frame in iterator:
            if frame.shape[:2] != (height, width):
                raise ValueError("all frames must have the same width and height")
            writer.write(frame)
    finally:
        writer.release()

    return output_path


def extract_frames(
    video_path: str,
    output_dir: str,
    every: int = 1,
    start: int = 0,
    stop: int | None = None,
    prefix: str = "frame",
    extension: str = ".jpg",
) -> list[str]:
    """Extract selected frames to disk and return the created paths."""
    if every < 1:
        raise ValueError("every must be >= 1")

    os.makedirs(output_dir, exist_ok=True)
    paths: list[str] = []

    for frame_idx, frame in read_frames(video_path, start=start, stop=stop, step=every):
        path = os.path.join(output_dir, f"{prefix}_{frame_idx:06d}{extension}")
        if not cv2.imwrite(path, frame):
            raise ValueError(f"Could not write image: {path}")
        paths.append(path)

    return paths


def images_to_video(
    image_paths: Sequence[str],
    output_path: str,
    fps: float = 30.0,
    codec: str = "mp4v",
    size: tuple[int, int] | None = None,
    size_factor: float | None = None,
) -> str:
    """Create a video from image paths without loading the sequence into memory."""
    if not image_paths:
        raise ValueError("image_paths is empty")

    def frames() -> Iterator[np.ndarray]:
        for image_path in image_paths:
            frame = cv2.imread(image_path)
            if frame is None:
                raise ValueError(f"Could not read image: {image_path}")
            if size is not None:
                frame = cv2.resize(frame, size)
            if size_factor is not None:
                frame = cv2.resize(frame, None, fx=size_factor, fy=size_factor)
            yield frame

    return write_frames(frames(), output_path, fps=fps, codec=codec)


def images_to_gif(
    image_paths: Sequence[str],
    output_path: str,
    duration_ms: int = 100,
    loop: int = 0,
) -> str:
    """Create a GIF through Pillow, with no external ffmpeg command."""
    if not image_paths:
        raise ValueError("image_paths is empty")
    if duration_ms <= 0:
        raise ValueError("duration_ms must be > 0")

    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError("Install the video extra with: pip install -e '.[video]'") from exc

    first = Image.open(image_paths[0]).convert("RGB")
    rest = []
    try:
        for path in image_paths[1:]:
            with Image.open(path) as image:
                rest.append(image.convert("RGB").copy())

        parent = os.path.dirname(output_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        first.save(
            output_path,
            save_all=True,
            append_images=rest,
            duration=duration_ms,
            loop=loop,
        )
    finally:
        first.close()
        for image in rest:
            image.close()

    return output_path
