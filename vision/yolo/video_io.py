"""Simple video I/O helpers exposed through Python APIs.

The default video backend is OpenCV. It does not call an external ``ffmpeg``
command. Compressed video codecs are still provided by native libraries bundled
with or available to OpenCV.
"""

from __future__ import annotations

import os
from typing import Generator, Optional

import numpy as np


def read_frames(
    video_path: str,
    start_frame: int = 0,
    max_frames: Optional[int] = None,
    step: int = 1,
) -> Generator[tuple[int, float, np.ndarray], None, None]:
    """Yield ``(frame_index, timestamp_seconds, frame_bgr)`` using OpenCV."""
    import cv2

    if step < 1:
        raise ValueError("step must be >= 1")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frame_index = start_frame
    yielded = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if (frame_index - start_frame) % step == 0:
                timestamp = frame_index / fps
                yield frame_index, timestamp, frame
                yielded += 1

                if max_frames is not None and yielded >= max_frames:
                    break

            frame_index += 1
    finally:
        cap.release()


def video_info(video_path: str) -> dict:
    """Return basic video metadata using OpenCV only."""
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    duration = frame_count / fps if fps > 0 else None
    return {
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration_seconds": duration,
    }


def write_frames(
    frames,
    output_path: str,
    fps: float,
    codec: str = "mp4v",
) -> str:
    """Write an iterable of BGR frames to video using OpenCV."""
    import cv2

    iterator = iter(frames)
    try:
        first_frame = next(iterator)
    except StopIteration as exc:
        raise ValueError("No frames were provided") from exc

    height, width = first_frame.shape[:2]
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*codec),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise ValueError(f"Could not open video for writing: {output_path}")

    try:
        writer.write(first_frame)
        for frame in iterator:
            if frame.shape[:2] != (height, width):
                raise ValueError("All frames must have the same width and height")
            writer.write(frame)
    finally:
        writer.release()

    return output_path


def extract_frames(
    video_path: str,
    output_dir: str,
    every_n_frames: int = 1,
    max_frames: Optional[int] = None,
    extension: str = "jpg",
) -> list[str]:
    """Extract frames without invoking an external video command."""
    import cv2

    os.makedirs(output_dir, exist_ok=True)
    saved = []

    for frame_index, _, frame in read_frames(
        video_path,
        max_frames=max_frames,
        step=every_n_frames,
    ):
        filename = os.path.join(output_dir, f"frame_{frame_index:06d}.{extension}")
        ok = cv2.imwrite(filename, frame)
        if not ok:
            raise ValueError(f"Could not write frame: {filename}")
        saved.append(filename)

    return saved


def images_to_video(
    image_paths: list[str],
    output_path: str,
    fps: float = 30.0,
    codec: str = "mp4v",
) -> str:
    """Create a video from image files using OpenCV and constant memory."""
    import cv2

    if not image_paths:
        raise ValueError("image_paths is empty")

    def frame_generator():
        for image_path in image_paths:
            frame = cv2.imread(image_path)
            if frame is None:
                raise ValueError(f"Could not read image: {image_path}")
            yield frame

    return write_frames(frame_generator(), output_path, fps=fps, codec=codec)


def images_to_gif(
    image_paths: list[str],
    output_path: str,
    fps: float = 10.0,
    loop: int = 0,
) -> str:
    """Create an animated GIF using Pillow, without ffmpeg."""
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError("Install GIF support with: pip install -e '.[video]'") from exc

    if not image_paths:
        raise ValueError("image_paths is empty")
    if fps <= 0:
        raise ValueError("fps must be > 0")

    images = []
    for image_path in image_paths:
        with Image.open(image_path) as image:
            images.append(image.convert("RGB").copy())

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    duration_ms = round(1000 / fps)
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=loop,
    )
    return output_path
