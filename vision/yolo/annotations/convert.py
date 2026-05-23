"""High-level annotation conversion dispatcher.

The conversion pipeline is::

    any format → AnnotationSample (internal) → YOLO

Usage::

    samples = convert_annotations(
        source_dir="coco_data/",
        target_dir="yolo_data/",
        source_fmt="coco",
        target_fmt="yolo",
        class_names=["cat", "dog"],
    )
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from vision.yolo.annotations.internal import AnnotationSample


_READERS: dict[str, str] = {
    "coco": "vision.yolo.annotations.coco",
    "yolo": "vision.yolo.annotations.yolo",
    "voc": "vision.yolo.annotations.voc",
    "labelme": "vision.yolo.annotations.labelme",
    "roboflow": "vision.yolo.annotations.roboflow",
}

_WRITERS: dict[str, str] = {
    "yolo": "vision.yolo.annotations.yolo",
    "coco": "vision.yolo.annotations.coco",
    "voc": "vision.yolo.annotations.voc",
}


def convert_annotations(
    source_dir: str | Path,
    target_dir: str | Path,
    source_fmt: str,
    target_fmt: str = "yolo",
    class_names: Optional[list[str]] = None,
    **kwargs,
) -> list[AnnotationSample]:
    """Convert annotations from *source_fmt* to *target_fmt*.

    Parameters
    ----------
    source_dir:
        Directory containing source annotation files.
    target_dir:
        Output directory.
    source_fmt:
        One of: ``'coco'``, ``'yolo'``, ``'voc'``, ``'labelme'``,
        ``'roboflow'``.
    target_fmt:
        One of: ``'yolo'``, ``'coco'``, ``'voc'``.
    class_names:
        Optional class name list; required when source does not embed names.

    Returns
    -------
    List of :class:`AnnotationSample` in the internal format.
    """
    source_fmt = source_fmt.lower()
    target_fmt = target_fmt.lower()

    if source_fmt not in _READERS:
        raise ValueError(f"Unknown source format {source_fmt!r}. Supported: {sorted(_READERS)}.")
    if target_fmt not in _WRITERS:
        raise ValueError(f"Unknown target format {target_fmt!r}. Supported: {sorted(_WRITERS)}.")

    # Load reader/writer lazily
    import importlib

    reader_mod = importlib.import_module(_READERS[source_fmt])
    writer_mod = importlib.import_module(_WRITERS[target_fmt])

    samples: list[AnnotationSample] = reader_mod.read(
        Path(source_dir), class_names=class_names, **kwargs
    )
    writer_mod.write(samples, Path(target_dir), class_names=class_names, **kwargs)

    return samples
