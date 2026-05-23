"""Roboflow YOLO-format annotation reader.

Roboflow exports YOLO datasets as a zip with the following layout::

    dataset/
    ├── train/
    │   ├── images/
    │   └── labels/
    ├── valid/
    │   ├── images/
    │   └── labels/
    ├── test/
    │   ├── images/
    │   └── labels/
    └── data.yaml

This reader accepts such a directory and delegates to the YOLO reader
for each split.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from vision.yolo.annotations.internal import AnnotationSample
from vision.yolo.annotations import yolo as _yolo_reader


def read(
    source_dir: Path,
    class_names: Optional[list[str]] = None,
    splits: Optional[list[str]] = None,
    **kwargs,
) -> list[AnnotationSample]:
    """Read a Roboflow YOLO export from *source_dir*.

    Parameters
    ----------
    splits:
        Subset of ``['train', 'valid', 'test']`` to load.
        Defaults to all present splits.
    """
    # Try to read class names from data.yaml
    if class_names is None:
        class_names = _read_classes_from_yaml(source_dir)

    if splits is None:
        splits = [d.name for d in source_dir.iterdir() if d.is_dir() and d.name in ("train", "valid", "test")]

    all_samples: list[AnnotationSample] = []
    for split in splits:
        labels_dir = source_dir / split / "labels"
        images_dir = source_dir / split / "images"
        if not labels_dir.exists():
            continue
        samples = _yolo_reader.read(
            labels_dir, class_names=class_names, image_dir=images_dir, **kwargs
        )
        all_samples.extend(samples)

    return all_samples


def write(
    samples: list[AnnotationSample],
    target_dir: Path,
    class_names: Optional[list[str]] = None,
    **kwargs,
) -> None:
    """Write samples in YOLO format – suitable for re-importing into Roboflow."""
    from vision.yolo.annotations import yolo as _yolo_writer

    _yolo_writer.write(samples, target_dir / "labels", class_names=class_names, **kwargs)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_classes_from_yaml(source_dir: Path) -> Optional[list[str]]:
    yaml_path = source_dir / "data.yaml"
    if not yaml_path.exists():
        return None
    try:
        import yaml

        with open(yaml_path) as fh:
            data = yaml.safe_load(fh)
        names = data.get("names", [])
        return list(names) if names else None
    except Exception:
        return None
