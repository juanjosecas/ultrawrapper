"""X-AnyLabeling JSON annotation reader and writer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from vision.yolo.annotations.internal import Annotation, AnnotationSample


def read(
    source_dir: Path,
    class_names: Optional[list[str]] = None,
    **kwargs,
) -> list[AnnotationSample]:
    """Read X-AnyLabeling JSON annotation files from *source_dir*."""
    samples: list[AnnotationSample] = []
    name_to_id: dict[str, int] = {name: idx for idx, name in enumerate(class_names or [])}

    for json_path in sorted(source_dir.glob("*.json")):
        with open(json_path) as fh:
            data = json.load(fh)

        image_path = _resolve_image_path(json_path, data.get("imagePath"))
        width = int(data.get("imageWidth", 0) or 0)
        height = int(data.get("imageHeight", 0) or 0)

        annotations: list[Annotation] = []
        for shape in data.get("shapes", []):
            cls_name = str(shape.get("label", ""))
            cls_id = name_to_id.setdefault(cls_name, len(name_to_id))
            shape_type = shape.get("shape_type", "polygon")
            points = shape.get("points", [])

            bbox: list[float] = []
            polygon: list[list[float]] = []

            if shape_type == "rectangle" and len(points) == 2:
                x1, y1 = points[0]
                x2, y2 = points[1]
                bbox = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
            elif shape_type == "polygon" and points:
                xs = [point[0] for point in points]
                ys = [point[1] for point in points]
                bbox = [min(xs), min(ys), max(xs), max(ys)]
                polygon = [[float(point[0]), float(point[1])] for point in points]
            else:
                continue

            annotations.append(
                Annotation(
                    task="segment" if polygon else "detect",
                    class_id=cls_id,
                    class_name=cls_name,
                    bbox=bbox,
                    polygon=polygon,
                )
            )

        samples.append(
            AnnotationSample(
                image_path=image_path,
                width=width,
                height=height,
                annotations=annotations,
            )
        )

    return samples


def write(
    samples: list[AnnotationSample],
    target_dir: Path,
    class_names: Optional[list[str]] = None,
    **kwargs,
) -> None:
    """Write X-AnyLabeling-compatible JSON annotation files to *target_dir*."""
    del class_names
    target_dir.mkdir(parents=True, exist_ok=True)

    for sample in samples:
        shapes = []
        for ann in sample.annotations:
            shape = _annotation_to_shape(ann)
            if shape is not None:
                shapes.append(shape)

        payload = {
            "version": "x-anylabeling",
            "flags": {},
            "tags": [],
            "shapes": shapes,
            "description": "",
            "imagePath": Path(sample.image_path).name,
            "imageData": None,
            "imageHeight": sample.height,
            "imageWidth": sample.width,
            "checked": False,
        }
        stem = Path(sample.image_path).stem
        with open(target_dir / f"{stem}.json", "w") as fh:
            json.dump(payload, fh, indent=2)


def _annotation_to_shape(ann: Annotation) -> dict | None:
    if ann.polygon:
        return {
            "label": ann.class_name,
            "score": None,
            "points": ann.polygon,
            "group_id": None,
            "description": "",
            "difficult": False,
            "shape_type": "polygon",
            "flags": {},
            "attributes": {},
        }

    if len(ann.bbox) == 4:
        x1, y1, x2, y2 = ann.bbox
        return {
            "label": ann.class_name,
            "score": None,
            "points": [[x1, y1], [x2, y2]],
            "group_id": None,
            "description": "",
            "difficult": False,
            "shape_type": "rectangle",
            "flags": {},
            "attributes": {},
        }

    return None


def _resolve_image_path(json_path: Path, declared_image_path: str | None) -> str:
    if declared_image_path:
        declared_path = Path(declared_image_path)
        if declared_path.is_absolute():
            return str(declared_path)
        return str((json_path.parent / declared_path).resolve())

    for suffix in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"):
        candidate = json_path.with_suffix(suffix)
        if candidate.exists():
            return str(candidate.resolve())

    return str(json_path.with_suffix(".jpg").resolve())
