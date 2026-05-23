"""LabelMe JSON annotation reader."""

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
    """Read LabelMe JSON annotation files from *source_dir*."""
    samples: list[AnnotationSample] = []
    name_to_id: dict[str, int] = {n: i for i, n in enumerate(class_names or [])}

    for json_path in sorted(source_dir.glob("*.json")):
        with open(json_path) as fh:
            data = json.load(fh)

        image_path = data.get("imagePath", json_path.stem)
        w = data.get("imageWidth", 0)
        h = data.get("imageHeight", 0)

        anns: list[Annotation] = []
        for shape in data.get("shapes", []):
            cls_name = shape.get("label", "")
            cls_id = name_to_id.setdefault(cls_name, len(name_to_id))
            shape_type = shape.get("shape_type", "polygon")
            points = shape.get("points", [])

            bbox: list[float] = []
            polygon: list[list[float]] = []

            if shape_type == "rectangle" and len(points) == 2:
                x1, y1 = points[0]
                x2, y2 = points[1]
                bbox = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
            elif points:
                xs = [p[0] for p in points]
                ys = [p[1] for p in points]
                bbox = [min(xs), min(ys), max(xs), max(ys)]
                polygon = [[float(p[0]), float(p[1])] for p in points]

            task = "segment" if polygon else "detect"
            anns.append(
                Annotation(
                    task=task,
                    class_id=cls_id,
                    class_name=cls_name,
                    bbox=bbox,
                    polygon=polygon,
                )
            )

        samples.append(
            AnnotationSample(
                image_path=image_path,
                width=w,
                height=h,
                annotations=anns,
            )
        )

    return samples


def write(
    samples: list[AnnotationSample],
    target_dir: Path,
    class_names: Optional[list[str]] = None,
    **kwargs,
) -> None:
    """Write LabelMe JSON annotation files to *target_dir*."""
    target_dir.mkdir(parents=True, exist_ok=True)
    for sample in samples:
        shapes = []
        for ann in sample.annotations:
            if ann.polygon:
                shape = {
                    "label": ann.class_name,
                    "points": ann.polygon,
                    "shape_type": "polygon",
                    "flags": {},
                }
            elif len(ann.bbox) == 4:
                x1, y1, x2, y2 = ann.bbox
                shape = {
                    "label": ann.class_name,
                    "points": [[x1, y1], [x2, y2]],
                    "shape_type": "rectangle",
                    "flags": {},
                }
            else:
                continue
            shapes.append(shape)

        data = {
            "version": "5.0.0",
            "flags": {},
            "shapes": shapes,
            "imagePath": Path(sample.image_path).name,
            "imageData": None,
            "imageWidth": sample.width,
            "imageHeight": sample.height,
        }
        stem = Path(sample.image_path).stem
        with open(target_dir / f"{stem}.json", "w") as fh:
            json.dump(data, fh, indent=2)
