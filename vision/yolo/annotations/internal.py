"""Internal unified annotation format.

All format converters work through this common representation so that
adding a new input or output format only requires implementing one
adapter.

Internal format::

    AnnotationSample(
        image_path="path/to/image.jpg",
        width=640,
        height=480,
        annotations=[
            Annotation(
                task="detect",       # "detect" | "segment" | "pose"
                class_id=0,
                class_name="cat",
                bbox=[x1, y1, x2, y2],   # pixel coords, absolute
                polygon=[[x, y], ...],   # optional, segmentation
                keypoints=[[x, y, v], ...],  # optional, pose
            )
        ]
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Annotation:
    task: str = "detect"           # "detect" | "segment" | "pose"
    class_id: int = 0
    class_name: str = ""
    bbox: list[float] = field(default_factory=list)      # [x1, y1, x2, y2] absolute px
    polygon: list[list[float]] = field(default_factory=list)   # [[x, y], ...]
    keypoints: list[list[float]] = field(default_factory=list) # [[x, y, visibility], ...]


@dataclass
class AnnotationSample:
    image_path: str = ""
    width: int = 0
    height: int = 0
    annotations: list[Annotation] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "image_path": self.image_path,
            "width": self.width,
            "height": self.height,
            "annotations": [
                {
                    "task": a.task,
                    "class_id": a.class_id,
                    "class_name": a.class_name,
                    "bbox": a.bbox,
                    "polygon": a.polygon,
                    "keypoints": a.keypoints,
                }
                for a in self.annotations
            ],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "AnnotationSample":
        anns = [
            Annotation(
                task=a.get("task", "detect"),
                class_id=a.get("class_id", 0),
                class_name=a.get("class_name", ""),
                bbox=a.get("bbox", []),
                polygon=a.get("polygon", []),
                keypoints=a.get("keypoints", []),
            )
            for a in d.get("annotations", [])
        ]
        return cls(
            image_path=d.get("image_path", ""),
            width=d.get("width", 0),
            height=d.get("height", 0),
            annotations=anns,
        )
