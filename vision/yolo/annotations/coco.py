"""COCO JSON annotation reader and writer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from vision.yolo.annotations.internal import Annotation, AnnotationSample


def read(
    source_dir: Path,
    annotation_file: str = "annotations.json",
    class_names: Optional[list[str]] = None,
    **kwargs,
) -> list[AnnotationSample]:
    """Read a COCO JSON file from *source_dir*.

    Parameters
    ----------
    annotation_file:
        Name of the JSON file inside *source_dir*.
    """
    json_path = source_dir / annotation_file
    if not json_path.exists():
        # Try common COCO patterns
        for candidate in source_dir.glob("*.json"):
            json_path = candidate
            break

    with open(json_path) as fh:
        data = json.load(fh)

    # Build id → name map
    cat_map: dict[int, str] = {c["id"]: c["name"] for c in data.get("categories", [])}
    # Build image id → image info map
    img_map: dict[int, dict] = {img["id"]: img for img in data.get("images", [])}
    # Build image id → annotations
    ann_by_img: dict[int, list] = {}
    for ann in data.get("annotations", []):
        ann_by_img.setdefault(ann["image_id"], []).append(ann)

    samples: list[AnnotationSample] = []
    for img_id, img_info in img_map.items():
        anns: list[Annotation] = []
        for ann in ann_by_img.get(img_id, []):
            cat_id = ann["category_id"]
            cls_name = cat_map.get(cat_id, str(cat_id))
            x, y, bw, bh = ann["bbox"]  # COCO: [x, y, width, height]
            x2, y2 = x + bw, y + bh
            segmentation = ann.get("segmentation", [])
            polygon: list[list[float]] = []
            if segmentation and isinstance(segmentation[0], list):
                flat = segmentation[0]
                polygon = [[flat[i], flat[i + 1]] for i in range(0, len(flat), 2)]

            anns.append(
                Annotation(
                    task="segment" if polygon else "detect",
                    class_id=cat_id,
                    class_name=cls_name,
                    bbox=[float(x), float(y), float(x2), float(y2)],
                    polygon=polygon,
                )
            )

        samples.append(
            AnnotationSample(
                image_path=img_info.get("file_name", ""),
                width=img_info.get("width", 0),
                height=img_info.get("height", 0),
                annotations=anns,
            )
        )

    return samples


def write(
    samples: list[AnnotationSample],
    target_dir: Path,
    class_names: Optional[list[str]] = None,
    annotation_file: str = "annotations.json",
    **kwargs,
) -> None:
    """Write samples to a COCO JSON file in *target_dir*."""
    target_dir.mkdir(parents=True, exist_ok=True)

    # Build category list from samples or class_names
    if class_names:
        categories = [{"id": i, "name": n, "supercategory": ""} for i, n in enumerate(class_names)]
        name_to_id = {n: i for i, n in enumerate(class_names)}
    else:
        seen: dict[str, int] = {}
        for s in samples:
            for a in s.annotations:
                if a.class_name not in seen:
                    seen[a.class_name] = a.class_id
        categories = [{"id": v, "name": k, "supercategory": ""} for k, v in sorted(seen.items(), key=lambda x: x[1])]
        name_to_id = {c["name"]: c["id"] for c in categories}

    images = []
    annotations = []
    ann_id = 1

    for img_id, sample in enumerate(samples, start=1):
        images.append(
            {
                "id": img_id,
                "file_name": sample.image_path,
                "width": sample.width,
                "height": sample.height,
            }
        )
        for ann in sample.annotations:
            x1, y1, x2, y2 = ann.bbox if len(ann.bbox) == 4 else [0, 0, 0, 0]
            bw, bh = x2 - x1, y2 - y1
            area = bw * bh
            seg: list = []
            if ann.polygon:
                flat = [coord for pt in ann.polygon for coord in pt]
                seg = [flat]
            coco_ann = {
                "id": ann_id,
                "image_id": img_id,
                "category_id": name_to_id.get(ann.class_name, ann.class_id),
                "bbox": [float(x1), float(y1), float(bw), float(bh)],
                "area": float(area),
                "segmentation": seg,
                "iscrowd": 0,
            }
            annotations.append(coco_ann)
            ann_id += 1

    output = {
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }

    with open(target_dir / annotation_file, "w") as fh:
        json.dump(output, fh, indent=2)
