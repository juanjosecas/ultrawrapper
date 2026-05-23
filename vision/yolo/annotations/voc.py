"""Pascal VOC XML annotation reader and writer."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

from vision.yolo.annotations.internal import Annotation, AnnotationSample


def read(
    source_dir: Path,
    class_names: Optional[list[str]] = None,
    **kwargs,
) -> list[AnnotationSample]:
    """Read Pascal VOC XML files from *source_dir*."""
    samples: list[AnnotationSample] = []
    name_to_id: dict[str, int] = {n: i for i, n in enumerate(class_names or [])}

    for xml_path in sorted(source_dir.glob("*.xml")):
        tree = ET.parse(xml_path)
        root = tree.getroot()

        filename = root.findtext("filename") or xml_path.stem
        size = root.find("size")
        w = int(size.findtext("width") or 0) if size is not None else 0
        h = int(size.findtext("height") or 0) if size is not None else 0

        anns: list[Annotation] = []
        for obj in root.findall("object"):
            cls_name = obj.findtext("name") or ""
            cls_id = name_to_id.setdefault(cls_name, len(name_to_id))
            bndbox = obj.find("bndbox")
            if bndbox is None:
                continue
            x1 = float(bndbox.findtext("xmin") or 0)
            y1 = float(bndbox.findtext("ymin") or 0)
            x2 = float(bndbox.findtext("xmax") or 0)
            y2 = float(bndbox.findtext("ymax") or 0)
            anns.append(
                Annotation(
                    task="detect",
                    class_id=cls_id,
                    class_name=cls_name,
                    bbox=[x1, y1, x2, y2],
                )
            )

        samples.append(
            AnnotationSample(
                image_path=filename,
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
    """Write Pascal VOC XML files to *target_dir*."""
    target_dir.mkdir(parents=True, exist_ok=True)
    for sample in samples:
        root = ET.Element("annotation")
        ET.SubElement(root, "folder").text = str(target_dir)
        ET.SubElement(root, "filename").text = Path(sample.image_path).name

        size = ET.SubElement(root, "size")
        ET.SubElement(size, "width").text = str(sample.width)
        ET.SubElement(size, "height").text = str(sample.height)
        ET.SubElement(size, "depth").text = "3"

        for ann in sample.annotations:
            if len(ann.bbox) != 4:
                continue
            obj = ET.SubElement(root, "object")
            ET.SubElement(obj, "name").text = ann.class_name
            ET.SubElement(obj, "pose").text = "Unspecified"
            ET.SubElement(obj, "truncated").text = "0"
            ET.SubElement(obj, "difficult").text = "0"
            bndbox = ET.SubElement(obj, "bndbox")
            x1, y1, x2, y2 = ann.bbox
            ET.SubElement(bndbox, "xmin").text = str(int(x1))
            ET.SubElement(bndbox, "ymin").text = str(int(y1))
            ET.SubElement(bndbox, "xmax").text = str(int(x2))
            ET.SubElement(bndbox, "ymax").text = str(int(y2))

        tree = ET.ElementTree(root)
        stem = Path(sample.image_path).stem
        ET.indent(tree, space="  ")
        tree.write(target_dir / f"{stem}.xml", encoding="unicode", xml_declaration=False)
