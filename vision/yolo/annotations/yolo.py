"""YOLO annotation reader and writer.

YOLO label format::

    <class_id> <x_center> <y_center> <width> <height>

All values are normalised to [0, 1].
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from vision.yolo.annotations.internal import Annotation, AnnotationSample


def read(
    source_dir: Path,
    class_names: Optional[list[str]] = None,
    image_dir: Optional[Path] = None,
    **kwargs,
) -> list[AnnotationSample]:
    """Read YOLO-format labels from *source_dir*.

    Looks for ``*.txt`` label files and matches them to images in
    *image_dir* (defaults to a sibling ``images/`` directory).
    """
    label_dir = source_dir
    img_dir = image_dir or source_dir.parent / "images"
    _IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")

    samples: list[AnnotationSample] = []
    for lbl_path in sorted(label_dir.glob("*.txt")):
        # Find matching image
        img_path: Optional[Path] = None
        for ext in _IMAGE_EXTS:
            candidate = img_dir / (lbl_path.stem + ext)
            if candidate.exists():
                img_path = candidate
                break

        w, h = _get_image_size(img_path)
        anns: list[Annotation] = []

        with open(lbl_path) as fh:
            for line in fh:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls_id = int(parts[0])
                xc, yc, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                x1 = (xc - bw / 2) * w
                y1 = (yc - bh / 2) * h
                x2 = (xc + bw / 2) * w
                y2 = (yc + bh / 2) * h
                cls_name = class_names[cls_id] if class_names and cls_id < len(class_names) else str(cls_id)
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
                image_path=str(img_path) if img_path else "",
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
    """Write YOLO-format labels to *target_dir*."""
    target_dir.mkdir(parents=True, exist_ok=True)
    for sample in samples:
        stem = Path(sample.image_path).stem if sample.image_path else "unknown"
        lbl_path = target_dir / f"{stem}.txt"
        w = sample.width or 1
        h = sample.height or 1
        lines: list[str] = []
        for ann in sample.annotations:
            if len(ann.bbox) != 4:
                continue
            x1, y1, x2, y2 = ann.bbox
            xc = ((x1 + x2) / 2) / w
            yc = ((y1 + y2) / 2) / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            lines.append(f"{ann.class_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
        with open(lbl_path, "w") as fh:
            fh.write("\n".join(lines))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_image_size(img_path: Optional[Path]) -> tuple[int, int]:
    if img_path is None or not img_path.exists():
        return 1, 1
    try:
        import cv2

        img = cv2.imread(str(img_path))
        if img is not None:
            return img.shape[1], img.shape[0]
    except Exception:
        pass
    return 1, 1
