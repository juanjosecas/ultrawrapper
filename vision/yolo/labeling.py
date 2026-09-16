"""Helpers to integrate X-AnyLabeling as an optional labeling UI."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Sequence

import pandas as pd

_X_ANYLABELING_INSTALL_HINT = (
    "Install it with `pip install -e \".[labeling]\"` or "
    "`pip install \"x-anylabeling-cvhub[cpu]>=4.0.6\"`."
)


def is_xanylabeling_available(executable: str = "xanylabeling") -> bool:
    """Return True when the X-AnyLabeling executable is available."""
    return shutil.which(executable) is not None


def launch_xanylabeling(
    filename: str | Path,
    output_dir: str | Path | None = None,
    labels: str | Path | Sequence[str] | None = None,
    autosave: bool = True,
    store_image_data: bool = False,
    sort_labels: bool = True,
    keep_prev: bool = False,
    executable: str = "xanylabeling",
    extra_args: Sequence[str] | None = None,
) -> subprocess.Popen:
    """Launch X-AnyLabeling against an image file or directory."""
    executable_path = shutil.which(executable)
    if executable_path is None:
        raise FileNotFoundError(
            f"X-AnyLabeling executable {executable!r} was not found. "
            f"{_X_ANYLABELING_INSTALL_HINT}"
        )

    command = [executable_path, "--filename", str(filename)]
    if output_dir is not None:
        command.extend(["--output", str(output_dir)])

    labels_arg = _normalize_labels_argument(labels)
    if labels_arg is not None:
        command.extend(["--labels", labels_arg])

    if autosave:
        command.append("--autosave")
    if not store_image_data:
        command.append("--nodata")
    if not sort_labels:
        command.append("--nosortlabels")
    if keep_prev:
        command.append("--keep-prev")
    if extra_args:
        command.extend(extra_args)

    return subprocess.Popen(command)


def export_image_predictions_to_xanylabeling(
    image_path: str | Path,
    predictions: pd.DataFrame,
    output_dir: str | Path | None = None,
    checked: bool = False,
    tags: Sequence[str] | None = None,
    description: str = "",
) -> Path:
    """Export predictions for one image to an X-AnyLabeling JSON file."""
    image_path = Path(image_path)
    output_dir = Path(output_dir) if output_dir is not None else image_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    width, height = _get_image_size(image_path)
    payload = {
        "version": "x-anylabeling",
        "flags": {},
        "tags": list(tags or []),
        "shapes": _dataframe_to_shapes(predictions),
        "description": description,
        "imagePath": image_path.name,
        "imageData": None,
        "imageHeight": height,
        "imageWidth": width,
        "checked": checked,
    }

    output_path = output_dir / f"{image_path.stem}.json"
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    return output_path


def export_predictions_to_xanylabeling(
    image_paths: Sequence[str | Path],
    predictions: pd.DataFrame,
    output_dir: str | Path,
    index_column: str = "frame",
    image_ids: Sequence[str | int] | None = None,
) -> list[Path]:
    """Export batched detection/segmentation predictions to X-AnyLabeling JSON."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_paths = list(image_paths)

    if index_column not in predictions.columns:
        raise ValueError(f"Predictions DataFrame must include the {index_column!r} column.")

    if image_ids is None:
        image_ids = list(range(len(image_paths)))
    elif len(image_ids) != len(image_paths):
        raise ValueError("image_ids and image_paths must have the same length.")

    exported: list[Path] = []
    for image_id, image_path in zip(image_ids, image_paths):
        image_predictions = predictions[predictions[index_column] == image_id]
        exported.append(
            export_image_predictions_to_xanylabeling(
                image_path=image_path,
                predictions=image_predictions,
                output_dir=output_dir,
            )
        )
    return exported


def _normalize_labels_argument(labels: str | Path | Sequence[str] | None) -> str | None:
    if labels is None:
        return None
    if isinstance(labels, (str, Path)):
        return str(labels)
    return ",".join(str(label) for label in labels)


def _dataframe_to_shapes(predictions: pd.DataFrame) -> list[dict]:
    shapes: list[dict] = []

    for row in predictions.to_dict(orient="records"):
        label = row.get("class_name") or str(row.get("class_id", ""))
        confidence = row.get("confidence")
        polygon = row.get("polygon")

        if _has_polygon(polygon):
            shapes.append(
                {
                    "label": label,
                    "score": float(confidence) if confidence is not None else None,
                    "points": polygon,
                    "group_id": None,
                    "description": "",
                    "difficult": False,
                    "shape_type": "polygon",
                    "flags": {},
                    "attributes": {},
                }
            )
            continue

        bbox = [row.get("xmin"), row.get("ymin"), row.get("xmax"), row.get("ymax")]
        if any(value is None or pd.isna(value) for value in bbox):
            continue

        shapes.append(
            {
                "label": label,
                "score": float(confidence) if confidence is not None else None,
                "points": [[float(bbox[0]), float(bbox[1])], [float(bbox[2]), float(bbox[3])]],
                "group_id": None,
                "description": "",
                "difficult": False,
                "shape_type": "rectangle",
                "flags": {},
                "attributes": {},
            }
        )

    return shapes


def _has_polygon(value) -> bool:
    return isinstance(value, list) and len(value) >= 3


def _get_image_size(image_path: Path) -> tuple[int, int]:
    try:
        import cv2

        image = cv2.imread(str(image_path))
        if image is not None:
            return int(image.shape[1]), int(image.shape[0])
    except Exception:
        pass
    raise ValueError(f"Could not read image dimensions from {image_path}.")
