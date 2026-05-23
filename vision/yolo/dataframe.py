"""DataFrame construction from Ultralytics results.

The single stable adapter between Ultralytics internals and the rest of
the library.  All other modules should call ``ultralytics_result_to_dataframe``
rather than touching result.boxes / result.masks directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd

from vision.yolo.utils import tensor_to_numpy


# ---------------------------------------------------------------------------
# Stable result dataclass
# ---------------------------------------------------------------------------

@dataclass
class PredictionResult:
    frame: int = 0
    track_id: Optional[int] = None
    class_id: int = 0
    class_name: str = ""
    confidence: float = 0.0
    xmin: float = 0.0
    ymin: float = 0.0
    xmax: float = 0.0
    ymax: float = 0.0
    polygon: Optional[list] = None
    mask: Optional[Any] = None
    keypoints: Optional[list] = None
    timestamp: Optional[float] = None


# ---------------------------------------------------------------------------
# Core adapter
# ---------------------------------------------------------------------------

def ultralytics_result_to_dataframe(
    result,
    frame_idx: int = 0,
    timestamp: Optional[float] = None,
) -> pd.DataFrame:
    """Convert one Ultralytics ``Results`` object to a tidy DataFrame.

    Parameters
    ----------
    result:
        A single ``ultralytics.engine.results.Results`` instance.
    frame_idx:
        Frame number (for video/batch contexts).
    timestamp:
        Optional wall-clock or video timestamp in seconds.

    Returns
    -------
    pd.DataFrame with columns matching :class:`PredictionResult`.
    """
    rows: list[dict] = []

    if result is None:
        return _empty_dataframe()

    names: dict[int, str] = result.names or {}
    boxes = result.boxes

    if boxes is None or len(boxes) == 0:
        return _empty_dataframe()

    xyxy = tensor_to_numpy(boxes.xyxy)
    confs = tensor_to_numpy(boxes.conf)
    cls_ids = tensor_to_numpy(boxes.cls).astype(int)
    track_ids_raw = boxes.id
    track_ids = tensor_to_numpy(track_ids_raw).astype(int) if track_ids_raw is not None else None

    # Masks (segmentation)
    masks_data = None
    polygons_data = None
    if result.masks is not None:
        try:
            masks_data = tensor_to_numpy(result.masks.data)
            polygons_data = result.masks.xy  # list of (N,2) arrays
        except Exception:
            pass

    # Keypoints (pose)
    keypoints_data = None
    if result.keypoints is not None:
        try:
            keypoints_data = tensor_to_numpy(result.keypoints.data)
        except Exception:
            pass

    for i in range(len(xyxy)):
        x1, y1, x2, y2 = xyxy[i]
        row: dict[str, Any] = {
            "frame": frame_idx,
            "track_id": int(track_ids[i]) if track_ids is not None else None,
            "class_id": int(cls_ids[i]),
            "class_name": names.get(int(cls_ids[i]), str(int(cls_ids[i]))),
            "confidence": float(confs[i]),
            "xmin": float(x1),
            "ymin": float(y1),
            "xmax": float(x2),
            "ymax": float(y2),
            "polygon": polygons_data[i].tolist() if polygons_data is not None else None,
            "mask": masks_data[i].tolist() if masks_data is not None else None,
            "keypoints": keypoints_data[i].tolist() if keypoints_data is not None else None,
            "timestamp": timestamp,
        }
        rows.append(row)

    if not rows:
        return _empty_dataframe()

    return pd.DataFrame(rows)


def _empty_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "frame", "track_id", "class_id", "class_name", "confidence",
            "xmin", "ymin", "xmax", "ymax", "polygon", "mask", "keypoints",
            "timestamp",
        ]
    )


def concat_results(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate a list of result DataFrames, resetting the index."""
    if not dfs:
        return _empty_dataframe()
    non_empty = [df for df in dfs if not df.empty]
    if not non_empty:
        return _empty_dataframe()
    return pd.concat(non_empty, ignore_index=True)
