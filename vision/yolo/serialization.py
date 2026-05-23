"""Serialization helpers – parquet, feather, CSV, JSON."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from vision.yolo.utils import to_serializable


def save_dataframe(
    df: pd.DataFrame,
    path: str | Path,
    fmt: str = "parquet",
) -> Path:
    """Save a DataFrame to disk in the requested format.

    Supported formats: ``parquet`` (default), ``feather``, ``csv``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "parquet":
        df.to_parquet(path, index=False)
    elif fmt == "feather":
        df.reset_index(drop=True).to_feather(path)
    elif fmt == "csv":
        df.to_csv(path, index=False)
    else:
        raise ValueError(f"Unsupported format: {fmt!r}. Use 'parquet', 'feather', or 'csv'.")

    return path


def load_dataframe(path: str | Path) -> pd.DataFrame:
    """Load a DataFrame from parquet, feather, or CSV based on file extension."""
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in (".feather", ".ipc", ".arrow"):
        return pd.read_feather(path)
    if suffix == ".csv":
        return pd.read_csv(path)

    raise ValueError(f"Cannot infer format from extension {suffix!r}.")


def append_to_parquet(df: pd.DataFrame, path: str | Path) -> Path:
    """Append rows to an existing parquet file (creates file if absent).

    Uses pandas concat – suitable for incremental video processing where
    the schema is fixed.
    """
    path = Path(path)
    if path.exists():
        existing = pd.read_parquet(path)
        df = pd.concat([existing, df], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def save_json(obj: Any, path: str | Path, indent: int = 2) -> Path:
    """Serialise an arbitrary object (including tensors/ndarrays) to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(to_serializable(obj), fh, indent=indent)
    return path


def load_json(path: str | Path) -> Any:
    """Load a JSON file."""
    with open(path) as fh:
        return json.load(fh)
