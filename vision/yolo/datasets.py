"""Dataset utilities – listing, splitting, and basic validation."""

from __future__ import annotations

import random
import shutil
from pathlib import Path
from typing import Optional

import pandas as pd


def list_dataset_images(
    dataset_dir: str | Path,
    extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"),
) -> list[Path]:
    """Return all image paths inside a directory tree."""
    root = Path(dataset_dir)
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in extensions)


def dataset_summary(dataset_dir: str | Path) -> pd.DataFrame:
    """Return a DataFrame with per-split image and label counts."""
    root = Path(dataset_dir)
    rows = []
    for split in ("train", "val", "test"):
        images_dir = root / "images" / split
        labels_dir = root / "labels" / split
        n_images = len(list(images_dir.glob("*"))) if images_dir.exists() else 0
        n_labels = len(list(labels_dir.glob("*.txt"))) if labels_dir.exists() else 0
        rows.append({"split": split, "images": n_images, "labels": n_labels})
    return pd.DataFrame(rows)


def split_dataset(
    source_images: list[Path],
    source_labels: list[Path],
    output_dir: str | Path,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    seed: int = 42,
    copy: bool = True,
) -> dict[str, list[Path]]:
    """Split image/label pairs into train/val/test subsets.

    Parameters
    ----------
    source_images / source_labels:
        Parallel lists of image and label paths.
    output_dir:
        Root directory; files are placed in ``images/{split}`` and
        ``labels/{split}``.
    train_ratio / val_ratio:
        Fractions for train and val; the rest goes to test.
    copy:
        When True files are copied; False creates symlinks instead.

    Returns
    -------
    dict mapping split name to list of output image paths.
    """
    assert len(source_images) == len(source_labels), "Images and labels must be paired."
    assert train_ratio + val_ratio <= 1.0

    rng = random.Random(seed)
    pairs = list(zip(source_images, source_labels))
    rng.shuffle(pairs)

    n = len(pairs)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    splits = {
        "train": pairs[:n_train],
        "val": pairs[n_train : n_train + n_val],
        "test": pairs[n_train + n_val :],
    }

    output = Path(output_dir)
    result: dict[str, list[Path]] = {}
    op = shutil.copy2 if copy else Path.symlink_to

    for split, items in splits.items():
        img_dir = output / "images" / split
        lbl_dir = output / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        out_images: list[Path] = []
        for img_path, lbl_path in items:
            dst_img = img_dir / img_path.name
            dst_lbl = lbl_dir / lbl_path.name
            if copy:
                shutil.copy2(img_path, dst_img)
                shutil.copy2(lbl_path, dst_lbl)
            else:
                dst_img.symlink_to(img_path.resolve())
                dst_lbl.symlink_to(lbl_path.resolve())
            out_images.append(dst_img)

        result[split] = out_images

    return result


def write_data_yaml(
    output_dir: str | Path,
    class_names: list[str],
    train: str = "images/train",
    val: str = "images/val",
    test: str = "images/test",
) -> Path:
    """Write a YOLO ``data.yaml`` configuration file."""
    import yaml  # pyyaml bundled with ultralytics

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = output_dir / "data.yaml"

    cfg = {
        "path": str(output_dir.resolve()),
        "train": train,
        "val": val,
        "test": test,
        "nc": len(class_names),
        "names": class_names,
    }
    with open(yaml_path, "w") as fh:
        yaml.safe_dump(cfg, fh, default_flow_style=False)

    return yaml_path
