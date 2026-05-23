"""Albumentations augmentation pipelines for YOLO training.

Provides preset pipelines and a builder function so users can modify
augmentation strategies without touching internal YOLO code.
"""

from __future__ import annotations

from typing import Any, Optional


_PRESETS = frozenset(["light", "medium", "heavy", "microscopy", "tracking", "pose"])


def build_augmentation_pipeline(
    preset: str = "medium",
    image_size: int = 640,
    custom_transforms: Optional[list] = None,
):
    """Build and return an Albumentations ``Compose`` pipeline.

    Parameters
    ----------
    preset:
        One of ``'light'``, ``'medium'``, ``'heavy'``, ``'microscopy'``,
        ``'tracking'``, ``'pose'``.
    image_size:
        Target side length for resizing transforms.
    custom_transforms:
        List of additional Albumentations transforms appended to the preset.
        Pass ``[]`` for a completely empty pipeline (not recommended).

    Returns
    -------
    ``albumentations.Compose``
    """
    import albumentations as A

    if preset not in _PRESETS:
        raise ValueError(f"Unknown preset {preset!r}. Choose from {sorted(_PRESETS)}.")

    base = _get_preset_transforms(preset, image_size)
    extra = custom_transforms or []
    transforms = base + extra

    return A.Compose(
        transforms,
        bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]),
    )


def _get_preset_transforms(preset: str, image_size: int) -> list:
    import albumentations as A

    if preset == "light":
        return [
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.Resize(image_size, image_size),
        ]

    if preset == "medium":
        return [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.HueSaturationValue(p=0.3),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=10, p=0.5),
            A.Blur(blur_limit=3, p=0.1),
            A.Resize(image_size, image_size),
        ]

    if preset == "heavy":
        return [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=0.6),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=30, p=0.6),
            A.GaussianBlur(blur_limit=(3, 7), p=0.3),
            A.GaussNoise(p=0.3),
            A.CLAHE(p=0.3),
            A.RandomShadow(p=0.2),
            A.Resize(image_size, image_size),
        ]

    if preset == "microscopy":
        return [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.GaussianBlur(blur_limit=(3, 5), p=0.3),
            A.GaussNoise(p=0.4),
            A.CLAHE(clip_limit=4.0, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
            A.Resize(image_size, image_size),
        ]

    if preset == "tracking":
        return [
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.MotionBlur(blur_limit=5, p=0.2),
            A.Resize(image_size, image_size),
        ]

    if preset == "pose":
        return [
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.4),
            A.Resize(image_size, image_size),
        ]

    return [A.Resize(image_size, image_size)]


def apply_augmentation(
    pipeline,
    image: "np.ndarray",  # type: ignore[name-defined]
    bboxes: list | None = None,
    class_labels: list | None = None,
) -> dict[str, Any]:
    """Apply an Albumentations pipeline to an image + optional bboxes.

    Parameters
    ----------
    pipeline:
        ``albumentations.Compose`` instance.
    image:
        BGR or RGB numpy array.
    bboxes:
        YOLO-format bboxes ``[[x_c, y_c, w, h], ...]`` normalised 0–1.
    class_labels:
        Integer class labels matching each bbox.

    Returns
    -------
    dict with keys ``image``, ``bboxes``, ``class_labels``.
    """
    kwargs: dict[str, Any] = {"image": image}
    if bboxes is not None:
        kwargs["bboxes"] = bboxes
        kwargs["class_labels"] = class_labels or [0] * len(bboxes)

    return pipeline(**kwargs)
