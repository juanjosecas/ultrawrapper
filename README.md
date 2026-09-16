# ultrawrapper

`ultrawrapper` es una capa simple para usar Ultralytics YOLO en flujos de trabajo con
Python, pandas y notebooks. La idea central es que las inferencias salgan como
`DataFrame` y que el ploteo sea directo, sin depender de objetos internos de
Ultralytics.

## Instalacion

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Para desarrollo local:

```bash
pip install -e .
```

To use `X-AnyLabeling` as an optional labeling UI:

```bash
pip install -e ".[labeling]"
```

This installs `x-anylabeling-cvhub[cpu]` as an optional dependency. If you need
GPU support or want to manage its environment separately, install it manually
following the official X-AnyLabeling documentation.

## Uso rapido

```python
from vision.yolo.infer import predict_image
from vision.yolo.plotting import plot_image_detections, plot_class_distribution

MODEL = "yolo11n.pt"
IMAGE = "https://ultralytics.com/images/bus.jpg"

df = predict_image(MODEL, IMAGE, confidence=0.25)
print(df.head())

plot_image_detections(IMAGE, df, save_to="detections.png")
plot_class_distribution(df, save_to="classes.png")
```

Las funciones de `vision.yolo.plotting` muestran el grafico por defecto y guardan
la imagen si se pasa `save_to`. No hace falta hacer `fig = ...` ni `fig.show()`.
Para ejecuciones por lotes o tests se puede usar `show=False`.

## Ploteo disponible

- `plot_image_detections`: cajas, poligonos de segmentacion y keypoints en una sola imagen.
- `plot_bounding_boxes`: solo cajas y labels.
- `plot_segmentation_masks`: poligonos de segmentacion, con cajas opcionales.
- `plot_pose_keypoints`: keypoints y skeleton COCO por defecto.
- `plot_class_distribution`: distribucion de clases.
- `plot_video_statistics`: resumen de detecciones por frame, confianza, clases y areas.
- `plot_tracking_trajectories`: trayectorias de objetos trackeados.
- `plot_training_metrics`, `plot_confusion_matrix`, `plot_precision_recall`, `plot_gpu_usage`.

## Video anotado

Para cargar un video, superponer predicciones y escribir un MP4 anotado:

```python
from vision.yolo.video import write_annotated_video

out_path, df = write_annotated_video(
    model_path="yolo11n.pt",
    video_path="input.mp4",
    output_path="annotated.mp4",
    tracker="bytetrack.yaml",      # activa track_id y permite dibujar tails
    color_by="confidence",         # tambien: "class" o "track_id"
    draw_tails=True,
    tail_length=30,
    save_predictions_to="predictions.parquet",
    return_predictions=True,
)
```

Tambien se puede pasar un `predictions_df` ya calculado para dibujar sin volver a
correr el modelo.

Si ya corriste `predict_video` y tenes el `DataFrame`:

```python
from vision.yolo.video import write_annotated_video_from_dataframe
from vision.yolo.track import track_detections_dataframe

tracked_df = track_detections_dataframe(
    df,
    iou_threshold=0.3,
    max_frame_gap=1,
    same_class_only=True,
)

annotated_path = write_annotated_video_from_dataframe(
    video_path="input.mp4",
    predictions=tracked_df,
    output_path="annotated_from_df.mp4",
    color_by="confidence",
    draw_tails=True,
)
```

Para acelerar el dibujo cuando no necesitás tails:

```python
annotated_path = write_annotated_video_from_dataframe(
    video_path="input.mp4",
    predictions=df,
    output_path="annotated_parallel.mp4",
    color_by="confidence",
    draw_tails=False,
    annotation_workers=4,
    annotation_batch_size=32,
)
```

Tambien podés pedir tracking directamente en `predict_video`:

```python
from vision.yolo.infer import predict_video

tracked_df = predict_video(
    "yolo11n.pt",
    "input.mp4",
    confidence=0.25,
    batch_size=8,
    tracker="bytetrack.yaml",
    tracker_config={
        "track_high_thresh": 0.35,
        "track_low_thresh": 0.1,
        "new_track_thresh": 0.35,
        "track_buffer": 60,
        "match_thresh": 0.8,
    },
    save_to="predictions_tracked.parquet",
)
```

## Notebooks

Los ejemplos estan en `vision/yolo/notebooks`:

1. `01_detection.ipynb`: deteccion, batch, filtrado y guardado de plots.
2. `02_segmentation.ipynb`: segmentacion, poligonos y filtros por confianza.
3. `03_pose.ipynb`: pose, keypoints, skeleton y conversion a tabla larga.
4. `04_tracking.ipynb`: tracking, estadisticas y trayectorias.
5. `05_training.ipynb`: entrenamiento, validacion y metricas.
6. `06_augmentations.ipynb`: Albumentations para imagenes/cajas.
7. `07_export.ipynb`: exportacion y benchmarks.
8. `08_annotation_conversion.ipynb`: conversion COCO/YOLO/VOC/LabelMe.
9. `09_video_processing.ipynb`: procesamiento de video por lotes.

Los notebooks guardan salidas de ejemplo en `vision/yolo/notebooks/outputs/`.

## Labeling with X-AnyLabeling

`ultrawrapper` does not vendor the X-AnyLabeling codebase; it treats it as an
optional UI installed via `pip`. That keeps this repository focused on being a
thin convenience layer on top of Ultralytics instead of embedding an external
desktop app.

Based on the official X-AnyLabeling documentation, the natural integration path
is:

- open an image directory or a single image from `xanylabeling`;
- import and export annotations in YOLO, VOC, and COCO;
- load native X-AnyLabeling JSON files;
- use built-in auto-labeling and batch auto-labeling inside the app;
- export the reviewed result back into `ultrawrapper`.

`ultrawrapper` now adds helpers for that round-trip:

```python
from pathlib import Path

from vision.yolo.infer import predict_image
from vision.yolo.annotations import convert_annotations
from vision.yolo.labeling import (
    export_image_predictions_to_xanylabeling,
    launch_xanylabeling,
)

# 1) run inference with ultrawrapper
df = predict_image("yolo11n.pt", "image.jpg", confidence=0.25)

# 2) export predictions as editable pre-labels for X-AnyLabeling
json_path = export_image_predictions_to_xanylabeling(
    image_path="image.jpg",
    predictions=df,
    output_dir="prelabels",
)

# 3) open the UI against the image directory
launch_xanylabeling(
    filename="dataset/images",
    output_dir="dataset/labels_xany",
    labels=["person", "car"],
)

# 4) bring edited annotations back into the format you need
convert_annotations(
    source_dir="dataset/labels_xany",
    target_dir="dataset/labels_yolo",
    source_fmt="xanylabeling",
    target_fmt="yolo",
    class_names=["person", "car"],
    image_dir=Path("dataset/images"),
)
```

You can also convert an existing dataset into a format X-AnyLabeling can edit:

```python
convert_annotations(
    source_dir="dataset/labels_yolo",
    target_dir="dataset/labels_xany",
    source_fmt="yolo",
    target_fmt="xanylabeling",
    class_names=["person", "car"],
    image_dir=Path("dataset/images"),
)
```

Practical notes:

- The export helper covers detection and segmentation well because both map
  cleanly from the current `DataFrame` schema.
- X-AnyLabeling already supports auto-labeling and importing existing
  annotations, so there is no need to modify Ultralytics or vendor its repo.
- For pose or more advanced formats, prefer X-AnyLabeling's native conversion
  tools when you need to preserve all grouping metadata.
- X-AnyLabeling is licensed under GPL-3.0, so this repo keeps the integration
  optional instead of embedding GPL code inside an MIT project.

## Tests

```bash
python -m pytest tests/test_vision_yolo.py -q
```

Si el entorno no tiene `pytest`, instala las dependencias de desarrollo antes de correrlos.
