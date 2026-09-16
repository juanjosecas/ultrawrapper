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

Para usar `X-AnyLabeling` como interfaz de anotacion opcional:

```bash
pip install -e ".[labeling]"
```

Eso instala `x-anylabeling-cvhub[cpu]` como dependencia opcional. Si queres una
instalacion GPU o manejar su entorno por separado, instalalo manualmente segun la
documentacion oficial de X-AnyLabeling.

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

## Labeling con X-AnyLabeling

`ultrawrapper` no incorpora el codebase de X-AnyLabeling; lo usa como una
interfaz opcional de labeling instalada via `pip`. Esto evita acoplar el repo a
una GUI externa y mantiene a `ultrawrapper` como capa de conveniencia sobre
Ultralytics.

Segun la documentacion oficial de X-AnyLabeling, la integracion natural es:

- abrir una carpeta de imagenes o una imagen puntual desde `xanylabeling`;
- importar/exportar anotaciones en YOLO, VOC y COCO;
- cargar JSON nativos de X-AnyLabeling;
- usar auto-labeling y batch auto-labeling dentro de la app;
- reexportar el resultado para seguir procesandolo con `ultrawrapper`.

`ultrawrapper` ahora agrega helpers para ese flujo:

```python
from pathlib import Path

from vision.yolo.infer import predict_image
from vision.yolo.annotations import convert_annotations
from vision.yolo.labeling import (
    export_image_predictions_to_xanylabeling,
    launch_xanylabeling,
)

# 1) correr inferencia con ultrawrapper
df = predict_image("yolo11n.pt", "image.jpg", confidence=0.25)

# 2) exportar predicciones como pre-labels editables en X-AnyLabeling
json_path = export_image_predictions_to_xanylabeling(
    image_path="image.jpg",
    predictions=df,
    output_dir="prelabels",
)

# 3) abrir la interfaz apuntando a la carpeta de imagenes
launch_xanylabeling(
    filename="dataset/images",
    output_dir="dataset/labels_xany",
    labels=["person", "car"],
)

# 4) volver a traer anotaciones editadas al formato que necesites
convert_annotations(
    source_dir="dataset/labels_xany",
    target_dir="dataset/labels_yolo",
    source_fmt="xanylabeling",
    target_fmt="yolo",
    class_names=["person", "car"],
)
```

Tambien podes convertir datasets ya existentes hacia un formato que X-AnyLabeling
entiende:

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

Notas practicas:

- El helper de exportacion a X-AnyLabeling cubre bien deteccion y segmentacion
  porque salen naturalmente del `DataFrame` actual.
- X-AnyLabeling si soporta flujos de auto-labeling y carga de anotaciones ya
  hechas, asi que no hace falta modificar Ultralytics ni clonar su repo para
  aprovecharlo.
- Para pose/formatos avanzados conviene seguir usando la conversion nativa de
  X-AnyLabeling si necesitás conservar todos sus metadatos de grouping.
- X-AnyLabeling esta licenciado bajo GPL-3.0; por eso la integracion aca queda
  como dependencia opcional y no como codigo embebido en este repo MIT.

## Tests

```bash
python -m pytest tests/test_vision_yolo.py -q
```

Si el entorno no tiene `pytest`, instala las dependencias de desarrollo antes de correrlos.
