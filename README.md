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

annotated_path = write_annotated_video_from_dataframe(
    video_path="input.mp4",
    predictions=df,  # o "predictions.parquet"
    output_path="annotated_from_df.mp4",
    color_by="confidence",
    draw_tails=False,
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

## Tests

```bash
python -m pytest tests/test_vision_yolo.py -q
```

Si el entorno no tiene `pytest`, instala las dependencias de desarrollo antes de correrlos.
