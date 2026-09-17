# ultrawrapper

`ultrawrapper` es una capa simple para usar Ultralytics YOLO en flujos de trabajo con
Python, pandas y notebooks. La idea central es que las inferencias salgan como
`DataFrame` y que el ploteo sea directo, sin depender de objetos internos de
Ultralytics.

## Instalacion

Instalacion normal:

```bash
python -m venv .venv
source .venv/bin/activate
pip install .
```

Instalacion editable para desarrollo:

```bash
pip install -e ".[dev]"
```

Extras opcionales:

```bash
pip install -e ".[video]"      # Pillow para GIF
pip install -e ".[ml]"         # scikit-learn
pip install -e ".[explain]"    # SHAP para modelos tabulares
pip install -e ".[labeling]"   # X-AnyLabeling
pip install -e ".[all]"
```

El I/O de video principal usa OpenCV y no ejecuta `ffmpeg` mediante `subprocess`.

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
la imagen si se pasa `save_to`. Para ejecuciones por lotes o tests se puede usar
`show=False`.

## Ploteo disponible

- `plot_image_detections`: cajas, poligonos de segmentacion y keypoints.
- `plot_bounding_boxes`: cajas y labels.
- `plot_segmentation_masks`: poligonos de segmentacion, con cajas opcionales.
- `plot_pose_keypoints`: keypoints y skeleton COCO por defecto.
- `plot_class_distribution`: distribucion de clases.
- `plot_video_statistics`: detecciones por frame, confianza, clases y areas.
- `plot_tracking_trajectories`: trayectorias de objetos trackeados.
- `plot_training_metrics`, `plot_confusion_matrix`, `plot_precision_recall`, `plot_gpu_usage`.

## Video anotado

```python
from vision.yolo.video import write_annotated_video

out_path, df = write_annotated_video(
    model_path="yolo11n.pt",
    video_path="input.mp4",
    output_path="annotated.mp4",
    tracker="bytetrack.yaml",
    color_by="confidence",
    draw_tails=True,
    tail_length=30,
    save_predictions_to="predictions.parquet",
    return_predictions=True,
)
```

Si ya existe el `DataFrame` de predicciones:

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

Tambien se puede pedir tracking directamente en `predict_video`:

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

## Video I/O sin ffmpeg CLI

`vision.yolo.video_io` agrega utilidades simples basadas en OpenCV:

```python
from vision.yolo.video_io import extract_frames, images_to_video, video_info

print(video_info("input.mp4"))

frames = extract_frames(
    "input.mp4",
    output_dir="frames",
    every=30,
)

images_to_video(frames, "rebuilt.mp4", fps=10)
```

Funciones disponibles: `read_frames`, `write_frames`, `video_info`, `extract_frames`,
`images_to_video` e `images_to_gif`. Las secuencias de video se procesan de forma
iterativa; no se carga el video completo en RAM.

## Explainability

Las herramientas de explainability se mantienen desacopladas de capas privadas de
YOLO. `detection_density_map` resume espacialmente las detecciones y
`occlusion_sensitivity` funciona con cualquier predictor que devuelva un
`DataFrame` con una columna `confidence`.

```python
import cv2

from vision.yolo.explain import detection_density_map, occlusion_sensitivity, overlay_heatmap
from vision.yolo.infer import predict_image

image = cv2.imread("image.jpg")
df = predict_image("yolo11n.pt", image)

density = detection_density_map(df, image.shape)
overlay = overlay_heatmap(image, density)


def predictor(frame):
    return predict_image("yolo11n.pt", frame)

sensitivity = occlusion_sensitivity(image, predictor, patch_size=96)
```

SHAP se reserva para modelos tabulares downstream mediante `vision.yolo.ml.shap_values`.

## Machine learning desde inferencias

```python
import pandas as pd

from vision.yolo.ml import frame_features, merge_labels, prepare_xy, track_features

predictions = pd.read_parquet("predictions.parquet")
features = frame_features(predictions)

labels = pd.read_csv("frame_labels.csv")
table = merge_labels(features, labels, on="frame")
X, y = prepare_xy(table, target="label", drop=["frame"])
```

`track_features` agrega por objeto trackeado distancia recorrida, velocidad, numero
de observaciones y confianza. `frame_features` produce una fila numerica por frame.

## Notebooks

Los ejemplos estan en `notebooks/`:

1. `01_detection.ipynb`: deteccion basica y salida tabular.
2. `02_segmentation.ipynb`: segmentacion y poligonos.
3. `03_pose.ipynb`: pose y keypoints.
4. `04_tracking.ipynb`: tracking y trayectorias.
5. `05_training.ipynb`: entrenamiento y validacion.
6. `06_augmentations.ipynb`: Albumentations.
7. `07_export.ipynb`: exportacion y benchmarks.
8. `08_annotation_conversion.ipynb`: conversion de anotaciones.
9. `09_video_processing.ipynb`: procesamiento de video.
10. `10_dataframe_analysis.ipynb`: analisis directo del `DataFrame`.
11. `11_confidence_thresholds.ipynb`: efecto del threshold de confianza.
12. `12_tracking_kinematics.ipynb`: distancia y velocidad por track.
13. `13_explainability.ipynb`: density maps y occlusion sensitivity.
14. `14_ml_from_inferences.ipynb`: features y modelo tabular sencillo.
15. `15_video_io.ipynb`: extraccion de frames, video y GIF sin ffmpeg CLI.

Los notebooks nuevos usan codigo lineal, rutas como strings y no incluyen outputs grandes embebidos.

## Labeling with X-AnyLabeling

`ultrawrapper` does not vendor the X-AnyLabeling codebase; it treats it as an
optional UI installed via `pip`. That keeps this repository focused on being a
thin convenience layer on top of Ultralytics instead of embedding an external
desktop app.

The intended round-trip is:

```python
from pathlib import Path

from vision.yolo.infer import predict_image
from vision.yolo.annotations import convert_annotations
from vision.yolo.labeling import (
    export_image_predictions_to_xanylabeling,
    launch_xanylabeling,
)

df = predict_image("yolo11n.pt", "image.jpg", confidence=0.25)

json_path = export_image_predictions_to_xanylabeling(
    image_path="image.jpg",
    predictions=df,
    output_dir="prelabels",
)

launch_xanylabeling(
    filename="dataset/images",
    output_dir="dataset/labels_xany",
    labels=["person", "car"],
)

convert_annotations(
    source_dir="dataset/labels_xany",
    target_dir="dataset/labels_yolo",
    source_fmt="xanylabeling",
    target_fmt="yolo",
    class_names=["person", "car"],
    image_dir=Path("dataset/images"),
)
```

La integracion sigue siendo opcional para no incorporar el codigo GPL de X-AnyLabeling
dentro del proyecto MIT.

## Tests

```bash
python -m pytest -q
ruff check vision tests
```

Para instalar el entorno de desarrollo:

```bash
pip install -e ".[dev]"
```
