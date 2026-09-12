# ultrawrapper

`ultrawrapper` es una capa simple para usar Ultralytics YOLO en flujos de trabajo con Python, pandas y notebooks. La idea central es que las inferencias salgan como `DataFrame` y que el ploteo, el análisis y el uso posterior en machine learning sean directos, sin depender de objetos internos de Ultralytics.

## Instalación

### Instalación normal desde el repositorio

```bash
git clone https://github.com/juanjosecas/ultrawrapper.git
cd ultrawrapper
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install .
```

En Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

También se puede instalar directamente desde GitHub:

```bash
pip install "git+https://github.com/juanjosecas/ultrawrapper.git"
```

Extras opcionales:

```bash
pip install ".[ml]"       # scikit-learn
pip install ".[explain]"  # scikit-learn + SHAP
pip install ".[onnx]"
pip install ".[openvino]"
pip install ".[all]"
```

`requirements.txt` se mantiene como alternativa para instalar las dependencias base:

```bash
pip install -r requirements.txt
```

### Instalación para desarrollo

La instalación editable permite modificar el código y probar cambios sin reinstalar el paquete.

```bash
git clone https://github.com/juanjosecas/ultrawrapper.git
cd ultrawrapper
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

Para desarrollo con machine learning y explicabilidad:

```bash
pip install -e ".[dev,ml,explain]"
```

El extra `dev` instala `pytest`, `ruff`, `jupyterlab` e `ipykernel`.

```bash
python -m pytest -q
ruff check .
jupyter lab
```

Los notebooks asumen que Jupyter se inicia desde la raíz del repositorio.

## Uso rápido

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

Las funciones de `vision.yolo.plotting` muestran el gráfico por defecto y guardan la imagen si se pasa `save_to`. Para ejecuciones por lotes o tests se puede usar `show=False`.

## Video sin comandos externos de ffmpeg

El módulo `vision.yolo.video_io` usa OpenCV desde Python. No ejecuta `ffmpeg` mediante `subprocess` ni requiere construir comandos de consola.

Esto no significa que la compresión de video sea Python puro: MP4/H.264 y otros codecs siempre dependen de bibliotecas nativas disponibles para OpenCV.

```python
from vision.yolo.video_io import video_info, read_frames, extract_frames

print(video_info("input.mp4"))

for frame_index, timestamp, frame in read_frames(
    "input.mp4",
    step=10,
    max_frames=20,
):
    print(frame_index, timestamp, frame.shape)

files = extract_frames(
    "input.mp4",
    output_dir="frames",
    every_n_frames=30,
)
```

También se puede reconstruir un video a partir de imágenes:

```python
from vision.yolo.video_io import images_to_video

images_to_video(
    files,
    output_path="rebuilt.mp4",
    fps=10,
    codec="mp4v",
)
```

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

Si las predicciones ya están calculadas:

```python
from vision.yolo.video import write_annotated_video_from_dataframe

annotated_path = write_annotated_video_from_dataframe(
    video_path="input.mp4",
    predictions=df,
    output_path="annotated_from_df.mp4",
    color_by="confidence",
    draw_tails=False,
)
```

## Explicabilidad y mapas de densidad

`detection_density_map` resume espacialmente dónde aparecen detecciones. Puede ponderarlas por confianza y filtrar una clase específica.

```python
import cv2
from vision.yolo.explain import detection_density_map, plot_density_map

image = cv2.imread("image.jpg")
height, width = image.shape[:2]

density = detection_density_map(
    df,
    image_width=width,
    image_height=height,
    bins=32,
    weight_by_confidence=True,
)

plot_density_map(density, image=image)
```

Para responder de forma más directa a "qué regiones de la imagen sostienen esta detección" se agregó `occlusion_sensitivity`. Divide la imagen en una grilla, oculta una región por vez y cuantifica cuánto cae la confianza del detector.

```python
from vision.yolo.explain import occlusion_sensitivity, plot_density_map

heatmap = occlusion_sensitivity(
    "yolo11n.pt",
    image,
    target_class="person",
    grid_size=8,
)

plot_density_map(heatmap, image=image)
```

Es más lento que un método basado en gradientes, pero no depende de nombres de capas internas de Ultralytics y por eso es más estable entre versiones.

SHAP se reserva para modelos tabulares entrenados sobre features derivadas de las inferencias. Aplicar SHAP directamente sobre un detector YOLO requiere definir cuidadosamente qué salida se quiere explicar y suele depender demasiado de detalles internos del modelo.

## Machine learning con las inferencias

Las predicciones de YOLO tienen una fila por detección. `vision.yolo.ml` permite transformarlas a una tabla directamente utilizable por pandas/scikit-learn.

Features por frame:

```python
from vision.yolo.ml import frame_features

features = frame_features(df)
print(features.head())
```

Incluye, entre otras variables:

- número de detecciones por frame;
- confianza media, máxima y desviación;
- área media y máxima de bounding boxes;
- posición media de los centroides;
- conteo por clase (`count_person`, `count_mouse`, etc.).

Para tracking puede obtenerse una fila por `track_id`:

```python
from vision.yolo.ml import track_features

tracks = track_features(df)
```

Esto agrega duración del track, distancia recorrida, velocidad media/máxima, área media y confianza.

Para combinar las inferencias con metadata experimental o etiquetas:

```python
from vision.yolo.ml import merge_labels, prepare_xy

labels = ...  # DataFrame con frame + variable objetivo
dataset = merge_labels(features, labels, on="frame")
X, y = prepare_xy(dataset, target="condition", drop_columns=["frame", "timestamp"])
```

A partir de ahí se usa scikit-learn normalmente:

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=300, random_state=42)
model.fit(X, y)
```

SHAP para ese modelo tabular:

```python
from vision.yolo.ml import shap_values

explanation = shap_values(model, X, max_samples=200)
```

## Ploteo disponible

- `plot_image_detections`: cajas, polígonos de segmentación y keypoints.
- `plot_bounding_boxes`: cajas y labels.
- `plot_segmentation_masks`: polígonos de segmentación, con cajas opcionales.
- `plot_pose_keypoints`: keypoints y skeleton COCO.
- `plot_class_distribution`: distribución de clases.
- `plot_video_statistics`: resumen por frame, confianza, clases y áreas.
- `plot_tracking_trajectories`: trayectorias de objetos trackeados.
- `plot_training_metrics`, `plot_confusion_matrix`, `plot_precision_recall`, `plot_gpu_usage`.
- `plot_density_map`: mapas de densidad y mapas de sensibilidad superpuestos a una imagen.

## Notebooks

Los ejemplos están en `notebooks/`. El código está escrito de forma deliberadamente lineal: rutas como strings con `os.path`, variables visibles y funciones auxiliares solo cuando existe reutilización real.

1. `01_detection.ipynb`: detección de objetos, filtros y batch de imágenes.
2. `02_segmentation.ipynb`: segmentación y filtrado por confianza.
3. `03_pose.ipynb`: pose estimation e inspección de keypoints.
4. `04_tracking.ipynb`: tracking, estadísticas y trayectorias.
5. `05_training.ipynb`: entrenamiento, reanudación, validación y métricas.
6. `06_augmentations.ipynb`: Albumentations y comparación de presets.
7. `07_export.ipynb`: exportación y benchmarking.
8. `08_annotation_conversion.ipynb`: conversión entre formatos de anotación.
9. `09_video_processing.ipynb`: inferencia de video y generación de video anotado.
10. `10_dataframe_analysis.ipynb`: análisis tabular de detecciones con pandas.
11. `11_confidence_thresholds.ipynb`: sensibilidad al threshold de confianza.
12. `12_tracking_kinematics.ipynb`: desplazamiento, distancia y velocidad de tracks.
13. `13_explainability.ipynb`: mapas de densidad y occlusion sensitivity.
14. `14_ml_from_inferences.ipynb`: features, etiquetas experimentales, scikit-learn y SHAP.
15. `15_video_io.ipynb`: lectura, extracción de frames y escritura de video desde Python.

Las salidas generadas por los ejemplos se guardan en `notebooks/outputs/`.

## Tests

```bash
python -m pytest -q
```
