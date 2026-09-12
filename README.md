# ultrawrapper

`ultrawrapper` es una capa simple para usar Ultralytics YOLO en flujos de trabajo con Python, pandas y notebooks. La idea central es que las inferencias salgan como `DataFrame` y que el ploteo sea directo, sin depender de objetos internos de Ultralytics.

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

En Windows PowerShell, la activación del entorno es:

```powershell
.venv\Scripts\Activate.ps1
```

También se puede instalar directamente desde GitHub:

```bash
pip install "git+https://github.com/juanjosecas/ultrawrapper.git"
```

Los extras opcionales definidos en `pyproject.toml` se instalan, por ejemplo, con:

```bash
pip install ".[onnx]"
pip install ".[openvino]"
pip install ".[all]"
```

`requirements.txt` se mantiene como alternativa para crear un entorno con las dependencias base:

```bash
pip install -r requirements.txt
```

### Instalación para desarrollo

La instalación editable permite modificar el código del repositorio y probar los cambios sin reinstalar el paquete en cada edición.

```bash
git clone https://github.com/juanjosecas/ultrawrapper.git
cd ultrawrapper
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

El extra `dev` instala `pytest`, `ruff`, `jupyterlab` e `ipykernel` además del paquete y sus dependencias normales.

Verificación rápida:

```bash
python -m pytest -q
ruff check .
```

Para trabajar con los notebooks:

```bash
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

## Ploteo disponible

- `plot_image_detections`: cajas, polígonos de segmentación y keypoints en una sola imagen.
- `plot_bounding_boxes`: cajas y labels.
- `plot_segmentation_masks`: polígonos de segmentación, con cajas opcionales.
- `plot_pose_keypoints`: keypoints y skeleton COCO por defecto.
- `plot_class_distribution`: distribución de clases.
- `plot_video_statistics`: resumen de detecciones por frame, confianza, clases y áreas.
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
11. `11_confidence_thresholds.ipynb`: sensibilidad de resultados al threshold de confianza.
12. `12_tracking_kinematics.ipynb`: desplazamiento, distancia y velocidad a partir de tracks.

Las salidas generadas por los ejemplos se guardan en `notebooks/outputs/`.

## Tests

```bash
python -m pytest tests/test_vision_yolo.py -q
```
