import numpy as np
import pandas as pd

from vision.yolo.explain import detection_density_map, occlusion_sensitivity
from vision.yolo.ml import frame_features, track_features


def test_detection_density_map_is_normalized():
    detections = pd.DataFrame(
        {
            "xmin": [10.0, 40.0],
            "ymin": [10.0, 40.0],
            "xmax": [20.0, 50.0],
            "ymax": [20.0, 50.0],
            "confidence": [0.9, 0.5],
        }
    )

    heatmap = detection_density_map(detections, (64, 64), sigma=3.0)

    assert heatmap.shape == (64, 64)
    assert heatmap.min() >= 0.0
    assert heatmap.max() <= 1.0
    assert np.isclose(heatmap.max(), 1.0)


def test_occlusion_sensitivity_finds_supporting_region():
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    image[:16, :16] = 255

    def predictor(frame):
        confidence = float(frame[:16, :16].mean() / 255.0)
        if confidence == 0:
            return pd.DataFrame(columns=["confidence"])
        return pd.DataFrame({"confidence": [confidence]})

    heatmap = occlusion_sensitivity(image, predictor, patch_size=16, stride=16)

    assert heatmap[:16, :16].mean() > heatmap[16:, 16:].mean()


def test_frame_features_summarizes_each_frame():
    detections = pd.DataFrame(
        {
            "frame": [0, 0, 1],
            "class_id": [0, 1, 0],
            "confidence": [0.8, 0.6, 0.9],
            "xmin": [0.0, 10.0, 0.0],
            "ymin": [0.0, 10.0, 0.0],
            "xmax": [10.0, 20.0, 20.0],
            "ymax": [10.0, 20.0, 10.0],
        }
    )

    features = frame_features(detections)

    assert features["frame"].tolist() == [0, 1]
    assert features["n_detections"].tolist() == [2, 1]
    assert features.loc[0, "n_classes"] == 2
    assert np.isclose(features.loc[1, "mean_box_area"], 200.0)


def test_track_features_calculates_distance_and_speed():
    detections = pd.DataFrame(
        {
            "frame": [0, 1, 2],
            "track_id": [7, 7, 7],
            "confidence": [0.9, 0.8, 0.85],
            "xmin": [0.0, 3.0, 6.0],
            "ymin": [0.0, 4.0, 8.0],
            "xmax": [2.0, 5.0, 8.0],
            "ymax": [2.0, 6.0, 10.0],
        }
    )

    features = track_features(detections, fps=2.0)

    assert features.loc[0, "track_id"] == 7
    assert np.isclose(features.loc[0, "total_distance"], 10.0)
    assert np.isclose(features.loc[0, "mean_speed"], 10.0)
