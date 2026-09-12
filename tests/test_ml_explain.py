import numpy as np
import pandas as pd


def sample_predictions():
    return pd.DataFrame(
        {
            "frame": [0, 0, 1, 1],
            "track_id": [1, 2, 1, 2],
            "class_name": ["mouse", "mouse", "mouse", "mouse"],
            "confidence": [0.9, 0.8, 0.85, 0.75],
            "xmin": [10, 100, 20, 90],
            "ymin": [10, 50, 15, 55],
            "xmax": [30, 130, 40, 120],
            "ymax": [40, 90, 45, 95],
            "timestamp": [0.0, 0.0, 0.1, 0.1],
        }
    )


def test_detection_density_map_shape_and_weight():
    from vision.yolo.explain import detection_density_map

    df = sample_predictions()
    density = detection_density_map(df, image_width=200, image_height=100, bins=10)

    assert density.shape == (10, 10)
    assert density.sum() == np.testing.assert_approx_equal(density.sum(), df["confidence"].sum())


def test_frame_features_one_row_per_frame():
    from vision.yolo.ml import frame_features

    features = frame_features(sample_predictions())

    assert len(features) == 2
    assert "detection_count" in features.columns
    assert "count_mouse" in features.columns
    assert features.loc[0, "detection_count"] == 2


def test_track_features_one_row_per_track():
    from vision.yolo.ml import track_features

    features = track_features(sample_predictions())

    assert len(features) == 2
    assert "distance_total" in features.columns
    assert "speed_mean" in features.columns
    assert set(features["track_id"]) == {1, 2}


def test_merge_labels_and_prepare_xy():
    from vision.yolo.ml import frame_features, merge_labels, prepare_xy

    features = frame_features(sample_predictions())
    labels = pd.DataFrame({"frame": [0, 1], "condition": [0, 1]})
    dataset = merge_labels(features, labels, on="frame")
    X, y = prepare_xy(dataset, target="condition", drop_columns=["frame"])

    assert "condition" not in X.columns
    assert "frame" not in X.columns
    assert list(y) == [0, 1]
