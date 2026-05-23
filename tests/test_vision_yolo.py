"""Tests for vision.yolo – unit tests that do NOT require GPU or model weights."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# utils.py
# ---------------------------------------------------------------------------

class TestTensorConversions:
    def test_tensor_to_numpy_numpy(self):
        from vision.yolo.utils import tensor_to_numpy

        arr = np.array([1.0, 2.0, 3.0])
        result = tensor_to_numpy(arr)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, arr)

    def test_tensor_to_numpy_list(self):
        from vision.yolo.utils import tensor_to_numpy

        result = tensor_to_numpy([1, 2, 3])
        assert isinstance(result, np.ndarray)

    def test_to_serializable_dict(self):
        from vision.yolo.utils import to_serializable

        obj = {"a": np.array([1, 2, 3]), "b": 3.14}
        result = to_serializable(obj)
        assert isinstance(result["a"], list)
        assert result["b"] == pytest.approx(3.14)

    def test_to_serializable_nested(self):
        from vision.yolo.utils import to_serializable

        obj = [np.int64(1), np.float32(2.0)]
        result = to_serializable(obj)
        assert result == [1, 2.0]

    def test_xywh_xyxy_roundtrip(self):
        from vision.yolo.utils import xywh_to_xyxy, xyxy_to_xywh

        original = [100.0, 200.0, 50.0, 80.0]
        xyxy = xywh_to_xyxy(original)
        back = xyxy_to_xywh(xyxy)
        assert back == pytest.approx(original)


# ---------------------------------------------------------------------------
# devices.py
# ---------------------------------------------------------------------------

class TestDevices:
    def test_detect_device_returns_string(self):
        from vision.yolo.devices import detect_device

        device = detect_device()
        assert isinstance(device, str)
        assert device in ("cpu", "cuda:0")

    def test_get_device_info_fields(self):
        from vision.yolo.devices import get_device_info

        info = get_device_info()
        assert hasattr(info, "device")
        assert hasattr(info, "ram_total_mb")
        assert info.ram_total_mb > 0

    def test_get_gpu_memory_returns_dict(self):
        from vision.yolo.devices import get_gpu_memory

        mem = get_gpu_memory()
        assert isinstance(mem, dict)

    def test_clear_gpu_memory_runs(self):
        from vision.yolo.devices import clear_gpu_memory

        clear_gpu_memory()  # should not raise

    def test_estimate_batch_size_positive(self):
        from vision.yolo.devices import estimate_optimal_batch_size

        bs = estimate_optimal_batch_size()
        assert bs >= 1


# ---------------------------------------------------------------------------
# memory.py
# ---------------------------------------------------------------------------

class TestMemory:
    def test_snapshot_fields(self):
        from vision.yolo.memory import get_memory_snapshot

        snap = get_memory_snapshot()
        assert snap.ram_total_mb > 0
        assert snap.ram_used_mb >= 0

    def test_tracker_to_dataframe(self):
        from vision.yolo.memory import MemoryTracker

        tracker = MemoryTracker()
        tracker.record("test")
        df = tracker.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert "ram_used_mb" in df.columns


# ---------------------------------------------------------------------------
# serialization.py
# ---------------------------------------------------------------------------

class TestSerialization:
    def test_save_load_parquet(self):
        from vision.yolo.serialization import save_dataframe, load_dataframe

        df = pd.DataFrame({"a": [1, 2], "b": [3.0, 4.0]})
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.parquet"
            save_dataframe(df, path, fmt="parquet")
            loaded = load_dataframe(path)
            pd.testing.assert_frame_equal(df, loaded)

    def test_save_load_csv(self):
        from vision.yolo.serialization import save_dataframe, load_dataframe

        df = pd.DataFrame({"x": [10, 20]})
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.csv"
            save_dataframe(df, path, fmt="csv")
            loaded = load_dataframe(path)
            pd.testing.assert_frame_equal(df, loaded)

    def test_append_to_parquet(self):
        from vision.yolo.serialization import append_to_parquet

        df1 = pd.DataFrame({"v": [1, 2]})
        df2 = pd.DataFrame({"v": [3, 4]})
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "out.parquet"
            append_to_parquet(df1, path)
            append_to_parquet(df2, path)
            merged = pd.read_parquet(path)
            assert list(merged["v"]) == [1, 2, 3, 4]

    def test_save_load_json(self):
        from vision.yolo.serialization import save_json, load_json

        obj = {"key": [1, 2, 3], "nested": {"x": np.array([1.0])}}
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "data.json"
            save_json(obj, path)
            loaded = load_json(path)
            assert loaded["key"] == [1, 2, 3]
            assert loaded["nested"]["x"] == [1.0]


# ---------------------------------------------------------------------------
# dataframe.py
# ---------------------------------------------------------------------------

class TestDataframe:
    def test_empty_dataframe_columns(self):
        from vision.yolo.dataframe import _empty_dataframe

        df = _empty_dataframe()
        expected_cols = [
            "frame", "track_id", "class_id", "class_name", "confidence",
            "xmin", "ymin", "xmax", "ymax", "polygon", "mask", "keypoints",
            "timestamp",
        ]
        assert list(df.columns) == expected_cols

    def test_concat_results_empty(self):
        from vision.yolo.dataframe import concat_results, _empty_dataframe

        result = concat_results([])
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_concat_results_merges(self):
        from vision.yolo.dataframe import concat_results, _empty_dataframe

        df1 = pd.DataFrame({"frame": [0], "confidence": [0.9]})
        df2 = pd.DataFrame({"frame": [1], "confidence": [0.8]})
        merged = concat_results([df1, df2])
        assert len(merged) == 2


# ---------------------------------------------------------------------------
# metrics.py
# ---------------------------------------------------------------------------

class TestMetrics:
    def test_compute_iou_full_overlap(self):
        from vision.yolo.metrics import compute_iou

        box = [0, 0, 10, 10]
        assert compute_iou(box, box) == pytest.approx(1.0)

    def test_compute_iou_no_overlap(self):
        from vision.yolo.metrics import compute_iou

        a = [0, 0, 5, 5]
        b = [10, 10, 20, 20]
        assert compute_iou(a, b) == pytest.approx(0.0)

    def test_compute_iou_partial(self):
        from vision.yolo.metrics import compute_iou

        a = [0, 0, 10, 10]
        b = [5, 5, 15, 15]
        iou = compute_iou(a, b)
        assert 0 < iou < 1

    def test_detections_per_class(self):
        from vision.yolo.metrics import detections_per_class

        df = pd.DataFrame({"class_name": ["cat", "cat", "dog"]})
        result = detections_per_class(df)
        assert result[result["class_name"] == "cat"]["count"].values[0] == 2


# ---------------------------------------------------------------------------
# annotations/internal.py
# ---------------------------------------------------------------------------

class TestAnnotationInternal:
    def test_roundtrip(self):
        from vision.yolo.annotations.internal import Annotation, AnnotationSample

        sample = AnnotationSample(
            image_path="img.jpg",
            width=640,
            height=480,
            annotations=[
                Annotation(
                    task="detect",
                    class_id=0,
                    class_name="cat",
                    bbox=[10, 20, 100, 200],
                )
            ],
        )
        d = sample.to_dict()
        restored = AnnotationSample.from_dict(d)
        assert restored.image_path == "img.jpg"
        assert restored.annotations[0].class_name == "cat"


# ---------------------------------------------------------------------------
# annotations/yolo.py
# ---------------------------------------------------------------------------

class TestYOLOAnnotations:
    def test_write_read_roundtrip(self):
        from vision.yolo.annotations.internal import Annotation, AnnotationSample
        from vision.yolo.annotations.yolo import read, write

        sample = AnnotationSample(
            image_path="frame_000.jpg",
            width=640,
            height=480,
            annotations=[
                Annotation(
                    task="detect",
                    class_id=1,
                    class_name="dog",
                    bbox=[100, 100, 200, 200],
                )
            ],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            labels_dir = Path(tmpdir) / "labels"
            write([sample], labels_dir)
            loaded = read(labels_dir, class_names=["cat", "dog"])
            assert len(loaded) == 1
            ann = loaded[0].annotations[0]
            assert ann.class_id == 1


# ---------------------------------------------------------------------------
# annotations/coco.py
# ---------------------------------------------------------------------------

class TestCOCOAnnotations:
    def test_write_read_roundtrip(self):
        from vision.yolo.annotations.internal import Annotation, AnnotationSample
        from vision.yolo.annotations.coco import read, write

        sample = AnnotationSample(
            image_path="cat.jpg",
            width=320,
            height=240,
            annotations=[
                Annotation(
                    task="detect",
                    class_id=0,
                    class_name="cat",
                    bbox=[10.0, 20.0, 100.0, 200.0],
                )
            ],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "coco_out"
            write([sample], out_dir, class_names=["cat"])
            loaded = read(out_dir)
            assert len(loaded) == 1
            assert loaded[0].annotations[0].class_name == "cat"


# ---------------------------------------------------------------------------
# annotations/voc.py
# ---------------------------------------------------------------------------

class TestVOCAnnotations:
    def test_write_read_roundtrip(self):
        from vision.yolo.annotations.internal import Annotation, AnnotationSample
        from vision.yolo.annotations.voc import read, write

        sample = AnnotationSample(
            image_path="img.jpg",
            width=800,
            height=600,
            annotations=[
                Annotation(
                    task="detect",
                    class_id=0,
                    class_name="person",
                    bbox=[50.0, 60.0, 300.0, 400.0],
                )
            ],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "voc"
            write([sample], out)
            loaded = read(out)
            assert len(loaded) == 1
            assert loaded[0].annotations[0].class_name == "person"


# ---------------------------------------------------------------------------
# annotations/labelme.py
# ---------------------------------------------------------------------------

class TestLabelMeAnnotations:
    def test_write_read_roundtrip(self):
        from vision.yolo.annotations.internal import Annotation, AnnotationSample
        from vision.yolo.annotations.labelme import read, write

        sample = AnnotationSample(
            image_path="scene.jpg",
            width=512,
            height=512,
            annotations=[
                Annotation(
                    task="segment",
                    class_id=0,
                    class_name="cell",
                    bbox=[10.0, 10.0, 100.0, 100.0],
                    polygon=[[10.0, 10.0], [100.0, 10.0], [100.0, 100.0], [10.0, 100.0]],
                )
            ],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "labelme"
            write([sample], out)
            loaded = read(out)
            assert len(loaded) == 1
            assert loaded[0].annotations[0].class_name == "cell"
            assert len(loaded[0].annotations[0].polygon) == 4


# ---------------------------------------------------------------------------
# track.py (pure pandas logic – no model needed)
# ---------------------------------------------------------------------------

class TestTrackingDataframe:
    def _make_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "frame": [0, 1, 2, 0, 1],
                "track_id": [1, 1, 1, 2, 2],
                "class_name": ["cat"] * 5,
                "confidence": [0.9, 0.85, 0.88, 0.7, 0.75],
                "xmin": [10.0, 11.0, 12.0, 50.0, 51.0],
                "ymin": [10.0, 11.0, 12.0, 50.0, 51.0],
                "xmax": [30.0, 31.0, 32.0, 70.0, 71.0],
                "ymax": [30.0, 31.0, 32.0, 70.0, 71.0],
                "timestamp": [0.0, 0.033, 0.066, 0.0, 0.033],
            }
        )

    def test_build_tracks_dataframe(self):
        from vision.yolo.track import build_tracks_dataframe

        df = self._make_df()
        result = build_tracks_dataframe(df)
        assert "x_center" in result.columns
        assert "velocity" in result.columns
        assert "trajectory_length" in result.columns

    def test_compute_track_statistics(self):
        from vision.yolo.track import compute_track_statistics

        df = self._make_df()
        stats = compute_track_statistics(df)
        assert "track_id" in stats.columns
        assert len(stats) == 2

    def test_filter_short_tracks(self):
        from vision.yolo.track import filter_short_tracks

        df = self._make_df()
        filtered = filter_short_tracks(df, min_frames=3)
        assert set(filtered["track_id"].unique()) == {1}

    def test_smooth_tracks(self):
        from vision.yolo.track import smooth_tracks, build_tracks_dataframe

        df = build_tracks_dataframe(self._make_df())
        smoothed = smooth_tracks(df, window=2)
        assert "x_center" in smoothed.columns


# ---------------------------------------------------------------------------
# plotting.py (smoke tests – just ensure no exceptions)
# ---------------------------------------------------------------------------

class TestPlotting:
    def test_plot_class_distribution(self):
        from vision.yolo.plotting import plot_class_distribution

        df = pd.DataFrame({"class_name": ["cat", "cat", "dog", "dog", "dog"]})
        fig = plot_class_distribution(df)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_tracking_trajectories(self):
        from vision.yolo.plotting import plot_tracking_trajectories

        df = pd.DataFrame(
            {
                "track_id": [1, 1, 1],
                "frame": [0, 1, 2],
                "xmin": [10.0, 20.0, 30.0],
                "xmax": [50.0, 60.0, 70.0],
                "ymin": [10.0, 20.0, 30.0],
                "ymax": [50.0, 60.0, 70.0],
            }
        )
        fig = plot_tracking_trajectories(df)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_video_statistics(self):
        from vision.yolo.plotting import plot_video_statistics

        df = pd.DataFrame(
            {
                "frame": [0, 0, 1, 1],
                "class_name": ["cat", "dog", "cat", "cat"],
                "confidence": [0.9, 0.8, 0.85, 0.75],
                "xmin": [10.0, 20.0, 15.0, 5.0],
                "ymin": [10.0, 20.0, 15.0, 5.0],
                "xmax": [50.0, 60.0, 55.0, 45.0],
                "ymax": [50.0, 60.0, 55.0, 45.0],
            }
        )
        fig = plot_video_statistics(df)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)


# ---------------------------------------------------------------------------
# augmentations.py (import only – albumentations required)
# ---------------------------------------------------------------------------

class TestAugmentations:
    def test_build_all_presets(self):
        pytest.importorskip("albumentations")
        from vision.yolo.augmentations import build_augmentation_pipeline

        for preset in ["light", "medium", "heavy", "microscopy", "tracking", "pose"]:
            pipeline = build_augmentation_pipeline(preset=preset, image_size=320)
            assert pipeline is not None

    def test_unknown_preset_raises(self):
        pytest.importorskip("albumentations")
        from vision.yolo.augmentations import build_augmentation_pipeline

        with pytest.raises(ValueError):
            build_augmentation_pipeline(preset="unknown")

    def test_apply_augmentation(self):
        pytest.importorskip("albumentations")
        from vision.yolo.augmentations import build_augmentation_pipeline, apply_augmentation

        pipeline = build_augmentation_pipeline("light", image_size=320)
        image = np.zeros((320, 320, 3), dtype=np.uint8)
        result = apply_augmentation(pipeline, image, bboxes=[[0.5, 0.5, 0.2, 0.2]], class_labels=[0])
        assert "image" in result
        assert result["image"].shape == (320, 320, 3)


# ---------------------------------------------------------------------------
# datasets.py
# ---------------------------------------------------------------------------

class TestDatasets:
    def test_list_dataset_images(self):
        from vision.yolo.datasets import list_dataset_images

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "a.jpg").touch()
            (root / "b.png").touch()
            (root / "c.txt").touch()
            imgs = list_dataset_images(root)
            assert len(imgs) == 2

    def test_write_data_yaml(self):
        from vision.yolo.datasets import write_data_yaml

        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_path = write_data_yaml(tmpdir, class_names=["cat", "dog"])
            assert yaml_path.exists()
            import yaml
            with open(yaml_path) as fh:
                cfg = yaml.safe_load(fh)
            assert cfg["nc"] == 2
            assert cfg["names"] == ["cat", "dog"]

    def test_dataset_summary(self):
        from vision.yolo.datasets import dataset_summary

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "images" / "train").mkdir(parents=True)
            (root / "images" / "train" / "a.jpg").touch()
            (root / "labels" / "train").mkdir(parents=True)
            (root / "labels" / "train" / "a.txt").touch()
            df = dataset_summary(root)
            train_row = df[df["split"] == "train"]
            assert train_row["images"].values[0] == 1
            assert train_row["labels"].values[0] == 1


# ---------------------------------------------------------------------------
# video.py (unit – no real video file needed)
# ---------------------------------------------------------------------------

class TestVideoInfo:
    def test_video_frame_generator_no_file(self):
        from vision.yolo.video import video_frame_generator

        gen = video_frame_generator("/nonexistent/video.mp4", batch_size=4)
        batches = list(gen)
        assert batches == []
