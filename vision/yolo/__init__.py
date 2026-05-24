"""vision.yolo – modular YOLO wrapper for scientific pipelines."""

from vision.yolo.devices import detect_device, get_device_info
from vision.yolo.export import export_model
from vision.yolo.infer import predict_directory, predict_image, predict_images, predict_video
from vision.yolo.track import make_tracker_config, track_detections_dataframe, track_video
from vision.yolo.train import resume_training, train_model, validate_model
from vision.yolo.video import (
    draw_predictions_on_frame,
    write_annotated_video,
    write_annotated_video_from_dataframe,
)

__all__ = [
    "detect_device",
    "get_device_info",
    "predict_image",
    "predict_images",
    "predict_video",
    "predict_directory",
    "track_video",
    "make_tracker_config",
    "track_detections_dataframe",
    "draw_predictions_on_frame",
    "write_annotated_video",
    "write_annotated_video_from_dataframe",
    "train_model",
    "resume_training",
    "validate_model",
    "export_model",
]
