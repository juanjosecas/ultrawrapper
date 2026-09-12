"""vision.yolo – modular YOLO wrapper for scientific pipelines."""

from vision.yolo.devices import detect_device, get_device_info
from vision.yolo.explain import detection_density_map, occlusion_sensitivity, plot_density_map
from vision.yolo.export import export_model
from vision.yolo.infer import predict_directory, predict_image, predict_images, predict_video
from vision.yolo.ml import frame_features, merge_labels, prepare_xy, shap_values, track_features
from vision.yolo.track import make_tracker_config, track_detections_dataframe, track_video
from vision.yolo.train import resume_training, train_model, validate_model
from vision.yolo.video import (
    draw_predictions_on_frame,
    write_annotated_video,
    write_annotated_video_from_dataframe,
)
from vision.yolo.video_io import (
    extract_frames,
    images_to_gif,
    images_to_video,
    read_frames,
    video_info,
    write_frames,
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
    "read_frames",
    "write_frames",
    "extract_frames",
    "images_to_video",
    "images_to_gif",
    "video_info",
    "detection_density_map",
    "plot_density_map",
    "occlusion_sensitivity",
    "frame_features",
    "track_features",
    "merge_labels",
    "prepare_xy",
    "shap_values",
    "train_model",
    "resume_training",
    "validate_model",
    "export_model",
]
