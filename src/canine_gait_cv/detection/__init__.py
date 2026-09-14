from canine_gait_cv.detection.base import Detector
from canine_gait_cv.detection.dummy import FixedBoxDetector
from canine_gait_cv.detection.types import (
    Detection,
    filter_detections_by_class,
    filter_detections_by_confidence,
    select_largest_detection,
)
from canine_gait_cv.detection.yolo import YOLODetector

__all__ = [
    "Detection",
    "Detector",
    "FixedBoxDetector",
    "YOLODetector",
    "filter_detections_by_class",
    "filter_detections_by_confidence",
    "select_largest_detection",
]
