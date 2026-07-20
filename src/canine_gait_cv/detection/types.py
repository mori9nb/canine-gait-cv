from __future__ import annotations

from dataclasses import dataclass

from canine_gait_cv.preprocessing import BoundingBox


@dataclass(frozen=True)
class Detection:
    """Single object detection result."""

    class_name: str
    confidence: float
    box: BoundingBox

    def __post_init__(self) -> None:
        if not self.class_name:
            raise ValueError("class_name cannot be empty.")

        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1.")

        if self.box.area == 0:
            raise ValueError("detection box area must be greater than zero.")


def filter_detections_by_class(
    detections: list[Detection],
    class_name: str,
) -> list[Detection]:
    """Return only detections that match a given class name."""

    return [
        detection
        for detection in detections
        if detection.class_name == class_name
    ]


def filter_detections_by_confidence(
    detections: list[Detection],
    min_confidence: float,
) -> list[Detection]:
    """Return detections with confidence greater than or equal to threshold."""

    if not 0.0 <= min_confidence <= 1.0:
        raise ValueError("min_confidence must be between 0 and 1.")

    return [
        detection
        for detection in detections
        if detection.confidence >= min_confidence
    ]


def select_largest_detection(
    detections: list[Detection],
) -> Detection | None:
    """Select the detection with the largest bounding box area."""

    if not detections:
        return None

    return max(detections, key=lambda detection: detection.box.area)