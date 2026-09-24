from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from canine_gait_cv.pose import DeepLabCutMultiPoseFrame
from canine_gait_cv.preprocessing import BoundingBox


class DetectionQualityFlag(str, Enum):
    VALID = "valid"
    LOW_BBOX_CONFIDENCE = "low_bbox_confidence"
    LOW_POSE_COVERAGE = "low_pose_coverage"
    OVERLAPPING_DETECTION = "overlapping_detection"


@dataclass(frozen=True)
class DetectionQualityAssessment:
    detection_index: int
    accepted_for_training: bool
    flags: tuple[DetectionQualityFlag, ...]
    bbox_confidence: float
    reliable_keypoint_fraction: float
    quality_score: float


def assess_detection_quality(
    frame: DeepLabCutMultiPoseFrame,
    *,
    min_bbox_confidence: float = 0.50,
    min_keypoint_confidence: float = 0.30,
    min_reliable_keypoint_fraction: float = 0.50,
    overlap_iou_threshold: float = 0.30,
) -> tuple[DetectionQualityAssessment, ...]:
    """Assess detections without deleting the original raw data."""

    _validate_threshold(
        "min_bbox_confidence",
        min_bbox_confidence,
    )
    _validate_threshold(
        "min_keypoint_confidence",
        min_keypoint_confidence,
    )
    _validate_threshold(
        "min_reliable_keypoint_fraction",
        min_reliable_keypoint_fraction,
    )
    _validate_threshold(
        "overlap_iou_threshold",
        overlap_iou_threshold,
    )

    overlapping_positions: set[int] = set()

    for first_position in range(len(frame.individuals)):
        for second_position in range(
            first_position + 1,
            len(frame.individuals),
        ):
            first_box = frame.individuals[
                first_position
            ].dog_box
            second_box = frame.individuals[
                second_position
            ].dog_box

            if _box_iou(first_box, second_box) >= overlap_iou_threshold:
                overlapping_positions.add(first_position)
                overlapping_positions.add(second_position)

    assessments: list[DetectionQualityAssessment] = []

    for position, individual in enumerate(frame.individuals):
        points = tuple(individual.pose.keypoints.values())

        if points:
            reliable_count = sum(
                point.confidence >= min_keypoint_confidence
                for point in points
            )
            reliable_fraction = reliable_count / len(points)
        else:
            reliable_fraction = 0.0

        flags: list[DetectionQualityFlag] = []

        if individual.bbox_confidence < min_bbox_confidence:
            flags.append(
                DetectionQualityFlag.LOW_BBOX_CONFIDENCE
            )

        if reliable_fraction < min_reliable_keypoint_fraction:
            flags.append(
                DetectionQualityFlag.LOW_POSE_COVERAGE
            )

        if position in overlapping_positions:
            flags.append(
                DetectionQualityFlag.OVERLAPPING_DETECTION
            )

        accepted = not flags

        if accepted:
            flags.append(DetectionQualityFlag.VALID)

        quality_score = (
            individual.bbox_confidence
            * reliable_fraction
        )

        assessments.append(
            DetectionQualityAssessment(
                detection_index=individual.detection_index,
                accepted_for_training=accepted,
                flags=tuple(flags),
                bbox_confidence=individual.bbox_confidence,
                reliable_keypoint_fraction=reliable_fraction,
                quality_score=quality_score,
            )
        )

    return tuple(assessments)


def _box_iou(
    first: BoundingBox,
    second: BoundingBox,
) -> float:
    x_min = max(first.x_min, second.x_min)
    y_min = max(first.y_min, second.y_min)
    x_max = min(first.x_max, second.x_max)
    y_max = min(first.y_max, second.y_max)

    intersection = (
        max(0, x_max - x_min)
        * max(0, y_max - y_min)
    )
    union = first.area + second.area - intersection

    return 0.0 if union == 0 else intersection / union


def _validate_threshold(name: str, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(
            f"{name} must be between 0 and 1."
        )