from canine_gait_cv.pose import (
    DeepLabCutIndividual,
    DeepLabCutMultiPoseFrame,
    DogKeypoint,
    DogPose,
    KeypointName,
)
from canine_gait_cv.preprocessing import BoundingBox
from canine_gait_cv.quality import (
    DetectionQualityFlag,
    assess_detection_quality,
)


def _dog(
    detection_index: int,
    box: BoundingBox,
    *,
    bbox_confidence: float = 0.95,
    keypoint_confidence: float = 0.90,
) -> DeepLabCutIndividual:
    pose = DogPose(
        {
            name: DogKeypoint(
                x=float(box.x_min + 10),
                y=float(box.y_min + 10),
                confidence=keypoint_confidence,
            )
            for name in KeypointName
        }
    )

    return DeepLabCutIndividual(
        detection_index=detection_index,
        dog_box=box,
        bbox_confidence=bbox_confidence,
        pose=pose,
    )


def test_valid_detection_is_accepted() -> None:
    frame = DeepLabCutMultiPoseFrame(
        frame_index=0,
        individuals=(
            _dog(0, BoundingBox(10, 20, 110, 120)),
        ),
    )

    results = assess_detection_quality(frame)

    assert results[0].accepted_for_training is True
    assert results[0].flags == (DetectionQualityFlag.VALID,)


def test_low_bbox_confidence_is_rejected() -> None:
    frame = DeepLabCutMultiPoseFrame(
        frame_index=0,
        individuals=(
            _dog(
                0,
                BoundingBox(10, 20, 110, 120),
                bbox_confidence=0.20,
            ),
        ),
    )

    results = assess_detection_quality(frame)

    assert results[0].accepted_for_training is False
    assert DetectionQualityFlag.LOW_BBOX_CONFIDENCE in results[0].flags


def test_low_pose_coverage_is_rejected() -> None:
    frame = DeepLabCutMultiPoseFrame(
        frame_index=0,
        individuals=(
            _dog(
                0,
                BoundingBox(10, 20, 110, 120),
                keypoint_confidence=0.10,
            ),
        ),
    )

    results = assess_detection_quality(frame)

    assert results[0].accepted_for_training is False
    assert DetectionQualityFlag.LOW_POSE_COVERAGE in results[0].flags


def test_overlapping_detections_are_marked_uncertain() -> None:
    frame = DeepLabCutMultiPoseFrame(
        frame_index=0,
        individuals=(
            _dog(0, BoundingBox(10, 20, 110, 120)),
            _dog(1, BoundingBox(50, 20, 150, 120)),
        ),
    )

    results = assess_detection_quality(
        frame,
        overlap_iou_threshold=0.30,
    )

    assert results[0].accepted_for_training is False
    assert results[1].accepted_for_training is False
    assert DetectionQualityFlag.OVERLAPPING_DETECTION in results[0].flags
    assert DetectionQualityFlag.OVERLAPPING_DETECTION in results[1].flags