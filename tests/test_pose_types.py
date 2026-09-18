import pytest

from canine_gait_cv.pose import (
    BodyOrientation,
    DogKeypoint,
    DogPose,
    KeypointName,
    estimate_body_orientation,
)
from canine_gait_cv.preprocessing import BoundingBox


DOG_BOX = BoundingBox(x_min=100, y_min=50, x_max=500, y_max=350)


def _point(x: float, confidence: float = 0.9) -> DogKeypoint:
    return DogKeypoint(x=x, y=100.0, confidence=confidence)


def test_orientation_is_right_when_nose_is_right_of_tail_root() -> None:
    pose = DogPose(
        {
            KeypointName.NOSE: _point(450),
            KeypointName.TAIL_ROOT: _point(150),
        }
    )

    estimate = estimate_body_orientation(pose, DOG_BOX)

    assert estimate.orientation is BodyOrientation.RIGHT
    assert estimate.confidence == pytest.approx(0.9)
    assert estimate.normalized_separation == pytest.approx(0.75)
    assert estimate.front_source == "nose"
    assert estimate.rear_source == "tail_root"


def test_orientation_is_left_when_nose_is_left_of_tail_root() -> None:
    pose = DogPose(
        {
            KeypointName.NOSE: _point(130, 0.8),
            KeypointName.TAIL_ROOT: _point(430, 0.7),
        }
    )

    estimate = estimate_body_orientation(pose, DOG_BOX)

    assert estimate.orientation is BodyOrientation.LEFT
    assert estimate.confidence == pytest.approx(0.7)


def test_orientation_uses_eyes_and_hips_as_fallbacks() -> None:
    pose = DogPose(
        {
            KeypointName.LEFT_EYE: _point(440, 0.8),
            KeypointName.RIGHT_EYE: _point(460, 0.7),
            KeypointName.LEFT_HIP: _point(180, 0.85),
            KeypointName.RIGHT_HIP: _point(200, 0.75),
        }
    )

    estimate = estimate_body_orientation(pose, DOG_BOX)

    assert estimate.orientation is BodyOrientation.RIGHT
    assert estimate.front_source == "eyes"
    assert estimate.rear_source == "hips"
    assert estimate.confidence == pytest.approx(0.7)


def test_orientation_is_unknown_when_required_anatomy_is_missing() -> None:
    pose = DogPose({KeypointName.NOSE: _point(450)})

    estimate = estimate_body_orientation(pose, DOG_BOX)

    assert estimate.orientation is BodyOrientation.UNKNOWN
    assert estimate.confidence == 0.0
    assert estimate.front_source == "nose"
    assert estimate.rear_source is None


def test_orientation_is_unknown_for_low_confidence_keypoints() -> None:
    pose = DogPose(
        {
            KeypointName.NOSE: _point(450, 0.2),
            KeypointName.TAIL_ROOT: _point(150, 0.9),
        }
    )

    estimate = estimate_body_orientation(pose, DOG_BOX)

    assert estimate.orientation is BodyOrientation.UNKNOWN


def test_orientation_is_unknown_when_anchors_are_too_close() -> None:
    pose = DogPose(
        {
            KeypointName.NOSE: _point(310),
            KeypointName.TAIL_ROOT: _point(290),
        }
    )

    estimate = estimate_body_orientation(pose, DOG_BOX)

    assert estimate.orientation is BodyOrientation.UNKNOWN
    assert estimate.normalized_separation == pytest.approx(0.05)


def test_orientation_rule_is_scale_independent() -> None:
    small_pose = DogPose(
        {
            KeypointName.NOSE: _point(90),
            KeypointName.TAIL_ROOT: _point(30),
        }
    )
    large_pose = DogPose(
        {
            KeypointName.NOSE: _point(900),
            KeypointName.TAIL_ROOT: _point(300),
        }
    )

    small = estimate_body_orientation(
        small_pose,
        BoundingBox(x_min=10, y_min=10, x_max=110, y_max=90),
    )
    large = estimate_body_orientation(
        large_pose,
        BoundingBox(x_min=100, y_min=100, x_max=1100, y_max=900),
    )

    assert small.orientation is BodyOrientation.RIGHT
    assert large.orientation is BodyOrientation.RIGHT
    assert small.normalized_separation == pytest.approx(0.6)
    assert large.normalized_separation == pytest.approx(0.6)


@pytest.mark.parametrize("confidence", [-0.1, 1.1])
def test_keypoint_rejects_invalid_confidence(confidence: float) -> None:
    with pytest.raises(ValueError, match="confidence"):
        DogKeypoint(x=10.0, y=20.0, confidence=confidence)
