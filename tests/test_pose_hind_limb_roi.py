import pytest

from canine_gait_cv.pose import BodyOrientation, DogKeypoint, DogPose, KeypointName
from canine_gait_cv.preprocessing import BoundingBox
from canine_gait_cv.roi import build_pose_hind_limb_roi


DOG_BOX = BoundingBox(100, 50, 500, 350)


def _point(x: float, y: float, confidence: float = 0.9) -> DogKeypoint:
    return DogKeypoint(x=x, y=y, confidence=confidence)


def test_pose_roi_uses_confident_bilateral_hind_limb_points() -> None:
    pose = DogPose({
        KeypointName.LEFT_HIP: _point(180, 150),
        KeypointName.LEFT_KNEE: _point(200, 230),
        KeypointName.LEFT_BACK_PAW: _point(160, 310),
        KeypointName.RIGHT_HIP: _point(230, 155),
        KeypointName.RIGHT_KNEE: _point(250, 235),
        KeypointName.RIGHT_BACK_PAW: _point(280, 315),
    })

    roi = build_pose_hind_limb_roi(
        pose, DOG_BOX,
        orientation=BodyOrientation.RIGHT,
        image_width=640,
        image_height=480,
        padding_x_fraction=0.05,
        padding_y_fraction=0.05,
    )

    assert roi is not None
    assert roi.box == BoundingBox(140, 135, 301, 331)
    assert roi.keypoint_count == 6
    assert roi.used_fallback is False
    assert roi.confidence == pytest.approx(0.9)


def test_low_confidence_points_are_excluded() -> None:
    pose = DogPose({
        KeypointName.LEFT_HIP: _point(180, 150),
        KeypointName.LEFT_KNEE: _point(200, 230, 0.1),
        KeypointName.LEFT_BACK_PAW: _point(160, 310),
        KeypointName.RIGHT_HIP: _point(230, 155),
    })

    roi = build_pose_hind_limb_roi(
        pose, DOG_BOX,
        orientation=BodyOrientation.RIGHT,
        image_width=640,
        image_height=480,
        padding_x_fraction=0.0,
        padding_y_fraction=0.0,
    )

    assert roi is not None
    assert roi.keypoint_count == 3
    assert roi.box == BoundingBox(160, 150, 231, 311)


def test_fallback_uses_anatomical_orientation_not_box_motion() -> None:
    roi = build_pose_hind_limb_roi(
        DogPose({}), DOG_BOX,
        orientation=BodyOrientation.RIGHT,
        image_width=640,
        image_height=480,
        padding_x_fraction=0.0,
        padding_y_fraction=0.0,
    )

    assert roi is not None
    assert roi.box == BoundingBox(100, 125, 320, 350)
    assert roi.used_fallback is True
    assert roi.keypoint_count == 0


def test_unknown_orientation_and_insufficient_pose_returns_none() -> None:
    roi = build_pose_hind_limb_roi(
        DogPose({}), DOG_BOX,
        orientation=BodyOrientation.UNKNOWN,
        image_width=640,
        image_height=480,
    )

    assert roi is None


def test_pose_roi_is_clipped_to_image() -> None:
    pose = DogPose({
        KeypointName.LEFT_HIP: _point(5, 5),
        KeypointName.LEFT_KNEE: _point(10, 50),
        KeypointName.LEFT_BACK_PAW: _point(20, 95),
    })

    roi = build_pose_hind_limb_roi(
        pose, BoundingBox(0, 0, 100, 100),
        orientation=BodyOrientation.RIGHT,
        image_width=100,
        image_height=100,
    )

    assert roi is not None
    assert roi.box.x_min == 0
    assert roi.box.y_min == 0
    assert roi.box.y_max == 100
