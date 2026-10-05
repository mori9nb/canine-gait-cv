import pytest

from canine_gait_cv.gait import joint_angle_degrees
from canine_gait_cv.smoothing import SmoothedKeypoint


def _point(x: float, y: float) -> SmoothedKeypoint:
    return SmoothedKeypoint(
        x=x,
        y=y,
        velocity_x=0.0,
        velocity_y=0.0,
        source_confidence=1.0,
        measurement_used=True,
    )


def test_straight_joint_angle_is_180_degrees() -> None:
    angle = joint_angle_degrees(
        _point(0.0, 0.0),
        _point(1.0, 0.0),
        _point(2.0, 0.0),
    )

    assert angle == pytest.approx(180.0)


def test_right_joint_angle_is_90_degrees() -> None:
    angle = joint_angle_degrees(
        _point(1.0, 0.0),
        _point(0.0, 0.0),
        _point(0.0, 1.0),
    )

    assert angle == pytest.approx(90.0)


def test_zero_length_segment_has_no_angle() -> None:
    angle = joint_angle_degrees(
        _point(0.0, 0.0),
        _point(0.0, 0.0),
        _point(1.0, 0.0),
    )

    assert angle is None