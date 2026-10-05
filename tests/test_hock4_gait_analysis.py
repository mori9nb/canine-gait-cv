import pytest

from canine_gait_cv.pose import DogKeypoint
from scripts.export_hock4_gait_analysis import (
    allowed_estimated_frames,
    joint_angle_degrees,
)


def _point(x: float, y: float) -> DogKeypoint:
    return DogKeypoint(
        x=x,
        y=y,
        confidence=1.0,
    )


def test_joint_angle_calculates_right_angle() -> None:
    angle = joint_angle_degrees(
        _point(0.0, 1.0),
        _point(0.0, 0.0),
        _point(1.0, 0.0),
    )

    assert angle == pytest.approx(90.0)


def test_joint_angle_returns_none_for_zero_length_segment() -> None:
    angle = joint_angle_degrees(
        _point(0.0, 0.0),
        _point(0.0, 0.0),
        _point(1.0, 0.0),
    )

    assert angle is None


def test_short_internal_gap_can_be_estimated() -> None:
    observed = [True, True, False, False, True, True]

    allowed = allowed_estimated_frames(
        observed,
        max_gap=2,
    )

    assert allowed == [True, True, True, True, True, True]


def test_long_internal_gap_is_not_estimated() -> None:
    observed = [True, False, False, False, True]

    allowed = allowed_estimated_frames(
        observed,
        max_gap=2,
    )

    assert allowed == observed


def test_edge_gaps_are_not_estimated() -> None:
    observed = [False, True, True, False]

    allowed = allowed_estimated_frames(
        observed,
        max_gap=3,
    )

    assert allowed == observed