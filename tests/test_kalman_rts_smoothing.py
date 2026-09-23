import numpy as np
import pytest

from canine_gait_cv.pose import DogKeypoint
from canine_gait_cv.smoothing import smooth_keypoint_trajectory


def _point(x: float, y: float = 20.0, confidence: float = 0.9) -> DogKeypoint:
    return DogKeypoint(x=x, y=y, confidence=confidence)


def test_empty_trajectory_returns_empty_result() -> None:
    assert smooth_keypoint_trajectory([]) == []


def test_frames_before_first_valid_measurement_remain_missing() -> None:
    result = smooth_keypoint_trajectory([None, _point(10.0), _point(12.0)])

    assert result[0] is None
    assert result[1] is not None


def test_missing_measurement_is_filled_by_motion_model() -> None:
    result = smooth_keypoint_trajectory(
        [_point(0.0), _point(10.0), None, _point(30.0), _point(40.0)],
        process_variance=0.1,
        measurement_variance=1.0,
    )

    assert result[2] is not None
    assert result[2].measurement_used is False
    assert result[2].x == pytest.approx(20.0, abs=3.0)


def test_low_confidence_measurement_is_not_used() -> None:
    result = smooth_keypoint_trajectory(
        [_point(0.0), _point(10.0), _point(100.0, confidence=0.05), _point(30.0)],
        min_confidence=0.2,
    )

    assert result[2].measurement_used is False
    assert result[2].source_confidence == pytest.approx(0.05)


def test_large_high_confidence_outlier_is_rejected() -> None:
    result = smooth_keypoint_trajectory(
        [_point(0.0), _point(10.0), _point(500.0), _point(30.0), _point(40.0)],
        process_variance=0.1,
        measurement_variance=1.0,
    )

    assert result[2].measurement_used is False
    assert result[2].x < 100.0


def test_rts_smoothing_reduces_error_on_noisy_linear_motion() -> None:
    true_x = np.arange(20, dtype=float) * 5.0
    noise = np.array(
        [0, 2, -2, 3, -3, 2, -1, 3, -2, 1, -1, 2, -3, 3, -2, 2, -1, 1, -2, 0],
        dtype=float,
    )
    points = [_point(float(x + n)) for x, n in zip(true_x, noise)]

    result = smooth_keypoint_trajectory(
        points,
        process_variance=0.05,
        measurement_variance=9.0,
    )
    smoothed_x = np.array([point.x for point in result if point is not None])

    raw_rmse = float(np.sqrt(np.mean(noise**2)))
    smoothed_rmse = float(np.sqrt(np.mean((smoothed_x - true_x) ** 2)))
    assert smoothed_rmse < raw_rmse


def test_invalid_parameters_are_rejected() -> None:
    with pytest.raises(ValueError, match="dt"):
        smooth_keypoint_trajectory([_point(0.0)], dt=0.0)
