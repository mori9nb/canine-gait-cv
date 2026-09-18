import pytest

from canine_gait_cv.preprocessing import BoundingBox
from canine_gait_cv.roi import (
    HindLimbROI,
    MovementDirection,
    build_hind_limb_roi,
    estimate_movement_direction,
)


def test_moving_right_uses_lower_left_of_dog_box() -> None:
    roi = build_hind_limb_roi(
        BoundingBox(x_min=100, y_min=50, x_max=300, y_max=250),
        movement_direction=MovementDirection.RIGHT,
        detection_confidence=0.9,
        image_width=640,
        image_height=480,
        padding_fraction=0.0,
    )

    assert roi == HindLimbROI(
        box=BoundingBox(x_min=100, y_min=110, x_max=210, y_max=250),
        movement_direction=MovementDirection.RIGHT,
        confidence=0.9,
    )


def test_moving_left_uses_lower_right_of_dog_box() -> None:
    roi = build_hind_limb_roi(
        BoundingBox(x_min=100, y_min=50, x_max=300, y_max=250),
        movement_direction=MovementDirection.LEFT,
        detection_confidence=0.8,
        image_width=640,
        image_height=480,
        padding_fraction=0.0,
    )

    assert roi.box == BoundingBox(x_min=190, y_min=110, x_max=300, y_max=250)


def test_roi_padding_is_clipped_to_image() -> None:
    roi = build_hind_limb_roi(
        BoundingBox(x_min=0, y_min=0, x_max=100, y_max=100),
        movement_direction=MovementDirection.RIGHT,
        detection_confidence=0.7,
        image_width=100,
        image_height=100,
        padding_fraction=0.1,
    )

    assert roi.box.x_min == 0
    assert roi.box.y_min >= 0
    assert roi.box.x_max <= 100
    assert roi.box.y_max == 100


@pytest.mark.parametrize("confidence", [-0.1, 1.1])
def test_roi_rejects_invalid_confidence(confidence: float) -> None:
    with pytest.raises(ValueError, match="detection_confidence"):
        build_hind_limb_roi(
            BoundingBox(x_min=10, y_min=10, x_max=100, y_max=100),
            movement_direction=MovementDirection.RIGHT,
            detection_confidence=confidence,
            image_width=200,
            image_height=200,
        )


def test_roi_rejects_invalid_geometry_parameters() -> None:
    box = BoundingBox(x_min=10, y_min=10, x_max=100, y_max=100)

    with pytest.raises(ValueError, match="rear_fraction"):
        build_hind_limb_roi(
            box,
            movement_direction=MovementDirection.RIGHT,
            detection_confidence=0.9,
            image_width=200,
            image_height=200,
            rear_fraction=0.0,
        )


def _box_at_center(center_x: int, width: int = 100) -> BoundingBox:
    half_width = width // 2
    return BoundingBox(
        x_min=center_x - half_width,
        y_min=20,
        x_max=center_x + half_width,
        y_max=120,
    )


def test_direction_estimator_detects_rightward_motion() -> None:
    estimate = estimate_movement_direction(
        [_box_at_center(center) for center in (100, 110, 120, 130, 140, 150)]
    )

    assert estimate.direction is MovementDirection.RIGHT
    assert estimate.displacement_pixels == pytest.approx(40.0)
    assert estimate.normalized_displacement == pytest.approx(0.4)
    assert estimate.confidence == pytest.approx(1.0)
    assert estimate.valid_sample_count == 6


def test_direction_estimator_detects_leftward_motion() -> None:
    estimate = estimate_movement_direction(
        [_box_at_center(center) for center in (200, 190, 180, 170, 160)]
    )

    assert estimate.direction is MovementDirection.LEFT
    assert estimate.displacement_pixels < 0.0


def test_direction_estimator_returns_unknown_for_low_motion() -> None:
    estimate = estimate_movement_direction(
        [_box_at_center(center) for center in (100, 101, 100, 102, 101, 100)]
    )

    assert estimate.direction is None
    assert abs(estimate.normalized_displacement) < 0.05


def test_direction_estimator_tolerates_missing_detections() -> None:
    estimate = estimate_movement_direction(
        [
            _box_at_center(100),
            None,
            _box_at_center(115),
            _box_at_center(130),
            None,
            _box_at_center(145),
        ]
    )

    assert estimate.direction is MovementDirection.RIGHT
    assert estimate.valid_sample_count == 4


def test_direction_estimator_returns_unknown_with_too_few_samples() -> None:
    estimate = estimate_movement_direction(
        [_box_at_center(100), None, _box_at_center(150)]
    )

    assert estimate.direction is None
    assert estimate.confidence == 0.0
    assert estimate.valid_sample_count == 2


def test_direction_estimator_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="min_valid_samples"):
        estimate_movement_direction([], min_valid_samples=2)

    with pytest.raises(ValueError, match="min_normalized_displacement"):
        estimate_movement_direction([], min_normalized_displacement=0.0)
