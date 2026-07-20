import pytest

from canine_gait_cv.detection import (
    Detection,
    filter_detections_by_class,
    filter_detections_by_confidence,
    select_largest_detection,
)
from canine_gait_cv.preprocessing import BoundingBox


def test_detection_can_be_created() -> None:
    detection = Detection(
        class_name="dog",
        confidence=0.95,
        box=BoundingBox(x_min=10, y_min=20, x_max=110, y_max=120),
    )

    assert detection.class_name == "dog"
    assert detection.confidence == 0.95
    assert detection.box.area == 10000


def test_detection_rejects_invalid_confidence() -> None:
    with pytest.raises(ValueError):
        Detection(
            class_name="dog",
            confidence=1.5,
            box=BoundingBox(x_min=10, y_min=20, x_max=110, y_max=120),
        )


def test_filter_detections_by_class() -> None:
    detections = [
        Detection(
            class_name="dog",
            confidence=0.90,
            box=BoundingBox(x_min=0, y_min=0, x_max=100, y_max=100),
        ),
        Detection(
            class_name="cat",
            confidence=0.80,
            box=BoundingBox(x_min=0, y_min=0, x_max=50, y_max=50),
        ),
    ]

    dog_detections = filter_detections_by_class(detections, "dog")

    assert len(dog_detections) == 1
    assert dog_detections[0].class_name == "dog"


def test_filter_detections_by_confidence() -> None:
    detections = [
        Detection(
            class_name="dog",
            confidence=0.95,
            box=BoundingBox(x_min=0, y_min=0, x_max=100, y_max=100),
        ),
        Detection(
            class_name="dog",
            confidence=0.40,
            box=BoundingBox(x_min=0, y_min=0, x_max=50, y_max=50),
        ),
    ]

    confident_detections = filter_detections_by_confidence(
        detections,
        min_confidence=0.50,
    )

    assert len(confident_detections) == 1
    assert confident_detections[0].confidence == 0.95


def test_select_largest_detection() -> None:
    small_detection = Detection(
        class_name="dog",
        confidence=0.90,
        box=BoundingBox(x_min=0, y_min=0, x_max=50, y_max=50),
    )
    large_detection = Detection(
        class_name="dog",
        confidence=0.85,
        box=BoundingBox(x_min=0, y_min=0, x_max=200, y_max=100),
    )

    selected = select_largest_detection([small_detection, large_detection])

    assert selected == large_detection


def test_select_largest_detection_returns_none_for_empty_list() -> None:
    selected = select_largest_detection([])

    assert selected is None