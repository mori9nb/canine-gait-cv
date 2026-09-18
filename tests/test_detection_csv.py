from canine_gait_cv.detection import Detection
from canine_gait_cv.detection.csv_export import (
    DETECTION_CSV_FIELDS,
    build_detection_csv_row,
)
from canine_gait_cv.preprocessing import BoundingBox


def test_build_detection_csv_row_with_detection() -> None:
    detection = Detection(
        class_name="dog",
        confidence=0.91234567,
        box=BoundingBox(
            x_min=100,
            y_min=200,
            x_max=500,
            y_max=600,
        ),
    )

    row = build_detection_csv_row(
        frame_index=12,
        timestamp_seconds=0.4000001,
        frame_width=1280,
        frame_height=720,
        detection=detection,
    )

    assert tuple(row.keys()) == DETECTION_CSV_FIELDS
    assert row == {
        "frame_index": 12,
        "timestamp_seconds": 0.4,
        "frame_width": 1280,
        "frame_height": 720,
        "detected": True,
        "class_name": "dog",
        "confidence": 0.912346,
        "x_min": 100,
        "y_min": 200,
        "x_max": 500,
        "y_max": 600,
    }


def test_build_detection_csv_row_without_detection() -> None:
    row = build_detection_csv_row(
        frame_index=9,
        timestamp_seconds=0.3,
        frame_width=1280,
        frame_height=720,
        detection=None,
    )

    assert tuple(row.keys()) == DETECTION_CSV_FIELDS
    assert row == {
        "frame_index": 9,
        "timestamp_seconds": 0.3,
        "frame_width": 1280,
        "frame_height": 720,
        "detected": False,
        "class_name": "",
        "confidence": "",
        "x_min": "",
        "y_min": "",
        "x_max": "",
        "y_max": "",
    }