from __future__ import annotations

from canine_gait_cv.detection.types import Detection


DETECTION_CSV_FIELDS: tuple[str, ...] = (
    "frame_index",
    "timestamp_seconds",
    "frame_width",
    "frame_height",
    "detected",
    "class_name",
    "confidence",
    "x_min",
    "y_min",
    "x_max",
    "y_max",
)


def build_detection_csv_row(
    *,
    frame_index: int,
    timestamp_seconds: float,
    frame_width: int,
    frame_height: int,
    detection: Detection | None,
) -> dict[str, object]:
    """Convert one frame detection result into a CSV-compatible row."""

    row: dict[str, object] = {
        "frame_index": frame_index,
        "timestamp_seconds": round(timestamp_seconds, 6),
        "frame_width": frame_width,
        "frame_height": frame_height,
        "detected": detection is not None,
        "class_name": "",
        "confidence": "",
        "x_min": "",
        "y_min": "",
        "x_max": "",
        "y_max": "",
    }

    if detection is None:
        return row

    row.update(
        {
            "class_name": detection.class_name,
            "confidence": round(detection.confidence, 6),
            "x_min": detection.box.x_min,
            "y_min": detection.box.y_min,
            "x_max": detection.box.x_max,
            "y_max": detection.box.y_max,
        }
    )

    return row