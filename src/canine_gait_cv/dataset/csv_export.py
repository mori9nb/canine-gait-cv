from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Mapping
from canine_gait_cv.quality import DetectionQualityAssessment

from canine_gait_cv.pose import KeypointName, estimate_body_orientation
from canine_gait_cv.roi import build_pose_hind_limb_roi
from canine_gait_cv.tracking import TrackedPoseFrame


_BASE_FIELDS = (
    "frame_index",
    "timestamp_seconds",
    "frame_width",
    "frame_height",
    "detected",
    "track_id",
    "detection_index",
    "bbox_confidence",
    "accepted_for_training",
    "quality_score",
    "quality_flags",
    "reliable_keypoint_fraction",
    "bbox_x_min",
    "bbox_y_min",
    "bbox_x_max",
    "bbox_y_max",
    "orientation",
    "orientation_confidence",
    "roi_available",
    "roi_x_min",
    "roi_y_min",
    "roi_x_max",
    "roi_y_max",
    "roi_confidence",
    "roi_keypoint_count",
    "roi_used_fallback",
)

_KEYPOINT_FIELDS = tuple(
    field
    for name in KeypointName
    for field in (
        f"{name.value}_x",
        f"{name.value}_y",
        f"{name.value}_confidence",
    )
)

TRACKED_GAIT_CSV_FIELDS = _BASE_FIELDS + _KEYPOINT_FIELDS


def build_tracked_gait_csv_rows(
    frame: TrackedPoseFrame,
    *,
    timestamp_seconds: float,
    frame_width: int,
    frame_height: int,
    quality_by_detection_index: Mapping[
        int,
        DetectionQualityAssessment,
    ] | None = None,
    accepted_only: bool = False,
) -> list[dict[str, object]]:
    """Create one CSV row per observed dog, or one empty-frame row."""

    if timestamp_seconds < 0.0:
        raise ValueError("timestamp_seconds cannot be negative.")
    if frame_width <= 0 or frame_height <= 0:
        raise ValueError("frame dimensions must be positive.")
    if accepted_only and quality_by_detection_index is None:
        raise ValueError(
            "accepted_only requires quality assessments."
        )
    if not frame.individuals:
        return [
            _empty_row(
                frame_index=frame.frame_index,
                timestamp_seconds=timestamp_seconds,
                frame_width=frame_width,
                frame_height=frame_height,
            )
        ]

    rows: list[dict[str, object]] = []
    for tracked in frame.individuals:
        detection = tracked.detection
        quality = (
            None
            if quality_by_detection_index is None
            else quality_by_detection_index.get(
                detection.detection_index
            )
        )

        if accepted_only and (
            quality is None
            or not quality.accepted_for_training
        ):
            continue
        orientation = estimate_body_orientation(detection.pose, detection.dog_box)
        roi = build_pose_hind_limb_roi(
            detection.pose,
            detection.dog_box,
            orientation=orientation.orientation,
            image_width=frame_width,
            image_height=frame_height,
        )
        row = _empty_row(
            frame_index=frame.frame_index,
            timestamp_seconds=timestamp_seconds,
            frame_width=frame_width,
            frame_height=frame_height,
        )
        row.update(
            {
                "detected": True,
                "track_id": tracked.track_id,
                "detection_index": detection.detection_index,
                "bbox_confidence": round(detection.bbox_confidence, 6),
                "bbox_x_min": detection.dog_box.x_min,
                "bbox_y_min": detection.dog_box.y_min,
                "bbox_x_max": detection.dog_box.x_max,
                "bbox_y_max": detection.dog_box.y_max,
                "orientation": orientation.orientation.value,
                "orientation_confidence": round(orientation.confidence, 6),
            }
        )
        if quality is not None:
            row.update(
                {
                    "accepted_for_training": (
                        quality.accepted_for_training
                    ),
                    "quality_score": round(
                        quality.quality_score,
                        6,
                    ),
                    "quality_flags": "|".join(
                        flag.value
                        for flag in quality.flags
                    ),
                    "reliable_keypoint_fraction": round(
                        quality.reliable_keypoint_fraction,
                        6,
                    ),
                }
            )
        if roi is not None:
            row.update(
                {
                    "roi_available": True,
                    "roi_x_min": roi.box.x_min,
                    "roi_y_min": roi.box.y_min,
                    "roi_x_max": roi.box.x_max,
                    "roi_y_max": roi.box.y_max,
                    "roi_confidence": round(roi.confidence, 6),
                    "roi_keypoint_count": roi.keypoint_count,
                    "roi_used_fallback": roi.used_fallback,
                }
            )
        for name, point in detection.pose.keypoints.items():
            prefix = name.value
            row[f"{prefix}_x"] = round(point.x, 6)
            row[f"{prefix}_y"] = round(point.y, 6)
            row[f"{prefix}_confidence"] = round(point.confidence, 6)
        rows.append(row)
        
    if rows:
        return rows

    return [
        _empty_row(
            frame_index=frame.frame_index,
            timestamp_seconds=timestamp_seconds,
            frame_width=frame_width,
            frame_height=frame_height,
        )
    ]


def write_tracked_gait_csv(
    path: str | Path,
    rows: Iterable[dict[str, object]],
) -> None:
    """Write tracked gait rows using the stable project CSV schema."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=TRACKED_GAIT_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _empty_row(
    *,
    frame_index: int,
    timestamp_seconds: float,
    frame_width: int,
    frame_height: int,
) -> dict[str, object]:
    row: dict[str, object] = {
        field: "" for field in TRACKED_GAIT_CSV_FIELDS
    }
    row.update(
        {
            "frame_index": frame_index,
            "timestamp_seconds": round(timestamp_seconds, 6),
            "frame_width": frame_width,
            "frame_height": frame_height,
            "detected": False,
            "roi_available": False,
        }
    )
    return row
