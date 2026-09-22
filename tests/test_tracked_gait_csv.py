import csv

import pytest

from canine_gait_cv.dataset import (
    TRACKED_GAIT_CSV_FIELDS,
    build_tracked_gait_csv_rows,
    write_tracked_gait_csv,
)
from canine_gait_cv.pose import (
    DeepLabCutIndividual,
    DogKeypoint,
    DogPose,
    KeypointName,
)
from canine_gait_cv.preprocessing import BoundingBox
from canine_gait_cv.tracking import TrackedIndividual, TrackedPoseFrame


def _tracked_frame() -> TrackedPoseFrame:
    pose = DogPose(
        {
            KeypointName.NOSE: DogKeypoint(180.1234567, 40.0, 0.95),
            KeypointName.TAIL_ROOT: DogKeypoint(40.0, 60.0, 0.90),
            KeypointName.LEFT_HIP: DogKeypoint(60.0, 80.0, 0.90),
            KeypointName.LEFT_KNEE: DogKeypoint(70.0, 120.0, 0.85),
            KeypointName.LEFT_BACK_PAW: DogKeypoint(80.0, 170.0, 0.80),
        }
    )
    detection = DeepLabCutIndividual(
        detection_index=3,
        dog_box=BoundingBox(20, 20, 220, 200),
        bbox_confidence=0.9876543,
        pose=pose,
    )
    return TrackedPoseFrame(
        frame_index=12,
        individuals=(TrackedIndividual(track_id=7, detection=detection),),
    )


def test_detected_row_contains_tracking_pose_orientation_and_roi() -> None:
    rows = build_tracked_gait_csv_rows(
        _tracked_frame(),
        timestamp_seconds=0.4000001,
        frame_width=640,
        frame_height=480,
    )

    assert len(rows) == 1
    row = rows[0]
    assert tuple(row.keys()) == TRACKED_GAIT_CSV_FIELDS
    assert row["frame_index"] == 12
    assert row["timestamp_seconds"] == 0.4
    assert row["track_id"] == 7
    assert row["detection_index"] == 3
    assert row["bbox_confidence"] == 0.987654
    assert row["orientation"] == "right"
    assert row["roi_available"] is True
    assert row["roi_keypoint_count"] == 3
    assert row["nose_x"] == 180.123457
    assert row["right_back_paw_x"] == ""


def test_empty_frame_is_preserved_in_dataset() -> None:
    rows = build_tracked_gait_csv_rows(
        TrackedPoseFrame(frame_index=8, individuals=()),
        timestamp_seconds=0.25,
        frame_width=1280,
        frame_height=720,
    )

    assert len(rows) == 1
    assert rows[0]["detected"] is False
    assert rows[0]["track_id"] == ""
    assert rows[0]["roi_available"] is False


def test_multiple_dogs_create_multiple_rows() -> None:
    first = _tracked_frame().individuals[0]
    second = TrackedIndividual(track_id=8, detection=first.detection)
    frame = TrackedPoseFrame(frame_index=12, individuals=(first, second))

    rows = build_tracked_gait_csv_rows(
        frame,
        timestamp_seconds=0.4,
        frame_width=640,
        frame_height=480,
    )

    assert [row["track_id"] for row in rows] == [7, 8]


def test_invalid_metadata_is_rejected() -> None:
    with pytest.raises(ValueError, match="timestamp"):
        build_tracked_gait_csv_rows(
            _tracked_frame(),
            timestamp_seconds=-0.1,
            frame_width=640,
            frame_height=480,
        )


def test_write_tracked_gait_csv_writes_stable_header(tmp_path) -> None:
    rows = build_tracked_gait_csv_rows(
        _tracked_frame(),
        timestamp_seconds=0.4,
        frame_width=640,
        frame_height=480,
    )
    path = tmp_path / "tracked.csv"

    write_tracked_gait_csv(path, rows)

    with path.open(newline="", encoding="utf-8") as stream:
        saved = list(csv.DictReader(stream))
    assert tuple(saved[0].keys()) == TRACKED_GAIT_CSV_FIELDS
    assert saved[0]["track_id"] == "7"
    assert saved[0]["orientation"] == "right"
