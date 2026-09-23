import pytest

from canine_gait_cv.pose import (
    DeepLabCutIndividual,
    DogKeypoint,
    DogPose,
    KeypointName,
)
from canine_gait_cv.preprocessing import BoundingBox
from canine_gait_cv.smoothing import smooth_tracked_pose_frames
from canine_gait_cv.tracking import TrackedIndividual, TrackedPoseFrame


def _tracked(track_id: int, x: float, *, confidence: float = 0.9) -> TrackedIndividual:
    detection = DeepLabCutIndividual(
        detection_index=track_id,
        dog_box=BoundingBox(0, 0, 200, 200),
        bbox_confidence=0.99,
        pose=DogPose(
            {
                KeypointName.LEFT_KNEE: DogKeypoint(x, 50.0, confidence),
                KeypointName.LEFT_BACK_PAW: DogKeypoint(x + 10.0, 100.0, confidence),
            }
        ),
    )
    return TrackedIndividual(track_id=track_id, detection=detection)


def _frame(index: int, *individuals: TrackedIndividual) -> TrackedPoseFrame:
    return TrackedPoseFrame(frame_index=index, individuals=individuals)


def test_empty_sequence_returns_empty_sequence() -> None:
    assert smooth_tracked_pose_frames([]) == []


def test_all_keypoints_are_smoothed_for_one_track() -> None:
    frames = [_frame(i, _tracked(4, x)) for i, x in enumerate([0, 10, 20, 30])]

    result = smooth_tracked_pose_frames(frames)

    last = result[-1].individuals[0]
    assert last.track_id == 4
    assert KeypointName.LEFT_KNEE in last.keypoints
    assert KeypointName.LEFT_BACK_PAW in last.keypoints


def test_tracks_are_smoothed_independently() -> None:
    frames = [
        _frame(0, _tracked(1, 0), _tracked(2, 300)),
        _frame(1, _tracked(1, 10), _tracked(2, 290)),
        _frame(2, _tracked(1, 20), _tracked(2, 280)),
    ]

    result = smooth_tracked_pose_frames(frames)

    first_x = result[-1].individuals[0].keypoints[KeypointName.LEFT_KNEE].x
    second_x = result[-1].individuals[1].keypoints[KeypointName.LEFT_KNEE].x
    assert first_x < 100
    assert second_x > 200


def test_raw_deeplabcut_pose_is_preserved() -> None:
    frames = [_frame(0, _tracked(3, 10)), _frame(1, _tracked(3, 200, confidence=0.05))]

    result = smooth_tracked_pose_frames(frames)

    raw = result[1].individuals[0].tracked.detection.pose.get(KeypointName.LEFT_KNEE)
    assert raw.x == 200
    assert raw.confidence == pytest.approx(0.05)
    assert result[1].individuals[0].keypoints[KeypointName.LEFT_KNEE].measurement_used is False


def test_missing_global_frame_is_accounted_for_in_timeline() -> None:
    frames = [
        _frame(0, _tracked(0, 0)),
        _frame(2, _tracked(0, 20)),
    ]

    result = smooth_tracked_pose_frames(frames)

    assert [frame.frame_index for frame in result] == [0, 2]
    assert len(result) == 2


def test_duplicate_track_id_in_one_frame_is_rejected() -> None:
    frames = [_frame(0, _tracked(5, 10), _tracked(5, 20))]

    with pytest.raises(ValueError, match="duplicate track_id"):
        smooth_tracked_pose_frames(frames)


def test_frame_indices_must_increase() -> None:
    frames = [_frame(1, _tracked(0, 10)), _frame(1, _tracked(0, 20))]

    with pytest.raises(ValueError, match="strictly increasing"):
        smooth_tracked_pose_frames(frames)
