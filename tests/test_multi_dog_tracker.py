import pytest

from canine_gait_cv.pose import DeepLabCutIndividual, DeepLabCutMultiPoseFrame, DogPose
from canine_gait_cv.preprocessing import BoundingBox
from canine_gait_cv.tracking import MultiDogTracker


def _dog(detection_index: int, x: int) -> DeepLabCutIndividual:
    return DeepLabCutIndividual(
        detection_index=detection_index,
        dog_box=BoundingBox(x, 20, x + 100, 120),
        bbox_confidence=0.99,
        pose=DogPose({}),
    )


def _frame(index: int, *dogs: DeepLabCutIndividual) -> DeepLabCutMultiPoseFrame:
    return DeepLabCutMultiPoseFrame(frame_index=index, individuals=dogs)


def test_first_detections_receive_new_track_ids() -> None:
    result = MultiDogTracker().update(_frame(0, _dog(0, 10), _dog(1, 300)))

    assert [item.track_id for item in result.individuals] == [0, 1]


def test_detection_order_can_swap_without_swapping_identity() -> None:
    tracker = MultiDogTracker()
    tracker.update(_frame(0, _dog(0, 10), _dog(1, 300)))
    result = tracker.update(_frame(1, _dog(0, 305), _dog(1, 15)))

    assert [item.track_id for item in result.individuals] == [1, 0]


def test_track_survives_one_empty_frame() -> None:
    tracker = MultiDogTracker(max_missed_frames=2)
    tracker.update(_frame(0, _dog(0, 10)))
    tracker.update(_frame(1))
    result = tracker.update(_frame(2, _dog(0, 20)))

    assert result.individuals[0].track_id == 0


def test_expired_track_is_not_reused() -> None:
    tracker = MultiDogTracker(max_missed_frames=1)
    tracker.update(_frame(0, _dog(0, 10)))
    tracker.update(_frame(1))
    tracker.update(_frame(2))
    result = tracker.update(_frame(3, _dog(0, 10)))

    assert result.individuals[0].track_id == 1


def test_large_jump_starts_a_new_track() -> None:
    tracker = MultiDogTracker(max_normalized_distance=0.5)
    tracker.update(_frame(0, _dog(0, 10)))
    result = tracker.update(_frame(1, _dog(0, 500)))

    assert result.individuals[0].track_id == 1


def test_frame_indices_must_be_strictly_increasing() -> None:
    tracker = MultiDogTracker()
    tracker.update(_frame(2, _dog(0, 10)))

    with pytest.raises(ValueError, match="strictly increasing"):
        tracker.update(_frame(2, _dog(0, 12)))


def test_reset_restarts_track_ids() -> None:
    tracker = MultiDogTracker()
    tracker.update(_frame(0, _dog(0, 10)))
    tracker.reset()
    result = tracker.update(_frame(0, _dog(0, 300)))

    assert result.individuals[0].track_id == 0
