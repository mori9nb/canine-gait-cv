import pytest

from canine_gait_cv.pose import DeepLabCutIndividual, DeepLabCutMultiPoseFrame, DogPose
from canine_gait_cv.preprocessing import BoundingBox
from canine_gait_cv.tracking import MultiDogTracker
from canine_gait_cv.appearance import (
    AppearanceEmbedding,
    AppearanceGallery,
)
from canine_gait_cv.quality import (
    DetectionQualityAssessment,
    DetectionQualityFlag,
)

def _dog(detection_index: int, x: int) -> DeepLabCutIndividual:
    return DeepLabCutIndividual(
        detection_index=detection_index,
        dog_box=BoundingBox(x, 20, x + 100, 120),
        bbox_confidence=0.99,
        pose=DogPose({}),
    )


def _frame(index: int, *dogs: DeepLabCutIndividual) -> DeepLabCutMultiPoseFrame:
    return DeepLabCutMultiPoseFrame(frame_index=index, individuals=dogs)

def _embedding(x: float, y: float) -> AppearanceEmbedding:
    return AppearanceEmbedding(
        vector=(x, y),
        crop_box=BoundingBox(0, 0, 20, 20),
    )


def _quality(
    detection_index: int,
    *,
    accepted: bool = True,
) -> DetectionQualityAssessment:
    return DetectionQualityAssessment(
        detection_index=detection_index,
        accepted_for_training=accepted,
        flags=(
            (DetectionQualityFlag.VALID,)
            if accepted
            else (DetectionQualityFlag.OVERLAPPING_DETECTION,)
        ),
        bbox_confidence=0.99,
        reliable_keypoint_fraction=1.0,
        quality_score=0.99,
    )

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

def test_max_active_tracks_discards_extra_unmatched_detection() -> None:
    tracker = MultiDogTracker(max_active_tracks=3)

    tracker.update(
        _frame(
            0,
            _dog(0, 10),
            _dog(1, 200),
            _dog(2, 400),
        )
    )

    result = tracker.update(
        _frame(
            1,
            _dog(0, 15),
            _dog(1, 205),
            _dog(2, 405),
            _dog(3, 700),
        )
    )

    assert len(result.individuals) == 3
    assert {item.track_id for item in result.individuals} == {0, 1, 2}

def test_max_active_tracks_must_be_positive() -> None:
    with pytest.raises(ValueError, match="max_active_tracks must be positive"):
        MultiDogTracker(max_active_tracks=0)

def test_appearance_prevents_identity_swap_when_positions_are_ambiguous() -> None:
    gallery = AppearanceGallery()
    tracker = MultiDogTracker(
        appearance_gallery=gallery,
        appearance_weight=2.0,
    )

    tracker.update(
        _frame(0, _dog(0, 0), _dog(1, 200)),
        appearance_embeddings={
            0: _embedding(1.0, 0.0),
            1: _embedding(0.0, 1.0),
        },
        quality_assessments={
            0: _quality(0),
            1: _quality(1),
        },
    )

    result = tracker.update(
        _frame(1, _dog(0, 100), _dog(1, 100)),
        appearance_embeddings={
            0: _embedding(0.0, 1.0),
            1: _embedding(1.0, 0.0),
        },
        quality_assessments={
            0: _quality(0),
            1: _quality(1),
        },
    )

    assert [item.track_id for item in result.individuals] == [1, 0]


def test_overlapping_detection_does_not_pollute_appearance_gallery() -> None:
    gallery = AppearanceGallery()
    tracker = MultiDogTracker(
        appearance_gallery=gallery,
        appearance_weight=2.0,
    )

    tracker.update(
        _frame(0, _dog(0, 10)),
        appearance_embeddings={
            0: _embedding(1.0, 0.0),
        },
        quality_assessments={
            0: _quality(0),
        },
    )

    tracker.update(
        _frame(1, _dog(0, 15)),
        appearance_embeddings={
            0: _embedding(0.0, 1.0),
        },
        quality_assessments={
            0: _quality(0, accepted=False),
        },
    )

    assert gallery.sample_count(0) == 1
    assert gallery.prototype(0) == pytest.approx((1.0, 0.0))