from __future__ import annotations

import pytest

from canine_gait_cv.appearance import (
    AppearanceEmbedding,
    AppearanceGallery,
)
from canine_gait_cv.preprocessing import BoundingBox
from canine_gait_cv.quality import (
    DetectionQualityAssessment,
    DetectionQualityFlag,
)


BOX = BoundingBox(0, 0, 20, 20)


def embedding(*values: float) -> AppearanceEmbedding:
    return AppearanceEmbedding(
        vector=tuple(values),
        crop_box=BOX,
    )


def quality(
    *,
    accepted: bool = True,
    score: float = 0.90,
    flags: tuple[DetectionQualityFlag, ...] = (
        DetectionQualityFlag.VALID,
    ),
) -> DetectionQualityAssessment:
    return DetectionQualityAssessment(
        detection_index=0,
        accepted_for_training=accepted,
        flags=flags,
        bbox_confidence=0.95,
        reliable_keypoint_fraction=0.95,
        quality_score=score,
    )


def test_clean_embedding_is_added_to_gallery() -> None:
    gallery = AppearanceGallery()

    added = gallery.update(
        3,
        embedding(3.0, 4.0),
        quality(),
    )

    assert added is True
    assert gallery.contains(3)
    assert gallery.sample_count(3) == 1
    assert gallery.prototype(3) == pytest.approx((0.6, 0.8))


def test_uncertain_detection_does_not_update_gallery() -> None:
    gallery = AppearanceGallery()

    added = gallery.update(
        1,
        embedding(1.0, 0.0),
        quality(
            accepted=False,
            flags=(
                DetectionQualityFlag.OVERLAPPING_DETECTION,
            ),
        ),
    )

    assert added is False
    assert not gallery.contains(1)


def test_low_quality_score_does_not_update_gallery() -> None:
    gallery = AppearanceGallery(min_quality_score=0.70)

    added = gallery.update(
        1,
        embedding(1.0, 0.0),
        quality(score=0.60),
    )

    assert added is False
    assert gallery.sample_count(1) == 0


def test_gallery_keeps_only_recent_samples() -> None:
    gallery = AppearanceGallery(max_samples_per_track=2)

    gallery.update(0, embedding(1.0, 0.0), quality())
    gallery.update(0, embedding(0.8, 0.2), quality())
    gallery.update(0, embedding(0.0, 1.0), quality())

    assert gallery.sample_count(0) == 2


def test_similarity_uses_normalized_prototype() -> None:
    gallery = AppearanceGallery()

    gallery.update(0, embedding(1.0, 0.0), quality())
    gallery.update(0, embedding(0.8, 0.2), quality())

    same = gallery.similarity(
        0,
        embedding(1.0, 0.0),
    )
    different = gallery.similarity(
        0,
        embedding(0.0, 1.0),
    )

    assert same > different
    assert same > 0.90


def test_incompatible_dimensions_are_rejected() -> None:
    gallery = AppearanceGallery()

    gallery.update(0, embedding(1.0, 0.0), quality())

    with pytest.raises(ValueError, match="same dimension"):
        gallery.update(
            1,
            embedding(1.0, 0.0, 0.0),
            quality(),
        )


def test_invalid_configuration_is_rejected() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        AppearanceGallery(max_samples_per_track=0)

    with pytest.raises(ValueError, match="between 0 and 1"):
        AppearanceGallery(min_quality_score=1.1)


def test_missing_track_has_no_prototype() -> None:
    gallery = AppearanceGallery()

    with pytest.raises(KeyError, match="no appearance samples"):
        gallery.prototype(8)