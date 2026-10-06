import pytest

from canine_gait_cv.gait import (
    StrideInterval,
    build_mean_reference_trajectory,
    normalize_stride_profiles,
)


def _stride(
    index: int,
    start: int,
    toe_off: int,
    end: int,
) -> StrideInterval:
    duration = (end - start) / 10.0
    stance = (toe_off - start) / 10.0
    swing = (end - toe_off) / 10.0

    return StrideInterval(
        stride_index=index,
        contact_frame=start,
        toe_off_frame=toe_off,
        next_contact_frame=end,
        stride_duration_seconds=duration,
        stance_duration_seconds=stance,
        swing_duration_seconds=swing,
        stance_fraction=stance / duration,
        swing_fraction=swing / duration,
    )


def test_normalizes_different_stride_lengths():
    profiles = normalize_stride_profiles(
        frame_indices=list(range(11)),
        knee_angles=[
            float(frame)
            for frame in range(11)
        ],
        hock_angles=[
            float(frame * 2)
            for frame in range(11)
        ],
        strides=[
            _stride(0, 0, 2, 4),
            _stride(1, 4, 7, 10),
        ],
        sample_count=5,
    )

    assert len(profiles) == 2

    assert profiles[0].cycle_percent == (
        0.0,
        25.0,
        50.0,
        75.0,
        100.0,
    )

    assert (
        profiles[0].toe_off_percent
        == pytest.approx(50.0)
    )

    assert (
        profiles[1].toe_off_percent
        == pytest.approx(50.0)
    )


def test_builds_mean_reference_trajectory():
    profiles = normalize_stride_profiles(
        frame_indices=list(range(9)),
        knee_angles=[
            100.0,
            110.0,
            120.0,
            110.0,
            100.0,
            120.0,
            140.0,
            120.0,
            100.0,
        ],
        hock_angles=[
            80.0,
            90.0,
            100.0,
            90.0,
            80.0,
            100.0,
            120.0,
            100.0,
            80.0,
        ],
        strides=[
            _stride(0, 0, 2, 4),
            _stride(1, 4, 6, 8),
        ],
        sample_count=5,
    )

    reference = (
        build_mean_reference_trajectory(
            profiles
        )
    )

    assert reference.stride_count == 2

    assert (
        reference.knee_mean_degrees[0]
        == pytest.approx(100.0)
    )

    assert (
        reference.knee_mean_degrees[2]
        == pytest.approx(130.0)
    )

    assert (
        reference.hock_mean_degrees[2]
        == pytest.approx(110.0)
    )

    assert (
        reference.mean_toe_off_percent
        == pytest.approx(50.0)
    )


def test_rejects_stride_with_missing_boundaries():
    profiles = normalize_stride_profiles(
        frame_indices=list(range(5)),
        knee_angles=[
            None,
            110.0,
            120.0,
            110.0,
            100.0,
        ],
        hock_angles=[
            None,
            90.0,
            100.0,
            90.0,
            80.0,
        ],
        strides=[
            _stride(0, 0, 2, 4),
        ],
        sample_count=5,
        min_source_coverage=0.50,
    )

    assert profiles == ()


def test_requires_at_least_one_profile():
    with pytest.raises(
        ValueError,
        match="At least one",
    ):
        build_mean_reference_trajectory(
            []
        )
