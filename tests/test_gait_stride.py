import pytest

from canine_gait_cv.gait import (
    GaitEventType,
    GaitPhase,
    detect_hind_paw_gait_phases,
)


def _synthetic_result(
    *,
    scale: float = 1.0,
    direction: str = "right",
):
    # Swing, stance, swing, stance.
    relative_x = [
        0, 1, 2, 3,
        2, 1, 0, -1,
        0, 1, 2, 3,
        2, 1, 0, -1,
    ]

    if direction == "left":
        relative_x = [
            -value
            for value in relative_x
        ]

    return detect_hind_paw_gait_phases(
        frame_indices=list(
            range(len(relative_x))
        ),
        thigh_x=[0.0] * len(relative_x),
        thigh_y=[0.0] * len(relative_x),
        paw_x=[
            value * scale
            for value in relative_x
        ],
        paw_y=[
            4.0 * scale
        ] * len(relative_x),
        fps=10.0,
        forward_direction=direction,
        smoothing_window=1,
        velocity_window=1,
        velocity_deadband=0.0,
    )


def test_detects_events_and_complete_stride():
    result = _synthetic_result()

    assert [
        event.event_type
        for event in result.events
    ] == [
        GaitEventType.CONTACT,
        GaitEventType.TOE_OFF,
        GaitEventType.CONTACT,
    ]

    assert len(result.strides) == 1

    stride = result.strides[0]

    assert stride.contact_frame == 4
    assert stride.toe_off_frame == 8
    assert stride.next_contact_frame == 12

    assert (
        stride.stride_duration_seconds
        == pytest.approx(0.8)
    )
    assert (
        stride.stance_duration_seconds
        == pytest.approx(0.4)
    )
    assert (
        stride.swing_duration_seconds
        == pytest.approx(0.4)
    )
    assert (
        stride.stance_fraction
        == pytest.approx(0.5)
    )
    assert (
        stride.swing_fraction
        == pytest.approx(0.5)
    )


def test_direction_correction_preserves_phases():
    right = _synthetic_result(
        direction="right"
    )
    left = _synthetic_result(
        direction="left"
    )

    assert [
        frame.phase
        for frame in right.frames
    ] == [
        frame.phase
        for frame in left.frames
    ]


def test_detection_is_body_scale_independent():
    small = _synthetic_result(
        scale=10.0
    )
    large = _synthetic_result(
        scale=1000.0
    )

    assert [
        frame.phase
        for frame in small.frames
    ] == [
        frame.phase
        for frame in large.frames
    ]


def test_short_internal_gap_is_interpolated():
    result = detect_hind_paw_gait_phases(
        frame_indices=list(range(8)),
        thigh_x=[0.0] * 8,
        thigh_y=[0.0] * 8,
        paw_x=[
            0.0,
            1.0,
            None,
            3.0,
            2.0,
            1.0,
            0.0,
            -1.0,
        ],
        paw_y=[4.0] * 8,
        fps=10.0,
        forward_direction="right",
        max_interpolated_gap=1,
        smoothing_window=1,
        velocity_window=1,
        velocity_deadband=0.0,
    )

    assert (
        result.frames[2].relative_paw_x
        is not None
    )

    phases = {
        frame.phase
        for frame in result.frames
    }

    assert GaitPhase.SWING in phases
    assert GaitPhase.STANCE in phases


def test_invalid_configuration_is_rejected():
    common = {
        "frame_indices": [0],
        "thigh_x": [0.0],
        "thigh_y": [0.0],
        "paw_x": [1.0],
        "paw_y": [1.0],
    }

    with pytest.raises(
        ValueError,
        match="fps must be positive",
    ):
        detect_hind_paw_gait_phases(
            **common,
            fps=0.0,
            forward_direction="right",
        )

    with pytest.raises(
        ValueError,
        match="forward_direction",
    ):
        detect_hind_paw_gait_phases(
            **common,
            fps=30.0,
            forward_direction="up",
        )