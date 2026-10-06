from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence

import numpy as np


class GaitPhase(str, Enum):
    STANCE = "stance"
    SWING = "swing"
    UNKNOWN = "unknown"


class GaitEventType(str, Enum):
    CONTACT = "contact"
    TOE_OFF = "toe_off"


@dataclass(frozen=True)
class GaitPhaseFrame:
    frame_index: int
    time_seconds: float
    phase: GaitPhase
    relative_paw_x: float | None
    relative_paw_y: float | None
    forward_velocity: float | None


@dataclass(frozen=True)
class GaitEvent:
    frame_index: int
    time_seconds: float
    event_type: GaitEventType


@dataclass(frozen=True)
class StrideInterval:
    stride_index: int
    contact_frame: int
    toe_off_frame: int
    next_contact_frame: int
    stride_duration_seconds: float
    stance_duration_seconds: float
    swing_duration_seconds: float
    stance_fraction: float
    swing_fraction: float


@dataclass(frozen=True)
class GaitPhaseResult:
    frames: tuple[GaitPhaseFrame, ...]
    events: tuple[GaitEvent, ...]
    strides: tuple[StrideInterval, ...]
    body_scale_pixels: float
    forward_direction: str


def detect_hind_paw_gait_phases(
    *,
    frame_indices: Sequence[int],
    thigh_x: Sequence[float | None],
    thigh_y: Sequence[float | None],
    paw_x: Sequence[float | None],
    paw_y: Sequence[float | None],
    fps: float,
    forward_direction: str,
    max_interpolated_gap: int = 3,
    smoothing_window: int = 5,
    velocity_window: int = 5,
    velocity_deadband: float = 0.10,
) -> GaitPhaseResult:
    """Detect stance, swing, contact and toe-off.

    Paw position is measured relative to the proximal hind-limb point and
    normalized by median thigh-to-paw length. This reduces sensitivity to
    image resolution, dog size, global translation and moderate camera motion.

    Relative motion in the anatomical forward direction represents swing.
    Relative motion opposite to the forward direction represents stance.
    """

    if fps <= 0.0:
        raise ValueError("fps must be positive.")

    if forward_direction not in {"left", "right"}:
        raise ValueError(
            "forward_direction must be 'left' or 'right'."
        )

    if max_interpolated_gap < 0:
        raise ValueError(
            "max_interpolated_gap cannot be negative."
        )

    if velocity_deadband < 0.0:
        raise ValueError(
            "velocity_deadband cannot be negative."
        )

    _validate_window(
        "smoothing_window",
        smoothing_window,
    )
    _validate_window(
        "velocity_window",
        velocity_window,
    )

    lengths = {
        len(frame_indices),
        len(thigh_x),
        len(thigh_y),
        len(paw_x),
        len(paw_y),
    }

    if len(lengths) != 1:
        raise ValueError(
            "All input trajectories must have the same length."
        )

    if not frame_indices:
        return GaitPhaseResult(
            frames=(),
            events=(),
            strides=(),
            body_scale_pixels=0.0,
            forward_direction=forward_direction,
        )

    if any(
        current <= previous
        for previous, current in zip(
            frame_indices,
            frame_indices[1:],
        )
    ):
        raise ValueError(
            "frame_indices must be strictly increasing."
        )

    thigh_x_array = _interpolate_short_gaps(
        _as_float_array(thigh_x),
        max_gap=max_interpolated_gap,
    )
    thigh_y_array = _interpolate_short_gaps(
        _as_float_array(thigh_y),
        max_gap=max_interpolated_gap,
    )
    paw_x_array = _interpolate_short_gaps(
        _as_float_array(paw_x),
        max_gap=max_interpolated_gap,
    )
    paw_y_array = _interpolate_short_gaps(
        _as_float_array(paw_y),
        max_gap=max_interpolated_gap,
    )

    relative_x_pixels = (
        paw_x_array - thigh_x_array
    )
    relative_y_pixels = (
        paw_y_array - thigh_y_array
    )

    limb_length = np.hypot(
        relative_x_pixels,
        relative_y_pixels,
    )

    body_scale = float(
        np.nanmedian(limb_length)
    )

    if (
        not np.isfinite(body_scale)
        or body_scale <= 0.0
    ):
        raise ValueError(
            "Cannot estimate a positive thigh-to-paw scale."
        )

    relative_x = _rolling_statistic(
        relative_x_pixels / body_scale,
        window=smoothing_window,
        statistic="median",
    )
    relative_y = _rolling_statistic(
        relative_y_pixels / body_scale,
        window=smoothing_window,
        statistic="median",
    )

    times = (
        np.asarray(frame_indices, dtype=float)
        / fps
    )

    horizontal_velocity = _finite_difference(
        relative_x,
        times,
    )

    horizontal_velocity = _rolling_statistic(
        horizontal_velocity,
        window=velocity_window,
        statistic="mean",
    )

    # Correct the image-x sign using the dog's movement direction.
    direction_sign = (
        1.0
        if forward_direction == "right"
        else -1.0
    )

    forward_velocity = (
        horizontal_velocity
        * direction_sign
    )

    phases = _classify_phases(
        forward_velocity,
        deadband=velocity_deadband,
    )

    phase_frames = tuple(
        GaitPhaseFrame(
            frame_index=int(frame_index),
            time_seconds=float(time),
            phase=phase,
            relative_paw_x=_optional_float(
                relative_x_value
            ),
            relative_paw_y=_optional_float(
                relative_y_value
            ),
            forward_velocity=_optional_float(
                velocity
            ),
        )
        for (
            frame_index,
            time,
            phase,
            relative_x_value,
            relative_y_value,
            velocity,
        ) in zip(
            frame_indices,
            times,
            phases,
            relative_x,
            relative_y,
            forward_velocity,
        )
    )

    events = _build_events(phase_frames)

    strides = _build_strides(
        events,
        fps=fps,
    )

    return GaitPhaseResult(
        frames=phase_frames,
        events=events,
        strides=strides,
        body_scale_pixels=body_scale,
        forward_direction=forward_direction,
    )


def _classify_phases(
    velocity: np.ndarray,
    *,
    deadband: float,
) -> list[GaitPhase]:
    phases: list[GaitPhase] = []
    previous_phase = GaitPhase.UNKNOWN

    for value in velocity:
        if not np.isfinite(value):
            phase = GaitPhase.UNKNOWN
        elif value > deadband:
            phase = GaitPhase.SWING
        elif value < -deadband:
            phase = GaitPhase.STANCE
        else:
            # Keep the previous phase close to the zero crossing.
            phase = previous_phase

        phases.append(phase)

        if phase is not GaitPhase.UNKNOWN:
            previous_phase = phase

    return phases


def _build_events(
    frames: tuple[GaitPhaseFrame, ...],
) -> tuple[GaitEvent, ...]:
    events: list[GaitEvent] = []

    for previous, current in zip(
        frames,
        frames[1:],
    ):
        if (
            previous.phase is GaitPhase.SWING
            and current.phase is GaitPhase.STANCE
        ):
            event_type = GaitEventType.CONTACT

        elif (
            previous.phase is GaitPhase.STANCE
            and current.phase is GaitPhase.SWING
        ):
            event_type = GaitEventType.TOE_OFF

        else:
            continue

        events.append(
            GaitEvent(
                frame_index=current.frame_index,
                time_seconds=current.time_seconds,
                event_type=event_type,
            )
        )

    return tuple(events)


def _build_strides(
    events: tuple[GaitEvent, ...],
    *,
    fps: float,
) -> tuple[StrideInterval, ...]:
    contacts = [
        event
        for event in events
        if event.event_type
        is GaitEventType.CONTACT
    ]

    toe_offs = [
        event
        for event in events
        if event.event_type
        is GaitEventType.TOE_OFF
    ]

    strides: list[StrideInterval] = []

    for contact, next_contact in zip(
        contacts,
        contacts[1:],
    ):
        toe_off = next(
            (
                event
                for event in toe_offs
                if (
                    contact.frame_index
                    < event.frame_index
                    < next_contact.frame_index
                )
            ),
            None,
        )

        if toe_off is None:
            continue

        stride_frame_count = (
            next_contact.frame_index
            - contact.frame_index
        )
        stance_frame_count = (
            toe_off.frame_index
            - contact.frame_index
        )
        swing_frame_count = (
            next_contact.frame_index
            - toe_off.frame_index
        )

        stride_duration = (
            stride_frame_count / fps
        )
        stance_duration = (
            stance_frame_count / fps
        )
        swing_duration = (
            swing_frame_count / fps
        )

        strides.append(
            StrideInterval(
                stride_index=len(strides),
                contact_frame=(
                    contact.frame_index
                ),
                toe_off_frame=(
                    toe_off.frame_index
                ),
                next_contact_frame=(
                    next_contact.frame_index
                ),
                stride_duration_seconds=(
                    stride_duration
                ),
                stance_duration_seconds=(
                    stance_duration
                ),
                swing_duration_seconds=(
                    swing_duration
                ),
                stance_fraction=(
                    stance_duration
                    / stride_duration
                ),
                swing_fraction=(
                    swing_duration
                    / stride_duration
                ),
            )
        )

    return tuple(strides)


def _finite_difference(
    values: np.ndarray,
    times: np.ndarray,
) -> np.ndarray:
    result = np.full(
        len(values),
        np.nan,
        dtype=float,
    )

    for index in range(1, len(values)):
        if (
            not np.isfinite(values[index])
            or not np.isfinite(
                values[index - 1]
            )
        ):
            continue

        delta_time = (
            times[index]
            - times[index - 1]
        )

        if delta_time <= 0.0:
            continue

        result[index] = (
            values[index]
            - values[index - 1]
        ) / delta_time

    return result


def _interpolate_short_gaps(
    values: np.ndarray,
    *,
    max_gap: int,
) -> np.ndarray:
    result = values.copy()
    index = 0

    while index < len(result):
        if np.isfinite(result[index]):
            index += 1
            continue

        start = index

        while (
            index < len(result)
            and not np.isfinite(result[index])
        ):
            index += 1

        end = index
        gap_length = end - start

        bounded = (
            start > 0
            and end < len(result)
            and np.isfinite(
                result[start - 1]
            )
            and np.isfinite(result[end])
        )

        if (
            bounded
            and gap_length <= max_gap
        ):
            result[start:end] = np.linspace(
                result[start - 1],
                result[end],
                gap_length + 2,
            )[1:-1]

    return result


def _rolling_statistic(
    values: np.ndarray,
    *,
    window: int,
    statistic: str,
) -> np.ndarray:
    radius = window // 2

    result = np.full(
        len(values),
        np.nan,
        dtype=float,
    )

    for index in range(len(values)):
        start = max(
            0,
            index - radius,
        )
        end = min(
            len(values),
            index + radius + 1,
        )

        sample = values[start:end]

        if not np.isfinite(sample).any():
            continue

        if statistic == "median":
            result[index] = float(
                np.nanmedian(sample)
            )
        else:
            result[index] = float(
                np.nanmean(sample)
            )

    return result


def _as_float_array(
    values: Sequence[float | None],
) -> np.ndarray:
    return np.asarray(
        [
            (
                np.nan
                if value is None
                else float(value)
            )
            for value in values
        ],
        dtype=float,
    )


def _validate_window(
    name: str,
    value: int,
) -> None:
    if value <= 0 or value % 2 == 0:
        raise ValueError(
            f"{name} must be a positive odd integer."
        )


def _optional_float(
    value: float,
) -> float | None:
    if not np.isfinite(value):
        return None

    return float(value)