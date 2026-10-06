from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from canine_gait_cv.gait.stride import (
    StrideInterval,
)


@dataclass(frozen=True)
class NormalizedStrideProfile:
    stride_index: int
    cycle_percent: tuple[float, ...]
    knee_angle_degrees: tuple[float, ...]
    hock_angle_degrees: tuple[float, ...]
    toe_off_percent: float
    stride_duration_seconds: float
    source_coverage: float


@dataclass(frozen=True)
class MeanReferenceTrajectory:
    cycle_percent: tuple[float, ...]
    knee_mean_degrees: tuple[float, ...]
    knee_std_degrees: tuple[float, ...]
    hock_mean_degrees: tuple[float, ...]
    hock_std_degrees: tuple[float, ...]
    knee_velocity_degrees_per_second: tuple[
        float, ...
    ]
    hock_velocity_degrees_per_second: tuple[
        float, ...
    ]
    mean_toe_off_percent: float
    mean_stride_duration_seconds: float
    stride_count: int


def normalize_stride_profiles(
    *,
    frame_indices: Sequence[int],
    knee_angles: Sequence[float | None],
    hock_angles: Sequence[float | None],
    strides: Sequence[StrideInterval],
    sample_count: int = 101,
    min_source_coverage: float = 0.80,
) -> tuple[NormalizedStrideProfile, ...]:
    """Normalize complete strides to a common 0-100% gait cycle."""

    if sample_count < 2:
        raise ValueError(
            "sample_count must be at least 2."
        )

    if not 0.0 <= min_source_coverage <= 1.0:
        raise ValueError(
            "min_source_coverage must be between 0 and 1."
        )

    lengths = {
        len(frame_indices),
        len(knee_angles),
        len(hock_angles),
    }

    if len(lengths) != 1:
        raise ValueError(
            "Frame and angle arrays must have the same length."
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

    frames = np.asarray(
        frame_indices,
        dtype=int,
    )

    knee = _as_float_array(
        knee_angles
    )
    hock = _as_float_array(
        hock_angles
    )

    target_percent = np.linspace(
        0.0,
        100.0,
        sample_count,
    )

    profiles: list[
        NormalizedStrideProfile
    ] = []

    for stride in strides:
        mask = (
            (frames >= stride.contact_frame)
            & (
                frames
                <= stride.next_contact_frame
            )
        )

        stride_frames = frames[mask]
        stride_knee = knee[mask]
        stride_hock = hock[mask]

        if len(stride_frames) < 2:
            continue

        both_valid = (
            np.isfinite(stride_knee)
            & np.isfinite(stride_hock)
        )

        coverage = float(
            np.mean(both_valid)
        )

        if coverage < min_source_coverage:
            continue

        if (
            stride_frames[0]
            != stride.contact_frame
            or stride_frames[-1]
            != stride.next_contact_frame
        ):
            continue

        denominator = (
            stride.next_contact_frame
            - stride.contact_frame
        )

        if denominator <= 0:
            continue

        source_percent = (
            (
                stride_frames
                - stride.contact_frame
            )
            / denominator
            * 100.0
        )

        knee_normalized = _interpolate_signal(
            source_percent,
            stride_knee,
            target_percent,
        )

        hock_normalized = _interpolate_signal(
            source_percent,
            stride_hock,
            target_percent,
        )

        if (
            knee_normalized is None
            or hock_normalized is None
        ):
            continue

        toe_off_percent = (
            (
                stride.toe_off_frame
                - stride.contact_frame
            )
            / denominator
            * 100.0
        )

        profiles.append(
            NormalizedStrideProfile(
                stride_index=(
                    stride.stride_index
                ),
                cycle_percent=tuple(
                    float(value)
                    for value in target_percent
                ),
                knee_angle_degrees=tuple(
                    float(value)
                    for value
                    in knee_normalized
                ),
                hock_angle_degrees=tuple(
                    float(value)
                    for value
                    in hock_normalized
                ),
                toe_off_percent=float(
                    toe_off_percent
                ),
                stride_duration_seconds=(
                    stride.stride_duration_seconds
                ),
                source_coverage=coverage,
            )
        )

    return tuple(profiles)


def build_mean_reference_trajectory(
    profiles: Sequence[
        NormalizedStrideProfile
    ],
) -> MeanReferenceTrajectory:
    """Build the reference trajectory used by the prosthesis model."""

    if not profiles:
        raise ValueError(
            "At least one normalized stride is required."
        )

    sample_counts = {
        len(profile.cycle_percent)
        for profile in profiles
    }

    if len(sample_counts) != 1:
        raise ValueError(
            "All profiles must have the same sample count."
        )

    cycle_percent = np.asarray(
        profiles[0].cycle_percent,
        dtype=float,
    )

    knee = np.asarray(
        [
            profile.knee_angle_degrees
            for profile in profiles
        ],
        dtype=float,
    )

    hock = np.asarray(
        [
            profile.hock_angle_degrees
            for profile in profiles
        ],
        dtype=float,
    )

    durations = np.asarray(
        [
            profile.stride_duration_seconds
            for profile in profiles
        ],
        dtype=float,
    )

    toe_offs = np.asarray(
        [
            profile.toe_off_percent
            for profile in profiles
        ],
        dtype=float,
    )

    knee_mean = np.mean(
        knee,
        axis=0,
    )
    hock_mean = np.mean(
        hock,
        axis=0,
    )

    if len(profiles) > 1:
        knee_std = np.std(
            knee,
            axis=0,
            ddof=1,
        )
        hock_std = np.std(
            hock,
            axis=0,
            ddof=1,
        )
    else:
        knee_std = np.zeros_like(
            knee_mean
        )
        hock_std = np.zeros_like(
            hock_mean
        )

    mean_duration = float(
        np.mean(durations)
    )

    time_seconds = (
        cycle_percent
        / 100.0
        * mean_duration
    )

    knee_velocity = np.gradient(
        knee_mean,
        time_seconds,
    )

    hock_velocity = np.gradient(
        hock_mean,
        time_seconds,
    )

    return MeanReferenceTrajectory(
        cycle_percent=tuple(
            float(value)
            for value in cycle_percent
        ),
        knee_mean_degrees=tuple(
            float(value)
            for value in knee_mean
        ),
        knee_std_degrees=tuple(
            float(value)
            for value in knee_std
        ),
        hock_mean_degrees=tuple(
            float(value)
            for value in hock_mean
        ),
        hock_std_degrees=tuple(
            float(value)
            for value in hock_std
        ),
        knee_velocity_degrees_per_second=tuple(
            float(value)
            for value in knee_velocity
        ),
        hock_velocity_degrees_per_second=tuple(
            float(value)
            for value in hock_velocity
        ),
        mean_toe_off_percent=float(
            np.mean(toe_offs)
        ),
        mean_stride_duration_seconds=(
            mean_duration
        ),
        stride_count=len(profiles),
    )


def _interpolate_signal(
    source_percent: np.ndarray,
    values: np.ndarray,
    target_percent: np.ndarray,
) -> np.ndarray | None:
    valid = (
        np.isfinite(source_percent)
        & np.isfinite(values)
    )

    if np.count_nonzero(valid) < 2:
        return None

    valid_percent = source_percent[
        valid
    ]
    valid_values = values[
        valid
    ]

    # Do not extrapolate missing contact boundaries.
    if (
        valid_percent[0] > 0.0
        or valid_percent[-1] < 100.0
    ):
        return None

    return np.interp(
        target_percent,
        valid_percent,
        valid_values,
    )


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
