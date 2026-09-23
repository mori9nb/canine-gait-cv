from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from canine_gait_cv.pose import DogKeypoint


@dataclass(frozen=True)
class SmoothedKeypoint:
    """One confidence-aware trajectory estimate for a video frame."""

    x: float
    y: float
    velocity_x: float
    velocity_y: float
    source_confidence: float
    measurement_used: bool


def smooth_keypoint_trajectory(
    points: list[DogKeypoint | None],
    *,
    dt: float = 1.0,
    min_confidence: float = 0.20,
    process_variance: float = 1.0,
    measurement_variance: float = 16.0,
    outlier_mahalanobis_squared: float = 9.21,
) -> list[SmoothedKeypoint | None]:
    """Apply a confidence-aware Kalman filter and backward RTS smoother.

    Low-confidence or missing measurements are predicted through. Measurements
    that fail the innovation gate are treated as outliers. Frames before the
    first usable measurement remain ``None`` because no track state exists yet.
    """

    if dt <= 0.0:
        raise ValueError("dt must be positive.")
    if not 0.0 <= min_confidence <= 1.0:
        raise ValueError("min_confidence must be between 0 and 1.")
    if process_variance <= 0.0 or measurement_variance <= 0.0:
        raise ValueError("variance values must be positive.")
    if outlier_mahalanobis_squared <= 0.0:
        raise ValueError("outlier gate must be positive.")
    if not points:
        return []

    first_index = next(
        (
            index
            for index, point in enumerate(points)
            if point is not None and point.confidence >= min_confidence
        ),
        None,
    )
    if first_index is None:
        return [None] * len(points)

    transition = np.array(
        [
            [1.0, 0.0, dt, 0.0],
            [0.0, 1.0, 0.0, dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    observation = np.array(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
        dtype=float,
    )
    dt2 = dt * dt
    dt3 = dt2 * dt
    dt4 = dt2 * dt2
    process_noise = process_variance * np.array(
        [
            [dt4 / 4.0, 0.0, dt3 / 2.0, 0.0],
            [0.0, dt4 / 4.0, 0.0, dt3 / 2.0],
            [dt3 / 2.0, 0.0, dt2, 0.0],
            [0.0, dt3 / 2.0, 0.0, dt2],
        ],
        dtype=float,
    )
    identity = np.eye(4)

    first = points[first_index]
    assert first is not None
    state = np.array([first.x, first.y, 0.0, 0.0], dtype=float)
    covariance = np.diag([measurement_variance, measurement_variance, 100.0, 100.0])

    filtered_states: list[np.ndarray] = []
    filtered_covariances: list[np.ndarray] = []
    predicted_states: list[np.ndarray] = []
    predicted_covariances: list[np.ndarray] = []
    measurement_used: list[bool] = []

    for index in range(first_index, len(points)):
        if index == first_index:
            predicted_state = state.copy()
            predicted_covariance = covariance.copy()
        else:
            predicted_state = transition @ state
            predicted_covariance = transition @ covariance @ transition.T + process_noise

        point = points[index]
        used = False
        if point is not None and point.confidence >= min_confidence:
            confidence = max(point.confidence, 0.05)
            noise = np.eye(2) * (measurement_variance / (confidence * confidence))
            innovation = np.array([point.x, point.y]) - observation @ predicted_state
            innovation_covariance = (
                observation @ predicted_covariance @ observation.T + noise
            )
            mahalanobis_squared = float(
                innovation.T @ np.linalg.solve(innovation_covariance, innovation)
            )
            if index == first_index or mahalanobis_squared <= outlier_mahalanobis_squared:
                gain = (
                    predicted_covariance
                    @ observation.T
                    @ np.linalg.inv(innovation_covariance)
                )
                state = predicted_state + gain @ innovation
                covariance = (
                    (identity - gain @ observation)
                    @ predicted_covariance
                    @ (identity - gain @ observation).T
                    + gain @ noise @ gain.T
                )
                used = True
            else:
                state = predicted_state
                covariance = predicted_covariance
        else:
            state = predicted_state
            covariance = predicted_covariance

        predicted_states.append(predicted_state)
        predicted_covariances.append(predicted_covariance)
        filtered_states.append(state.copy())
        filtered_covariances.append(covariance.copy())
        measurement_used.append(used)

    smoothed_states = [state.copy() for state in filtered_states]
    smoothed_covariances = [cov.copy() for cov in filtered_covariances]
    for index in range(len(filtered_states) - 2, -1, -1):
        smoother_gain = (
            filtered_covariances[index]
            @ transition.T
            @ np.linalg.inv(predicted_covariances[index + 1])
        )
        smoothed_states[index] = filtered_states[index] + smoother_gain @ (
            smoothed_states[index + 1] - predicted_states[index + 1]
        )
        smoothed_covariances[index] = filtered_covariances[index] + smoother_gain @ (
            smoothed_covariances[index + 1] - predicted_covariances[index + 1]
        ) @ smoother_gain.T

    result: list[SmoothedKeypoint | None] = [None] * first_index
    for offset, state in enumerate(smoothed_states):
        source = points[first_index + offset]
        result.append(
            SmoothedKeypoint(
                x=float(state[0]),
                y=float(state[1]),
                velocity_x=float(state[2]),
                velocity_y=float(state[3]),
                source_confidence=0.0 if source is None else source.confidence,
                measurement_used=measurement_used[offset],
            )
        )
    return result
