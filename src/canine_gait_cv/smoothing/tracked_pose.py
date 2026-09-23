from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from canine_gait_cv.pose import KeypointName
from canine_gait_cv.smoothing.kalman_rts import (
    SmoothedKeypoint,
    smooth_keypoint_trajectory,
)
from canine_gait_cv.tracking import TrackedIndividual, TrackedPoseFrame


@dataclass(frozen=True)
class SmoothedTrackedIndividual:
    """Raw tracked detection plus its temporally smoothed keypoints."""

    tracked: TrackedIndividual
    keypoints: Mapping[KeypointName, SmoothedKeypoint]

    @property
    def track_id(self) -> int:
        return self.tracked.track_id


@dataclass(frozen=True)
class SmoothedTrackedPoseFrame:
    """Smoothed pose estimates for every observed dog in one frame."""

    frame_index: int
    individuals: tuple[SmoothedTrackedIndividual, ...]


def smooth_tracked_pose_frames(
    frames: list[TrackedPoseFrame],
    *,
    min_confidence: float = 0.20,
    process_variance: float = 1.0,
    measurement_variance: float = 16.0,
    outlier_mahalanobis_squared: float = 9.21,
) -> list[SmoothedTrackedPoseFrame]:
    """Smooth every keypoint independently within each persistent track ID.

    Missing global frames are inserted internally so motion timing remains
    correct. Output frames retain only actually observed dogs; raw detections
    remain available through ``SmoothedTrackedIndividual.tracked``.
    """

    if not frames:
        return []
    frame_indices = [frame.frame_index for frame in frames]
    if any(current <= previous for previous, current in zip(frame_indices, frame_indices[1:])):
        raise ValueError("frame indices must be strictly increasing.")

    observations: dict[int, dict[int, TrackedIndividual]] = {}
    for frame in frames:
        ids_in_frame: set[int] = set()
        for individual in frame.individuals:
            if individual.track_id in ids_in_frame:
                raise ValueError(
                    f"Frame {frame.frame_index}: duplicate track_id "
                    f"{individual.track_id}."
                )
            ids_in_frame.add(individual.track_id)
            observations.setdefault(individual.track_id, {})[
                frame.frame_index
            ] = individual

    smoothed_by_frame_and_track: dict[
        tuple[int, int], dict[KeypointName, SmoothedKeypoint]
    ] = {}
    for track_id, track_observations in observations.items():
        start = min(track_observations)
        end = max(track_observations)
        timeline = range(start, end + 1)
        for name in KeypointName:
            points = [
                None
                if (individual := track_observations.get(frame_index)) is None
                else individual.detection.pose.get(name)
                for frame_index in timeline
            ]
            smoothed = smooth_keypoint_trajectory(
                points,
                min_confidence=min_confidence,
                process_variance=process_variance,
                measurement_variance=measurement_variance,
                outlier_mahalanobis_squared=outlier_mahalanobis_squared,
            )
            for offset, estimate in enumerate(smoothed):
                frame_index = start + offset
                if estimate is None or frame_index not in track_observations:
                    continue
                smoothed_by_frame_and_track.setdefault(
                    (frame_index, track_id), {}
                )[name] = estimate

    result: list[SmoothedTrackedPoseFrame] = []
    for frame in frames:
        result.append(
            SmoothedTrackedPoseFrame(
                frame_index=frame.frame_index,
                individuals=tuple(
                    SmoothedTrackedIndividual(
                        tracked=individual,
                        keypoints=smoothed_by_frame_and_track.get(
                            (frame.frame_index, individual.track_id), {}
                        ),
                    )
                    for individual in frame.individuals
                ),
            )
        )
    return result
