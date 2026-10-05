from __future__ import annotations

from dataclasses import dataclass
from math import acos, degrees, hypot

from canine_gait_cv.pose import (
    DeepLabCutIndividual,
    DeepLabCutMultiPoseFrame,
    DogKeypoint,
    KeypointName,
)
from canine_gait_cv.smoothing.kalman_rts import (
    smooth_keypoint_trajectory,
)


@dataclass(frozen=True)
class HindLimbAngleFrame:
    """Hind-limb angle estimate for one video frame."""

    frame_index: int
    time_seconds: float
    angle_degrees: float | None
    fully_observed: bool
    minimum_source_confidence: float


def select_primary_individual(
    frame: DeepLabCutMultiPoseFrame,
    *,
    min_keypoint_confidence: float = 0.30,
) -> DeepLabCutIndividual | None:
    """Select the most reliable full-dog detection in a frame."""

    if not frame.individuals:
        return None

    def ranking(
        individual: DeepLabCutIndividual,
    ) -> tuple[float, int, int]:
        reliable_keypoints = sum(
            point.confidence >= min_keypoint_confidence
            for point in individual.pose.keypoints.values()
        )

        return (
            individual.bbox_confidence,
            reliable_keypoints,
            individual.dog_box.area,
        )

    return max(frame.individuals, key=ranking)


def joint_angle_degrees(
    proximal: DogKeypoint,
    joint: DogKeypoint,
    distal: DogKeypoint,
) -> float | None:
    """Calculate the internal 2D angle at the middle joint.

    For the hind limb, the points are hip, knee and back paw.
    """

    first_x = proximal.x - joint.x
    first_y = proximal.y - joint.y
    second_x = distal.x - joint.x
    second_y = distal.y - joint.y

    first_length = hypot(first_x, first_y)
    second_length = hypot(second_x, second_y)

    if first_length == 0.0 or second_length == 0.0:
        return None

    cosine = (
        first_x * second_x + first_y * second_y
    ) / (first_length * second_length)

    # Avoid numerical errors slightly outside the acos domain.
    cosine = max(-1.0, min(1.0, cosine))

    return degrees(acos(cosine))


def build_hind_limb_angle_series(
    frames: list[DeepLabCutMultiPoseFrame],
    *,
    fps: float,
    side: str = "right",
    min_confidence: float = 0.30,
    process_variance: float = 4.0,
    measurement_variance: float = 9.0,
    outlier_mahalanobis_squared: float = 25.0,
) -> list[HindLimbAngleFrame]:
    """Build a confidence-aware smoothed hind-limb angle trajectory.

    The raw hip-knee-paw angle is calculated first. The resulting scalar
    angle trajectory is then smoothed. This preserves limb geometry better
    than independently smoothing the three keypoints before calculating
    their angle.

    ``fully_observed`` means all three source keypoints passed the confidence
    threshold in the original DeepLabCut output. Missing or unreliable angle
    measurements may still receive a temporal estimate from the smoother.
    """

    if fps <= 0.0:
        raise ValueError("fps must be positive.")

    if side not in {"left", "right"}:
        raise ValueError("side must be 'left' or 'right'.")

    if not 0.0 <= min_confidence <= 1.0:
        raise ValueError(
            "min_confidence must be between 0 and 1."
        )

    if process_variance <= 0.0:
        raise ValueError(
            "process_variance must be positive."
        )

    if measurement_variance <= 0.0:
        raise ValueError(
            "measurement_variance must be positive."
        )

    if outlier_mahalanobis_squared <= 0.0:
        raise ValueError(
            "outlier_mahalanobis_squared must be positive."
        )

    if side == "right":
        keypoint_names = (
            KeypointName.RIGHT_HIP,
            KeypointName.RIGHT_KNEE,
            KeypointName.RIGHT_BACK_PAW,
        )
    else:
        keypoint_names = (
            KeypointName.LEFT_HIP,
            KeypointName.LEFT_KNEE,
            KeypointName.LEFT_BACK_PAW,
        )

    angle_measurements: list[DogKeypoint | None] = []
    fully_observed_flags: list[bool] = []
    source_confidences: list[float] = []

    for frame in frames:
        individual = select_primary_individual(
            frame,
            min_keypoint_confidence=min_confidence,
        )

        if individual is None:
            angle_measurements.append(None)
            fully_observed_flags.append(False)
            source_confidences.append(0.0)
            continue

        proximal = individual.pose.get(keypoint_names[0])
        joint = individual.pose.get(keypoint_names[1])
        distal = individual.pose.get(keypoint_names[2])

        minimum_confidence = min(
            proximal.confidence,
            joint.confidence,
            distal.confidence,
        )

        source_confidences.append(minimum_confidence)

        fully_observed = minimum_confidence >= min_confidence
        fully_observed_flags.append(fully_observed)

        if not fully_observed:
            angle_measurements.append(None)
            continue

        raw_angle = joint_angle_degrees(
            proximal,
            joint,
            distal,
        )

        if raw_angle is None:
            angle_measurements.append(None)
            fully_observed_flags[-1] = False
            continue

        # The existing Kalman implementation expects a 2D point.
        # Store the scalar angle in x and keep y fixed at zero.
        angle_measurements.append(
            DogKeypoint(
                x=raw_angle,
                y=0.0,
                confidence=minimum_confidence,
            )
        )

    smoothed_angles = smooth_keypoint_trajectory(
        angle_measurements,
        min_confidence=min_confidence,
        process_variance=process_variance,
        measurement_variance=measurement_variance,
        outlier_mahalanobis_squared=outlier_mahalanobis_squared,
    )

    result: list[HindLimbAngleFrame] = []

    for frame, estimate, fully_observed, confidence in zip(
        frames,
        smoothed_angles,
        fully_observed_flags,
        source_confidences,
    ):
        if estimate is None:
            angle_degrees = None
        else:
            # The internal projected joint angle is bounded to [0, 180].
            angle_degrees = max(
                0.0,
                min(180.0, estimate.x),
            )

        result.append(
            HindLimbAngleFrame(
                frame_index=frame.frame_index,
                time_seconds=frame.frame_index / fps,
                angle_degrees=angle_degrees,
                fully_observed=fully_observed,
                minimum_source_confidence=confidence,
            )
        )

    return result