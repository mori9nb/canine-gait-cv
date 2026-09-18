from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from typing import Mapping

from canine_gait_cv.preprocessing import BoundingBox


class KeypointName(str, Enum):
    """AP-10K-compatible animal keypoint names.

    AP-10K provides knee and back-paw landmarks but no explicit canine hock.
    A later project-specific model can extend this schema with a hock point.
    """

    NOSE = "nose"
    LEFT_EYE = "left_eye"
    RIGHT_EYE = "right_eye"
    NECK = "neck"
    TAIL_ROOT = "tail_root"
    LEFT_SHOULDER = "left_shoulder"
    LEFT_ELBOW = "left_elbow"
    LEFT_FRONT_PAW = "left_front_paw"
    RIGHT_SHOULDER = "right_shoulder"
    RIGHT_ELBOW = "right_elbow"
    RIGHT_FRONT_PAW = "right_front_paw"
    LEFT_HIP = "left_hip"
    LEFT_KNEE = "left_knee"
    LEFT_BACK_PAW = "left_back_paw"
    RIGHT_HIP = "right_hip"
    RIGHT_KNEE = "right_knee"
    RIGHT_BACK_PAW = "right_back_paw"


@dataclass(frozen=True)
class DogKeypoint:
    """One predicted anatomical landmark in image coordinates."""

    x: float
    y: float
    confidence: float

    def __post_init__(self) -> None:
        if not isfinite(self.x) or not isfinite(self.y):
            raise ValueError("keypoint coordinates must be finite.")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1.")


@dataclass(frozen=True)
class DogPose:
    """Named keypoints predicted for one dog in one frame."""

    keypoints: Mapping[KeypointName, DogKeypoint]

    def get(self, name: KeypointName) -> DogKeypoint | None:
        return self.keypoints.get(name)


class BodyOrientation(str, Enum):
    """Direction the dog's head points in image coordinates."""

    LEFT = "left"
    RIGHT = "right"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class BodyOrientationEstimate:
    """Anatomical orientation with confidence and landmark provenance."""

    orientation: BodyOrientation
    confidence: float
    normalized_separation: float
    front_source: str | None
    rear_source: str | None

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1.")


@dataclass(frozen=True)
class _Anchor:
    x: float
    confidence: float
    source: str


def estimate_body_orientation(
    pose: DogPose,
    dog_box: BoundingBox,
    *,
    min_keypoint_confidence: float = 0.30,
    min_normalized_separation: float = 0.15,
    min_orientation_confidence: float = 0.30,
) -> BodyOrientationEstimate:
    """Estimate head direction from anatomical front and rear anchors.

    Preferred evidence is nose plus tail root.  When either is unavailable,
    the mean of two confident eyes or two confident hips is used.  Separation
    is normalized by dog-box width, making the rule resolution-independent.
    Missing, low-confidence, or geometrically ambiguous evidence returns
    ``UNKNOWN`` instead of a guessed orientation.
    """

    if dog_box.area == 0:
        raise ValueError("dog_box area must be greater than zero.")
    if not 0.0 <= min_keypoint_confidence <= 1.0:
        raise ValueError("min_keypoint_confidence must be between 0 and 1.")
    if min_normalized_separation <= 0.0:
        raise ValueError("min_normalized_separation must be positive.")
    if not 0.0 <= min_orientation_confidence <= 1.0:
        raise ValueError("min_orientation_confidence must be between 0 and 1.")

    front = _select_front_anchor(pose, min_keypoint_confidence)
    rear = _select_rear_anchor(pose, min_keypoint_confidence)
    if front is None or rear is None:
        return _unknown_orientation(front=front, rear=rear)

    signed_separation = (front.x - rear.x) / dog_box.width
    normalized_separation = abs(signed_separation)
    separation_strength = min(
        1.0,
        normalized_separation / (2.0 * min_normalized_separation),
    )
    confidence = min(front.confidence, rear.confidence) * separation_strength

    if (
        normalized_separation < min_normalized_separation
        or confidence < min_orientation_confidence
    ):
        orientation = BodyOrientation.UNKNOWN
    elif signed_separation > 0.0:
        orientation = BodyOrientation.RIGHT
    else:
        orientation = BodyOrientation.LEFT

    return BodyOrientationEstimate(
        orientation=orientation,
        confidence=confidence,
        normalized_separation=normalized_separation,
        front_source=front.source,
        rear_source=rear.source,
    )


def _select_front_anchor(
    pose: DogPose,
    min_confidence: float,
) -> _Anchor | None:
    nose = pose.get(KeypointName.NOSE)
    if nose is not None and nose.confidence >= min_confidence:
        return _Anchor(x=nose.x, confidence=nose.confidence, source="nose")

    return _mean_anchor(
        pose,
        KeypointName.LEFT_EYE,
        KeypointName.RIGHT_EYE,
        min_confidence=min_confidence,
        source="eyes",
    )


def _select_rear_anchor(
    pose: DogPose,
    min_confidence: float,
) -> _Anchor | None:
    tail_root = pose.get(KeypointName.TAIL_ROOT)
    if tail_root is not None and tail_root.confidence >= min_confidence:
        return _Anchor(
            x=tail_root.x,
            confidence=tail_root.confidence,
            source="tail_root",
        )

    return _mean_anchor(
        pose,
        KeypointName.LEFT_HIP,
        KeypointName.RIGHT_HIP,
        min_confidence=min_confidence,
        source="hips",
    )


def _mean_anchor(
    pose: DogPose,
    first_name: KeypointName,
    second_name: KeypointName,
    *,
    min_confidence: float,
    source: str,
) -> _Anchor | None:
    first = pose.get(first_name)
    second = pose.get(second_name)
    if (
        first is None
        or second is None
        or first.confidence < min_confidence
        or second.confidence < min_confidence
    ):
        return None

    return _Anchor(
        x=(first.x + second.x) / 2.0,
        confidence=min(first.confidence, second.confidence),
        source=source,
    )


def _unknown_orientation(
    *,
    front: _Anchor | None,
    rear: _Anchor | None,
) -> BodyOrientationEstimate:
    return BodyOrientationEstimate(
        orientation=BodyOrientation.UNKNOWN,
        confidence=0.0,
        normalized_separation=0.0,
        front_source=None if front is None else front.source,
        rear_source=None if rear is None else rear.source,
    )
