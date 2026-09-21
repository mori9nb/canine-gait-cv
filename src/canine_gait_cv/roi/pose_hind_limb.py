from __future__ import annotations

from dataclasses import dataclass
from math import ceil, floor

from canine_gait_cv.pose import BodyOrientation, DogPose, KeypointName
from canine_gait_cv.preprocessing import BoundingBox


HIND_LIMB_KEYPOINTS = (
    KeypointName.LEFT_HIP,
    KeypointName.LEFT_KNEE,
    KeypointName.LEFT_BACK_PAW,
    KeypointName.RIGHT_HIP,
    KeypointName.RIGHT_KNEE,
    KeypointName.RIGHT_BACK_PAW,
)


@dataclass(frozen=True)
class PoseHindLimbROI:
    """Hind-limb ROI supported by pose landmarks or anatomical fallback."""

    box: BoundingBox
    confidence: float
    keypoint_count: int
    used_fallback: bool

    def __post_init__(self) -> None:
        if self.box.area == 0:
            raise ValueError("ROI box area must be greater than zero.")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1.")
        if self.keypoint_count < 0:
            raise ValueError("keypoint_count cannot be negative.")


def build_pose_hind_limb_roi(
    pose: DogPose,
    dog_box: BoundingBox,
    *,
    orientation: BodyOrientation,
    image_width: int,
    image_height: int,
    min_keypoint_confidence: float = 0.30,
    min_keypoints: int = 3,
    padding_x_fraction: float = 0.08,
    padding_y_fraction: float = 0.10,
    fallback_rear_fraction: float = 0.55,
    fallback_vertical_start_fraction: float = 0.25,
) -> PoseHindLimbROI | None:
    """Build a hind-limb ROI from confident bilateral hind-limb landmarks.

    When too few pose landmarks survive the threshold, an anatomical rear-side
    crop is used only if head orientation is known. Unknown orientation returns
    ``None`` rather than guessing from camera-relative box displacement.
    """

    if dog_box.area == 0:
        raise ValueError("dog_box area must be greater than zero.")
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image dimensions must be positive.")
    if not 0.0 <= min_keypoint_confidence <= 1.0:
        raise ValueError("min_keypoint_confidence must be between 0 and 1.")
    if min_keypoints < 2:
        raise ValueError("min_keypoints must be at least 2.")
    if padding_x_fraction < 0.0 or padding_y_fraction < 0.0:
        raise ValueError("padding fractions cannot be negative.")

    points = [
        point
        for name in HIND_LIMB_KEYPOINTS
        if (point := pose.get(name)) is not None
        and point.confidence >= min_keypoint_confidence
    ]
    padding_x = dog_box.width * padding_x_fraction
    padding_y = dog_box.height * padding_y_fraction

    if len(points) >= min_keypoints:
        raw_box = BoundingBox(
            x_min=floor(min(point.x for point in points) - padding_x),
            y_min=floor(min(point.y for point in points) - padding_y),
            x_max=ceil(max(point.x for point in points) + padding_x) + 1,
            y_max=ceil(max(point.y for point in points) + padding_y) + 1,
        )
        clipped = raw_box.clip_to_image(image_width, image_height)
        if clipped.area == 0:
            return None
        return PoseHindLimbROI(
            box=clipped,
            confidence=sum(point.confidence for point in points) / len(points),
            keypoint_count=len(points),
            used_fallback=False,
        )

    if orientation is BodyOrientation.UNKNOWN:
        return None

    rear_width = dog_box.width * fallback_rear_fraction
    if orientation is BodyOrientation.RIGHT:
        x_min = dog_box.x_min - padding_x
        x_max = dog_box.x_min + rear_width + padding_x
    else:
        x_min = dog_box.x_max - rear_width - padding_x
        x_max = dog_box.x_max + padding_x

    raw_box = BoundingBox(
        x_min=floor(x_min),
        y_min=floor(
            dog_box.y_min + dog_box.height * fallback_vertical_start_fraction
            - padding_y
        ),
        x_max=ceil(x_max),
        y_max=ceil(dog_box.y_max + padding_y),
    )
    clipped = raw_box.clip_to_image(image_width, image_height)
    if clipped.area == 0:
        return None
    return PoseHindLimbROI(
        box=clipped,
        confidence=0.0,
        keypoint_count=len(points),
        used_fallback=True,
    )
