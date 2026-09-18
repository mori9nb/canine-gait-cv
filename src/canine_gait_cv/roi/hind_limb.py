from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import ceil, floor
from statistics import median
from typing import Sequence

from canine_gait_cv.preprocessing import BoundingBox


class MovementDirection(str, Enum):
    """Horizontal movement direction of the gait subject in image space."""

    LEFT = "left"
    RIGHT = "right"


@dataclass(frozen=True)
class DirectionEstimate:
    """Temporal movement-direction estimate and its supporting evidence."""

    direction: MovementDirection | None
    confidence: float
    displacement_pixels: float
    normalized_displacement: float
    valid_sample_count: int

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1.")
        if self.valid_sample_count < 0:
            raise ValueError("valid_sample_count cannot be negative.")


@dataclass(frozen=True)
class HindLimbROI:
    """Candidate hind-limb region derived from a detected dog box.

    The ROI is only a geometric candidate.  It must not be interpreted as a
    validated hind-limb pose or as direct evidence that a joint is visible.
    """

    box: BoundingBox
    movement_direction: MovementDirection
    confidence: float

    def __post_init__(self) -> None:
        if self.box.area == 0:
            raise ValueError("ROI box area must be greater than zero.")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1.")


def estimate_movement_direction(
    boxes: Sequence[BoundingBox | None],
    *,
    min_valid_samples: int = 3,
    min_normalized_displacement: float = 0.05,
    min_confidence: float = 0.50,
) -> DirectionEstimate:
    """Estimate horizontal movement direction from a short box sequence.

    The displacement is measured between median centers in the first and last
    thirds of the valid sequence.  Dividing by median dog-box width makes the
    threshold less dependent on image resolution or dog-camera distance.
    Directional consistency measures how often consecutive center movements
    agree with the overall direction.  Low-motion or inconsistent sequences
    return ``direction=None`` instead of guessing.
    """

    if min_valid_samples < 3:
        raise ValueError("min_valid_samples must be at least 3.")
    if min_normalized_displacement <= 0.0:
        raise ValueError("min_normalized_displacement must be positive.")
    if not 0.0 <= min_confidence <= 1.0:
        raise ValueError("min_confidence must be between 0 and 1.")

    valid_boxes = [box for box in boxes if box is not None and box.area > 0]
    if len(valid_boxes) < min_valid_samples:
        return DirectionEstimate(
            direction=None,
            confidence=0.0,
            displacement_pixels=0.0,
            normalized_displacement=0.0,
            valid_sample_count=len(valid_boxes),
        )

    centers = [(box.x_min + box.x_max) / 2.0 for box in valid_boxes]
    median_width = float(median(box.width for box in valid_boxes))
    window_size = max(1, len(centers) // 3)
    start_center = float(median(centers[:window_size]))
    end_center = float(median(centers[-window_size:]))
    displacement = end_center - start_center
    normalized_displacement = displacement / median_width

    nonzero_steps = [
        current - previous
        for previous, current in zip(centers, centers[1:])
        if current != previous
    ]
    if displacement == 0.0 or not nonzero_steps:
        consistency = 0.0
    else:
        expected_positive = displacement > 0.0
        agreeing_steps = sum(
            (step > 0.0) == expected_positive for step in nonzero_steps
        )
        consistency = agreeing_steps / len(nonzero_steps)

    motion_strength = min(
        1.0,
        abs(normalized_displacement) / (2.0 * min_normalized_displacement),
    )
    confidence = motion_strength * consistency

    direction: MovementDirection | None = None
    if (
        abs(normalized_displacement) >= min_normalized_displacement
        and confidence >= min_confidence
    ):
        direction = (
            MovementDirection.RIGHT
            if displacement > 0.0
            else MovementDirection.LEFT
        )

    return DirectionEstimate(
        direction=direction,
        confidence=confidence,
        displacement_pixels=displacement,
        normalized_displacement=normalized_displacement,
        valid_sample_count=len(valid_boxes),
    )


def build_hind_limb_roi(
    dog_box: BoundingBox,
    *,
    movement_direction: MovementDirection,
    detection_confidence: float,
    image_width: int,
    image_height: int,
    rear_fraction: float = 0.55,
    vertical_start_fraction: float = 0.30,
    padding_fraction: float = 0.05,
) -> HindLimbROI:
    """Build a lower-rear candidate ROI inside and around a detected dog box.

    A dog moving right has its rear on the left side of the detection; a dog
    moving left has its rear on the right.  Direction estimation is kept as a
    separate temporal step so this function stays deterministic and testable.
    """

    if dog_box.area == 0:
        raise ValueError("dog_box area must be greater than zero.")
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image dimensions must be positive.")
    if not 0.0 <= detection_confidence <= 1.0:
        raise ValueError("detection_confidence must be between 0 and 1.")
    if not 0.0 < rear_fraction <= 1.0:
        raise ValueError("rear_fraction must be in (0, 1].")
    if not 0.0 <= vertical_start_fraction < 1.0:
        raise ValueError("vertical_start_fraction must be in [0, 1).")
    if padding_fraction < 0.0:
        raise ValueError("padding_fraction cannot be negative.")

    rear_width = dog_box.width * rear_fraction
    padding_x = dog_box.width * padding_fraction
    padding_y = dog_box.height * padding_fraction

    if movement_direction is MovementDirection.RIGHT:
        x_min = dog_box.x_min - padding_x
        x_max = dog_box.x_min + rear_width + padding_x
    else:
        x_min = dog_box.x_max - rear_width - padding_x
        x_max = dog_box.x_max + padding_x

    y_min = dog_box.y_min + dog_box.height * vertical_start_fraction - padding_y
    y_max = dog_box.y_max + padding_y

    roi_box = BoundingBox(
        x_min=floor(x_min),
        y_min=floor(y_min),
        x_max=ceil(x_max),
        y_max=ceil(y_max),
    ).clip_to_image(image_width=image_width, image_height=image_height)

    return HindLimbROI(
        box=roi_box,
        movement_direction=movement_direction,
        confidence=detection_confidence,
    )
