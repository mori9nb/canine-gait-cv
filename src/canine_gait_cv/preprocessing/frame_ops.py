from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class BoundingBox:
    """Rectangle region in image coordinates."""

    x_min: int
    y_min: int
    x_max: int
    y_max: int

    @property
    def width(self) -> int:
        return max(0, self.x_max - self.x_min)

    @property
    def height(self) -> int:
        return max(0, self.y_max - self.y_min)

    @property
    def area(self) -> int:
        return self.width * self.height

    def clip_to_image(self, image_width: int, image_height: int) -> BoundingBox:
        """Clip bounding box coordinates so they stay inside the image."""

        return BoundingBox(
            x_min=max(0, min(self.x_min, image_width)),
            y_min=max(0, min(self.y_min, image_height)),
            x_max=max(0, min(self.x_max, image_width)),
            y_max=max(0, min(self.y_max, image_height)),
        )


def resize_frame(
    frame: np.ndarray,
    target_width: int | None = None,
    target_height: int | None = None,
) -> np.ndarray:
    """Resize a frame while preserving aspect ratio if only one side is given."""

    if frame.size == 0:
        raise ValueError("Cannot resize an empty frame.")

    if target_width is None and target_height is None:
        raise ValueError("Either target_width or target_height must be provided.")

    original_height, original_width = frame.shape[:2]

    if target_width is not None and target_width <= 0:
        raise ValueError("target_width must be positive.")

    if target_height is not None and target_height <= 0:
        raise ValueError("target_height must be positive.")

    if target_width is not None and target_height is not None:
        new_width = target_width
        new_height = target_height

    elif target_width is not None:
        scale = target_width / original_width
        new_width = target_width
        new_height = int(round(original_height * scale))

    else:
        scale = target_height / original_height
        new_width = int(round(original_width * scale))
        new_height = target_height

    return cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)


def crop_frame(frame: np.ndarray, box: BoundingBox) -> np.ndarray:
    """Crop a frame using a bounding box."""

    if frame.size == 0:
        raise ValueError("Cannot crop an empty frame.")

    image_height, image_width = frame.shape[:2]
    clipped_box = box.clip_to_image(image_width=image_width, image_height=image_height)

    if clipped_box.area == 0:
        raise ValueError(f"Invalid crop box: {box}")

    return frame[
        clipped_box.y_min : clipped_box.y_max,
        clipped_box.x_min : clipped_box.x_max,
    ].copy()