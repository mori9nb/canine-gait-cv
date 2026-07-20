from __future__ import annotations

import cv2
import numpy as np

from canine_gait_cv.preprocessing import BoundingBox


def draw_bounding_box(
    frame: np.ndarray,
    box: BoundingBox,
    label: str | None = None,
    thickness: int = 2,
) -> np.ndarray:
    """Draw a bounding box on a copy of the input frame."""

    if frame.size == 0:
        raise ValueError("Cannot draw on an empty frame.")

    if thickness <= 0:
        raise ValueError("thickness must be positive.")

    output = frame.copy()

    image_height, image_width = output.shape[:2]
    clipped_box = box.clip_to_image(
        image_width=image_width,
        image_height=image_height,
    )

    if clipped_box.area == 0:
        raise ValueError(f"Invalid bounding box: {box}")

    color = (0, 255, 0)

    cv2.rectangle(
        output,
        (clipped_box.x_min, clipped_box.y_min),
        (clipped_box.x_max, clipped_box.y_max),
        color,
        thickness,
    )

    if label is not None:
        cv2.putText(
            output,
            label,
            (clipped_box.x_min, max(0, clipped_box.y_min - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            thickness,
            cv2.LINE_AA,
        )

    return output