from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from canine_gait_cv.detection.base import Detector
from canine_gait_cv.detection.types import Detection
from canine_gait_cv.preprocessing import BoundingBox


@dataclass(frozen=True)
class FixedBoxDetector(Detector):
    """Detector that always returns one fixed bounding box.

    This is useful for testing the pipeline before adding YOLO.
    """

    box: BoundingBox
    class_name: str = "dog"
    confidence: float = 1.0

    def detect(self, frame: np.ndarray) -> list[Detection]:
        if frame.size == 0:
            raise ValueError("Cannot run detector on an empty frame.")

        image_height, image_width = frame.shape[:2]

        clipped_box = self.box.clip_to_image(
            image_width=image_width,
            image_height=image_height,
        )

        if clipped_box.area == 0:
            return []

        return [
            Detection(
                class_name=self.class_name,
                confidence=self.confidence,
                box=clipped_box,
            )
        ]