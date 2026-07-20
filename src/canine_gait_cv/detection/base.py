from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from canine_gait_cv.detection.types import Detection


class Detector(ABC):
    """Base interface for all object detectors."""

    @abstractmethod
    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Detect objects in a frame."""
        raise NotImplementedError