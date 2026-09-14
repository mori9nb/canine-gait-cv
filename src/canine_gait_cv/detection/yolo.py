from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from math import ceil, floor
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from canine_gait_cv.detection.base import Detector
from canine_gait_cv.detection.types import Detection
from canine_gait_cv.preprocessing import BoundingBox


class YOLOModel(Protocol):
    """Small subset of the Ultralytics model API used by this adapter."""

    names: Mapping[int, str] | Sequence[str]

    def predict(self, **kwargs: Any) -> Sequence[Any]: ...


class YOLODetector(Detector):
    """Run an Ultralytics YOLO model behind the project detector interface.

    Ultralytics is imported lazily so the lightweight video/preprocessing
    modules can still be used without installing the optional YOLO dependency.
    A model can be injected for deterministic unit testing or custom inference.
    """

    def __init__(
        self,
        model_path: str | Path = "yolo11n.pt",
        min_confidence: float = 0.25,
        target_class_names: Collection[str] | None = ("dog",),
        device: str | None = None,
        image_size: int | None = None,
        model: YOLOModel | None = None,
    ) -> None:
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError("min_confidence must be between 0 and 1.")
        if image_size is not None and image_size <= 0:
            raise ValueError("image_size must be positive.")
        if target_class_names is not None and not target_class_names:
            raise ValueError("target_class_names cannot be empty.")

        self.min_confidence = min_confidence
        self.target_class_names = (
            None if target_class_names is None else frozenset(target_class_names)
        )
        self.device = device
        self.image_size = image_size
        self._model = model if model is not None else self._load_model(model_path)

    @staticmethod
    def _load_model(model_path: str | Path) -> YOLOModel:
        try:
            from ultralytics import YOLO
        except ImportError as error:
            raise ImportError(
                "YOLO support requires Ultralytics. Install it with "
                "`python -m pip install -e '.[yolo]'`."
            ) from error

        return YOLO(str(model_path))

    def detect(self, frame: np.ndarray) -> list[Detection]:
        if frame.size == 0:
            raise ValueError("Cannot run detector on an empty frame.")

        predict_kwargs: dict[str, Any] = {
            "source": frame,
            "conf": self.min_confidence,
            "verbose": False,
        }
        if self.device is not None:
            predict_kwargs["device"] = self.device
        if self.image_size is not None:
            predict_kwargs["imgsz"] = self.image_size

        results = self._model.predict(**predict_kwargs)
        image_height, image_width = frame.shape[:2]
        detections: list[Detection] = []

        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue

            coordinates = _to_numpy(boxes.xyxy).reshape(-1, 4)
            confidences = _to_numpy(boxes.conf).reshape(-1)
            class_ids = _to_numpy(boxes.cls).reshape(-1)
            names = getattr(result, "names", self._model.names)

            for coordinates_row, confidence, class_id in zip(
                coordinates,
                confidences,
                class_ids,
                strict=True,
            ):
                class_name = _resolve_class_name(names, int(class_id))
                if (
                    self.target_class_names is not None
                    and class_name not in self.target_class_names
                ):
                    continue

                x_min, y_min, x_max, y_max = coordinates_row.tolist()
                box = BoundingBox(
                    x_min=floor(x_min),
                    y_min=floor(y_min),
                    x_max=ceil(x_max),
                    y_max=ceil(y_max),
                ).clip_to_image(
                    image_width=image_width,
                    image_height=image_height,
                )
                if box.area == 0:
                    continue

                detections.append(
                    Detection(
                        class_name=class_name,
                        confidence=float(confidence),
                        box=box,
                    )
                )

        return detections


def _to_numpy(value: Any) -> np.ndarray:
    """Convert a CPU/GPU tensor-like result into a NumPy array."""

    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _resolve_class_name(
    names: Mapping[int, str] | Sequence[str],
    class_id: int,
) -> str:
    try:
        return names[class_id]
    except (IndexError, KeyError) as error:
        raise ValueError(f"YOLO returned unknown class id: {class_id}") from error
