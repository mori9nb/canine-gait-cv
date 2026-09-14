from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from canine_gait_cv.detection import YOLODetector


class FakeBoxes:
    def __init__(
        self,
        xyxy: list[list[float]],
        confidence: list[float],
        class_ids: list[float],
    ) -> None:
        self.xyxy = np.asarray(xyxy)
        self.conf = np.asarray(confidence)
        self.cls = np.asarray(class_ids)


class FakeResult:
    def __init__(self, boxes: FakeBoxes | None) -> None:
        self.boxes = boxes
        self.names = {15: "cat", 16: "dog"}


class FakeModel:
    names = {15: "cat", 16: "dog"}

    def __init__(self, results: list[FakeResult]) -> None:
        self.results = results
        self.last_predict_kwargs: dict[str, Any] = {}

    def predict(self, **kwargs: Any) -> list[FakeResult]:
        self.last_predict_kwargs = kwargs
        return self.results


def test_yolo_detector_returns_only_dogs_and_clips_boxes() -> None:
    model = FakeModel(
        [
            FakeResult(
                FakeBoxes(
                    xyxy=[[-5.2, 10.4, 110.1, 90.8], [1, 2, 20, 30]],
                    confidence=[0.91, 0.88],
                    class_ids=[16, 15],
                )
            )
        ]
    )
    detector = YOLODetector(model=model, min_confidence=0.4)

    detections = detector.detect(np.zeros((100, 100, 3), dtype=np.uint8))

    assert len(detections) == 1
    assert detections[0].class_name == "dog"
    assert detections[0].confidence == pytest.approx(0.91)
    assert detections[0].box.x_min == 0
    assert detections[0].box.y_min == 10
    assert detections[0].box.x_max == 100
    assert detections[0].box.y_max == 91
    assert model.last_predict_kwargs["conf"] == 0.4
    assert model.last_predict_kwargs["verbose"] is False


def test_yolo_detector_can_return_all_classes() -> None:
    model = FakeModel(
        [FakeResult(FakeBoxes([[1, 2, 20, 30]], [0.88], [15]))]
    )
    detector = YOLODetector(model=model, target_class_names=None)

    detections = detector.detect(np.zeros((100, 100, 3), dtype=np.uint8))

    assert [detection.class_name for detection in detections] == ["cat"]


def test_yolo_detector_passes_optional_inference_settings() -> None:
    model = FakeModel([])
    detector = YOLODetector(
        model=model,
        device="cpu",
        image_size=640,
    )

    detector.detect(np.zeros((100, 100, 3), dtype=np.uint8))

    assert model.last_predict_kwargs["device"] == "cpu"
    assert model.last_predict_kwargs["imgsz"] == 640


def test_yolo_detector_skips_missing_and_zero_area_boxes() -> None:
    model = FakeModel(
        [
            FakeResult(None),
            FakeResult(FakeBoxes([[10, 10, 10, 20]], [0.8], [16])),
        ]
    )
    detector = YOLODetector(model=model)

    detections = detector.detect(np.zeros((100, 100, 3), dtype=np.uint8))

    assert detections == []


@pytest.mark.parametrize("confidence", [-0.1, 1.1])
def test_yolo_detector_rejects_invalid_confidence(confidence: float) -> None:
    with pytest.raises(ValueError, match="min_confidence"):
        YOLODetector(model=FakeModel([]), min_confidence=confidence)


def test_yolo_detector_rejects_empty_frame() -> None:
    detector = YOLODetector(model=FakeModel([]))

    with pytest.raises(ValueError, match="empty frame"):
        detector.detect(np.empty((0, 0, 3), dtype=np.uint8))
