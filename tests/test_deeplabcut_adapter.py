import json

import pytest

from canine_gait_cv.pose import (
    KeypointName,
    SUPERANIMAL_QUADRUPED_BODY_PARTS,
    load_superanimal_json,
    load_superanimal_multi_json,
)
from canine_gait_cv.preprocessing import BoundingBox


def _frame() -> dict:
    points = [
        [float(index), float(index + 10), 0.9]
        for index in range(len(SUPERANIMAL_QUADRUPED_BODY_PARTS))
    ]
    return {
        "bodyparts": [points],
        "bboxes": [[10.2, 20.8, 100.1, 200.2]],
        "bbox_scores": [0.99],
    }


def test_load_superanimal_json_maps_native_hind_limb_names(tmp_path) -> None:
    path = tmp_path / "pose.json"
    path.write_text(json.dumps([_frame()]), encoding="utf-8")

    frames = load_superanimal_json(path)

    assert len(frames) == 1
    frame = frames[0]
    assert frame.frame_index == 0
    assert frame.dog_box == BoundingBox(10, 20, 101, 201)
    assert frame.bbox_confidence == pytest.approx(0.99)
    assert frame.pose.get(KeypointName.NOSE).x == 0.0
    assert frame.pose.get(KeypointName.TAIL_ROOT).x == 22.0
    assert frame.pose.get(KeypointName.LEFT_BACK_PAW).x == 30.0
    assert frame.pose.get(KeypointName.LEFT_HIP).x == 31.0
    assert frame.pose.get(KeypointName.RIGHT_HIP).x == 32.0
    assert frame.pose.get(KeypointName.LEFT_KNEE).x == 33.0
    assert frame.pose.get(KeypointName.RIGHT_KNEE).x == 34.0
    assert frame.pose.get(KeypointName.RIGHT_BACK_PAW).x == 35.0


def test_load_superanimal_json_rejects_wrong_bodypart_count(tmp_path) -> None:
    frame = _frame()
    frame["bodyparts"][0].pop()
    path = tmp_path / "pose.json"
    path.write_text(json.dumps([frame]), encoding="utf-8")

    with pytest.raises(ValueError, match="expected 39 bodyparts"):
        load_superanimal_json(path)


def test_load_superanimal_json_rejects_missing_individual(tmp_path) -> None:
    path = tmp_path / "pose.json"
    path.write_text(json.dumps([_frame()]), encoding="utf-8")

    with pytest.raises(ValueError, match="missing individual 1"):
        load_superanimal_json(path, individual_index=1)


def test_load_superanimal_multi_json_preserves_all_individuals(tmp_path) -> None:
    frame = _frame()
    second_points = [
        [float(index + 100), float(index + 200), 0.8]
        for index in range(len(SUPERANIMAL_QUADRUPED_BODY_PARTS))
    ]
    frame["bodyparts"].append(second_points)
    frame["bboxes"].append([300.0, 50.0, 500.0, 250.0])
    frame["bbox_scores"].append(0.95)
    path = tmp_path / "pose.json"
    path.write_text(json.dumps([frame]), encoding="utf-8")

    frames = load_superanimal_multi_json(path)

    assert len(frames[0].individuals) == 2
    assert frames[0].individuals[0].detection_index == 0
    assert frames[0].individuals[1].detection_index == 1
    assert frames[0].individuals[1].dog_box == BoundingBox(300, 50, 500, 250)
    assert frames[0].individuals[1].pose.get(KeypointName.NOSE).x == 100.0


def test_multi_loader_accepts_a_frame_with_no_detections(tmp_path) -> None:
    path = tmp_path / "pose.json"
    path.write_text(
        json.dumps([{"bodyparts": [], "bboxes": [], "bbox_scores": []}]),
        encoding="utf-8",
    )

    frames = load_superanimal_multi_json(path)

    assert frames[0].individuals == ()


def test_multi_loader_rejects_mismatched_individual_arrays(tmp_path) -> None:
    frame = _frame()
    frame["bbox_scores"] = []
    path = tmp_path / "pose.json"
    path.write_text(json.dumps([frame]), encoding="utf-8")

    with pytest.raises(ValueError, match="same number of individuals"):
        load_superanimal_multi_json(path)
