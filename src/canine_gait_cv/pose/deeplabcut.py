from __future__ import annotations

import json
from dataclasses import dataclass
from math import ceil, floor
from pathlib import Path
from typing import Any

from canine_gait_cv.pose.types import DogKeypoint, DogPose, KeypointName
from canine_gait_cv.preprocessing import BoundingBox


# Order used by DeepLabCut 3.0.1 SuperAnimal-Quadruped JSON output.
SUPERANIMAL_QUADRUPED_BODY_PARTS = (
    "nose", "upper_jaw", "lower_jaw", "mouth_end_right", "mouth_end_left",
    "right_eye", "right_earbase", "right_earend", "right_antler_base",
    "right_antler_end", "left_eye", "left_earbase", "left_earend",
    "left_antler_base", "left_antler_end", "neck_base", "neck_end",
    "throat_base", "throat_end", "back_base", "back_end", "back_middle",
    "tail_base", "tail_end", "front_left_thai", "front_left_knee",
    "front_left_paw", "front_right_thai", "front_right_knee",
    "front_right_paw", "back_left_paw", "back_left_thai",
    "back_right_thai", "back_left_knee", "back_right_knee",
    "back_right_paw", "belly_bottom", "body_middle_right", "body_middle_left",
)


# SuperAnimal uses the label "thai" (not "thigh").  It is treated as the
# proximal limb/hip anchor in our coarse project schema, not as an exact hip
# joint annotation.
SUPERANIMAL_TO_PROJECT = {
    "nose": KeypointName.NOSE,
    "left_eye": KeypointName.LEFT_EYE,
    "right_eye": KeypointName.RIGHT_EYE,
    "neck_base": KeypointName.NECK,
    "tail_base": KeypointName.TAIL_ROOT,
    "back_left_thai": KeypointName.LEFT_HIP,
    "back_left_knee": KeypointName.LEFT_KNEE,
    "back_left_paw": KeypointName.LEFT_BACK_PAW,
    "back_right_thai": KeypointName.RIGHT_HIP,
    "back_right_knee": KeypointName.RIGHT_KNEE,
    "back_right_paw": KeypointName.RIGHT_BACK_PAW,
}


@dataclass(frozen=True)
class DeepLabCutPoseFrame:
    frame_index: int
    dog_box: BoundingBox
    bbox_confidence: float
    pose: DogPose


@dataclass(frozen=True)
class DeepLabCutIndividual:
    """One detected individual in a frame before temporal identity tracking."""

    detection_index: int
    dog_box: BoundingBox
    bbox_confidence: float
    pose: DogPose


@dataclass(frozen=True)
class DeepLabCutMultiPoseFrame:
    """All individuals detected in one video frame."""

    frame_index: int
    individuals: tuple[DeepLabCutIndividual, ...]


def load_superanimal_json(
    path: str | Path,
    *,
    individual_index: int = 0,
) -> list[DeepLabCutPoseFrame]:
    """Load DLC SuperAnimal JSON and map one individual to project types."""

    if individual_index < 0:
        raise ValueError("individual_index cannot be negative.")

    multi_frames = load_superanimal_multi_json(path)
    frames: list[DeepLabCutPoseFrame] = []
    for frame in multi_frames:
        try:
            individual = frame.individuals[individual_index]
        except IndexError as exc:
            raise ValueError(
                f"Frame {frame.frame_index}: missing individual "
                f"{individual_index} data."
            ) from exc
        frames.append(
            DeepLabCutPoseFrame(
                frame_index=frame.frame_index,
                dog_box=individual.dog_box,
                bbox_confidence=individual.bbox_confidence,
                pose=individual.pose,
            )
        )
    return frames


def load_superanimal_multi_json(
    path: str | Path,
) -> list[DeepLabCutMultiPoseFrame]:
    """Load every detected individual; no cross-frame identity is inferred."""

    with Path(path).open(encoding="utf-8") as stream:
        raw = json.load(stream)
    if not isinstance(raw, list):
        raise ValueError("Expected a JSON list with one item per video frame.")

    frames: list[DeepLabCutMultiPoseFrame] = []
    for frame_index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"Frame {frame_index}: expected an object.")
        try:
            bodyparts = item["bodyparts"]
            bboxes = item["bboxes"]
            bbox_scores = item["bbox_scores"]
        except KeyError as exc:
            raise ValueError(f"Frame {frame_index}: missing {exc.args[0]}.") from exc
        if not (len(bodyparts) == len(bboxes) == len(bbox_scores)):
            raise ValueError(
                f"Frame {frame_index}: bodyparts, bboxes and bbox_scores "
                "must contain the same number of individuals."
            )

        individuals = tuple(
            _parse_individual(
                item,
                frame_index=frame_index,
                individual_index=index,
            )
            for index in range(len(bodyparts))
        )
        frames.append(
            DeepLabCutMultiPoseFrame(
                frame_index=frame_index,
                individuals=individuals,
            )
        )
    return frames


def _parse_frame(
    item: Any,
    *,
    frame_index: int,
    individual_index: int,
) -> DeepLabCutPoseFrame:
    individual = _parse_individual(
        item,
        frame_index=frame_index,
        individual_index=individual_index,
    )
    return DeepLabCutPoseFrame(
        frame_index=frame_index,
        dog_box=individual.dog_box,
        bbox_confidence=individual.bbox_confidence,
        pose=individual.pose,
    )


def _parse_individual(
    item: Any,
    *,
    frame_index: int,
    individual_index: int,
) -> DeepLabCutIndividual:
    if not isinstance(item, dict):
        raise ValueError(f"Frame {frame_index}: expected an object.")

    try:
        raw_points = item["bodyparts"][individual_index]
        raw_box = item["bboxes"][individual_index]
        bbox_confidence = float(item["bbox_scores"][individual_index])
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(
            f"Frame {frame_index}: missing individual {individual_index} data."
        ) from exc

    if len(raw_points) != len(SUPERANIMAL_QUADRUPED_BODY_PARTS):
        raise ValueError(
            f"Frame {frame_index}: expected "
            f"{len(SUPERANIMAL_QUADRUPED_BODY_PARTS)} bodyparts, "
            f"got {len(raw_points)}."
        )
    if len(raw_box) != 4:
        raise ValueError(f"Frame {frame_index}: bbox must contain four values.")
    if not 0.0 <= bbox_confidence <= 1.0:
        raise ValueError(f"Frame {frame_index}: invalid bbox confidence.")

    keypoints: dict[KeypointName, DogKeypoint] = {}
    for native_name, raw_point in zip(SUPERANIMAL_QUADRUPED_BODY_PARTS, raw_points):
        project_name = SUPERANIMAL_TO_PROJECT.get(native_name)
        if project_name is None:
            continue
        if len(raw_point) != 3:
            raise ValueError(
                f"Frame {frame_index}: bodypart {native_name} must be [x, y, confidence]."
            )
        keypoints[project_name] = DogKeypoint(
            x=float(raw_point[0]),
            y=float(raw_point[1]),
            confidence=float(raw_point[2]),
        )

    box = BoundingBox(
        x_min=floor(float(raw_box[0])),
        y_min=floor(float(raw_box[1])),
        x_max=ceil(float(raw_box[2])),
        y_max=ceil(float(raw_box[3])),
    )
    if box.area == 0:
        raise ValueError(f"Frame {frame_index}: bbox area must be positive.")

    return DeepLabCutIndividual(
        detection_index=individual_index,
        dog_box=box,
        bbox_confidence=bbox_confidence,
        pose=DogPose(keypoints),
    )
