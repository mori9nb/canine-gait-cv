from __future__ import annotations

from dataclasses import dataclass
from math import hypot

from canine_gait_cv.pose import DeepLabCutIndividual, DeepLabCutMultiPoseFrame
from canine_gait_cv.preprocessing import BoundingBox


@dataclass(frozen=True)
class TrackedIndividual:
    """One DeepLabCut detection associated with a persistent temporal ID."""

    track_id: int
    detection: DeepLabCutIndividual


@dataclass(frozen=True)
class TrackedPoseFrame:
    """All tracked dogs observed in one frame."""

    frame_index: int
    individuals: tuple[TrackedIndividual, ...]


@dataclass
class _TrackState:
    track_id: int
    box: BoundingBox
    center_x: float
    center_y: float
    velocity_x: float
    velocity_y: float
    last_seen_frame: int


class MultiDogTracker:
    """Associate per-frame dog detections using motion and box overlap.

    This is a deterministic short-term geometric tracker.  It does not use
    appearance embeddings, so a track that has expired is deliberately given
    a new ID if the dog later re-enters the scene.
    """

    def __init__(
        self,
        *,
        max_missed_frames: int = 5,
        max_normalized_distance: float = 1.5,
        iou_weight: float = 0.35,
    ) -> None:
        if max_missed_frames < 0:
            raise ValueError("max_missed_frames cannot be negative.")
        if max_normalized_distance <= 0.0:
            raise ValueError("max_normalized_distance must be positive.")
        if not 0.0 <= iou_weight <= 1.0:
            raise ValueError("iou_weight must be between 0 and 1.")

        self.max_missed_frames = max_missed_frames
        self.max_normalized_distance = max_normalized_distance
        self.iou_weight = iou_weight
        self._tracks: dict[int, _TrackState] = {}
        self._next_track_id = 0
        self._last_frame_index: int | None = None

    def reset(self) -> None:
        """Forget all active tracks and restart ID allocation from zero."""

        self._tracks.clear()
        self._next_track_id = 0
        self._last_frame_index = None

    def update(self, frame: DeepLabCutMultiPoseFrame) -> TrackedPoseFrame:
        """Assign stable IDs to the detections in one chronological frame."""

        if (
            self._last_frame_index is not None
            and frame.frame_index <= self._last_frame_index
        ):
            raise ValueError("frame indices must be strictly increasing.")
        self._last_frame_index = frame.frame_index
        self._expire_old_tracks(frame.frame_index)

        assignments = self._associate(frame)
        tracked: list[TrackedIndividual] = []
        for detection_index, detection in enumerate(frame.individuals):
            track_id = assignments.get(detection_index)
            if track_id is None:
                track_id = self._create_track(detection, frame.frame_index)
            else:
                self._update_track(track_id, detection, frame.frame_index)
            tracked.append(TrackedIndividual(track_id=track_id, detection=detection))

        return TrackedPoseFrame(
            frame_index=frame.frame_index,
            individuals=tuple(tracked),
        )

    def track_frames(
        self,
        frames: list[DeepLabCutMultiPoseFrame],
    ) -> list[TrackedPoseFrame]:
        """Track a chronological sequence using the current tracker state."""

        return [self.update(frame) for frame in frames]

    def _associate(self, frame: DeepLabCutMultiPoseFrame) -> dict[int, int]:
        candidates: list[tuple[float, int, int]] = []
        for detection_index, detection in enumerate(frame.individuals):
            det_x, det_y = _box_center(detection.dog_box)
            for track_id, track in self._tracks.items():
                gap = frame.frame_index - track.last_seen_frame
                predicted_x = track.center_x + track.velocity_x * gap
                predicted_y = track.center_y + track.velocity_y * gap
                scale = max(_box_diagonal(track.box), _box_diagonal(detection.dog_box), 1.0)
                distance = hypot(det_x - predicted_x, det_y - predicted_y) / scale
                if distance > self.max_normalized_distance:
                    continue
                cost = distance + self.iou_weight * (1.0 - _box_iou(track.box, detection.dog_box))
                candidates.append((cost, track_id, detection_index))

        assignments: dict[int, int] = {}
        used_tracks: set[int] = set()
        for _, track_id, detection_index in sorted(candidates):
            if track_id in used_tracks or detection_index in assignments:
                continue
            assignments[detection_index] = track_id
            used_tracks.add(track_id)
        return assignments

    def _create_track(
        self,
        detection: DeepLabCutIndividual,
        frame_index: int,
    ) -> int:
        track_id = self._next_track_id
        self._next_track_id += 1
        center_x, center_y = _box_center(detection.dog_box)
        self._tracks[track_id] = _TrackState(
            track_id=track_id,
            box=detection.dog_box,
            center_x=center_x,
            center_y=center_y,
            velocity_x=0.0,
            velocity_y=0.0,
            last_seen_frame=frame_index,
        )
        return track_id

    def _update_track(
        self,
        track_id: int,
        detection: DeepLabCutIndividual,
        frame_index: int,
    ) -> None:
        track = self._tracks[track_id]
        center_x, center_y = _box_center(detection.dog_box)
        gap = frame_index - track.last_seen_frame
        track.velocity_x = (center_x - track.center_x) / gap
        track.velocity_y = (center_y - track.center_y) / gap
        track.box = detection.dog_box
        track.center_x = center_x
        track.center_y = center_y
        track.last_seen_frame = frame_index

    def _expire_old_tracks(self, frame_index: int) -> None:
        expired = [
            track_id
            for track_id, track in self._tracks.items()
            if frame_index - track.last_seen_frame > self.max_missed_frames + 1
        ]
        for track_id in expired:
            del self._tracks[track_id]


def _box_center(box: BoundingBox) -> tuple[float, float]:
    return ((box.x_min + box.x_max) / 2.0, (box.y_min + box.y_max) / 2.0)


def _box_diagonal(box: BoundingBox) -> float:
    return hypot(box.width, box.height)


def _box_iou(first: BoundingBox, second: BoundingBox) -> float:
    x_min = max(first.x_min, second.x_min)
    y_min = max(first.y_min, second.y_min)
    x_max = min(first.x_max, second.x_max)
    y_max = min(first.y_max, second.y_max)
    intersection = max(0, x_max - x_min) * max(0, y_max - y_min)
    union = first.area + second.area - intersection
    return 0.0 if union == 0 else intersection / union
