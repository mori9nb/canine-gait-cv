from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from math import isfinite, sqrt

from canine_gait_cv.appearance.megadescriptor import AppearanceEmbedding
from canine_gait_cv.quality import (
    DetectionQualityAssessment,
    DetectionQualityFlag,
)


@dataclass(frozen=True)
class AppearanceGalleryEntry:
    track_id: int
    sample_count: int
    prototype: tuple[float, ...]


class AppearanceGallery:
    """Maintain clean appearance samples for every tracked dog."""

    def __init__(
        self,
        *,
        max_samples_per_track: int = 20,
        min_quality_score: float = 0.50,
    ) -> None:
        if max_samples_per_track <= 0:
            raise ValueError(
                "max_samples_per_track must be positive."
            )
        if not 0.0 <= min_quality_score <= 1.0:
            raise ValueError(
                "min_quality_score must be between 0 and 1."
            )

        self.max_samples_per_track = max_samples_per_track
        self.min_quality_score = min_quality_score
        self._dimension: int | None = None
        self._samples: dict[
            int,
            deque[tuple[float, ...]],
        ] = defaultdict(
            lambda: deque(maxlen=self.max_samples_per_track)
        )

    def update(
        self,
        track_id: int,
        embedding: AppearanceEmbedding,
        quality: DetectionQualityAssessment,
    ) -> bool:
        """Store a clean embedding and return whether it was accepted."""

        if track_id < 0:
            raise ValueError("track_id cannot be negative.")

        if not quality.accepted_for_training:
            return False

        if DetectionQualityFlag.OVERLAPPING_DETECTION in quality.flags:
            return False

        if quality.quality_score < self.min_quality_score:
            return False

        vector = self._normalized_vector(embedding.vector)

        if self._dimension is None:
            self._dimension = len(vector)
        elif len(vector) != self._dimension:
            raise ValueError(
                "All gallery embeddings must have the same dimension."
            )

        self._samples[track_id].append(vector)
        return True

    def contains(self, track_id: int) -> bool:
        return bool(self._samples.get(track_id))

    def sample_count(self, track_id: int) -> int:
        samples = self._samples.get(track_id)
        return 0 if samples is None else len(samples)

    def prototype(self, track_id: int) -> tuple[float, ...]:
        """Return the normalized mean embedding for one track."""

        samples = self._samples.get(track_id)

        if not samples:
            raise KeyError(
                f"Track {track_id} has no appearance samples."
            )

        dimension = len(samples[0])
        mean = tuple(
            sum(sample[index] for sample in samples) / len(samples)
            for index in range(dimension)
        )

        return self._normalized_vector(mean)

    def similarity(
        self,
        track_id: int,
        embedding: AppearanceEmbedding,
    ) -> float:
        prototype = self.prototype(track_id)
        candidate = self._normalized_vector(embedding.vector)

        if len(candidate) != len(prototype):
            raise ValueError(
                "Candidate embedding has an incompatible dimension."
            )

        return sum(
            first * second
            for first, second in zip(prototype, candidate)
        )

    def entry(self, track_id: int) -> AppearanceGalleryEntry:
        return AppearanceGalleryEntry(
            track_id=track_id,
            sample_count=self.sample_count(track_id),
            prototype=self.prototype(track_id),
        )

    def clear_track(self, track_id: int) -> None:
        self._samples.pop(track_id, None)

    @staticmethod
    def _normalized_vector(
        vector: tuple[float, ...],
    ) -> tuple[float, ...]:
        if not vector:
            raise ValueError("Appearance embedding cannot be empty.")

        if not all(isfinite(value) for value in vector):
            raise ValueError(
                "Appearance embedding must contain finite values."
            )

        norm = sqrt(sum(value * value for value in vector))

        if norm == 0.0:
            raise ValueError(
                "Appearance embedding cannot be a zero vector."
            )

        return tuple(value / norm for value in vector)