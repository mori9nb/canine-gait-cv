from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


@dataclass(frozen=True)
class VideoMetadata:
    """Basic metadata extracted from a video file."""

    path: Path
    fps: float
    frame_count: int
    width: int
    height: int

    @property
    def duration_seconds(self) -> float:
        """Return video duration in seconds."""
        if self.fps <= 0:
            return 0.0
        return self.frame_count / self.fps


@dataclass(frozen=True)
class VideoFrame:
    """Single video frame with timing information."""

    index: int
    timestamp_seconds: float
    image: np.ndarray


def read_video_metadata(video_path: str | Path) -> VideoMetadata:
    """Open a video file and extract basic metadata."""

    path = Path(video_path)

    if not path.exists():
        raise FileNotFoundError(f"Video file does not exist: {path}")

    capture = cv2.VideoCapture(str(path))

    if not capture.isOpened():
        raise RuntimeError(f"Could not open video file: {path}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    capture.release()

    return VideoMetadata(
        path=path,
        fps=fps,
        frame_count=frame_count,
        width=width,
        height=height,
    )


def iter_video_frames(video_path: str | Path) -> Iterator[VideoFrame]:
    """Yield frames from a video one by one."""

    metadata = read_video_metadata(video_path)
    capture = cv2.VideoCapture(str(metadata.path))

    if not capture.isOpened():
        raise RuntimeError(f"Could not open video file: {metadata.path}")

    frame_index = 0

    try:
        while True:
            success, frame = capture.read()

            if not success:
                break

            timestamp_seconds = frame_index / metadata.fps if metadata.fps > 0 else 0.0

            yield VideoFrame(
                index=frame_index,
                timestamp_seconds=timestamp_seconds,
                image=frame,
            )

            frame_index += 1

    finally:
        capture.release()
