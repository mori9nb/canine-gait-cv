from pathlib import Path

import cv2
import numpy as np

from canine_gait_cv.video.reader import iter_video_frames, read_video_metadata


def create_test_video(tmp_path: Path) -> Path:
    video_path = tmp_path / "test_video.avi"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        20.0,
        (64, 48),
    )
    if not writer.isOpened():
        raise RuntimeError("OpenCV could not create the test video.")

    try:
        for value in (0, 80, 160):
            frame = np.full((48, 64, 3), value, dtype=np.uint8)
            writer.write(frame)
    finally:
        writer.release()

    return video_path


def test_read_video_metadata(tmp_path: Path) -> None:
    video_path = create_test_video(tmp_path)
    metadata = read_video_metadata(video_path)

    assert metadata.path == video_path
    assert metadata.fps == 20.0
    assert metadata.frame_count == 3
    assert metadata.width == 64
    assert metadata.height == 48
    assert metadata.duration_seconds == 0.15


def test_iter_video_frames_reads_first_frame(tmp_path: Path) -> None:
    video_path = create_test_video(tmp_path)

    frame_iterator = iter_video_frames(video_path)
    first_frame = next(frame_iterator)

    assert first_frame.index == 0
    assert first_frame.timestamp_seconds == 0.0
    assert first_frame.image.shape == (48, 64, 3)
