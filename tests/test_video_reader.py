from pathlib import Path

from canine_gait_cv.video.reader import iter_video_frames,read_video_metadata

def test_read_video_metadata() -> None:
    video_path = Path("data/raw/dog_test.mp4")
    metadata = read_video_metadata(video_path)

    assert metadata.path == video_path
    assert metadata.fps > 0
    assert metadata.frame_count > 0
    assert metadata.width > 0
    assert metadata.height > 0
    assert metadata.duration_seconds > 0

def test_iter_video_frames_reads_first_frame() -> None:
    video_path = Path("data/raw/dog_test.mp4")

    frame_iterator = iter_video_frames(video_path)
    first_frame = next(frame_iterator)

    assert first_frame.index == 0
    assert first_frame.timestamp_seconds == 0.0
    assert first_frame.image is not None
    assert first_frame.image.shape[0] > 0
    assert first_frame.image.shape[1] > 0
    assert first_frame.image.shape[2] == 3