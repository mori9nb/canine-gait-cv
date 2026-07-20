from pathlib import Path

from canine_gait_cv.video.reader import iter_video_frames, read_video_metadata


video_path = Path("data/raw/dog_test.mp4")

metadata = read_video_metadata(video_path)

print("Metadata")
print("-" * 40)
print(metadata)

print("\nFirst 5 frames")
print("-" * 40)

for frame in iter_video_frames(video_path):
    print(
        f"index={frame.index}, "
        f"time={frame.timestamp_seconds:.3f}s, "
        f"shape={frame.image.shape}"
    )

    if frame.index >= 4:
        break
