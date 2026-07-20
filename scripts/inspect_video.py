from pathlib import Path

from canine_gait_cv.video.reader import read_video_metadata


raw_dir = Path("data/raw")

videos = sorted(
    list(raw_dir.glob("*.mp4"))
    + list(raw_dir.glob("*.avi"))
    + list(raw_dir.glob("*.mov"))
    + list(raw_dir.glob("*.mkv"))
)

if not videos:
    raise SystemExit("No video file found in data/raw")

video_path = videos[0]
metadata = read_video_metadata(video_path)

print("Video metadata")
print("-" * 40)
print(f"Path:      {metadata.path}")
print(f"FPS:       {metadata.fps}")
print(f"Frames:    {metadata.frame_count}")
print(f"Width:     {metadata.width}")
print(f"Height:    {metadata.height}")
print(f"Duration:  {metadata.duration_seconds:.2f} seconds")
