from pathlib import Path

import cv2

from canine_gait_cv.video.reader import read_video_metadata


raw_dir = Path("data/raw")
output_dir = Path("outputs/sample_frames")
output_dir.mkdir(parents=True, exist_ok=True)

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

capture = cv2.VideoCapture(str(video_path))

if not capture.isOpened():
    raise SystemExit(f"Could not open video: {video_path}")

# Save one frame per second, maximum 6 frames
frame_interval = int(metadata.fps)
saved_count = 0
frame_index = 0

while True:
    success, frame = capture.read()

    if not success:
        break

    if frame_index % frame_interval == 0:
        output_path = output_dir / f"frame_{frame_index:06d}.jpg"

        # Resize only for easier viewing, original video remains unchanged
        preview = cv2.resize(frame, (1280, 720))
        cv2.imwrite(str(output_path), preview)

        print(f"Saved: {output_path}")
        saved_count += 1

    if saved_count >= 6:
        break

    frame_index += 1

capture.release()

print("-" * 40)
print(f"Extracted {saved_count} sample frames from {video_path}")
