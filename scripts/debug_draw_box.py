from pathlib import Path

import cv2

from canine_gait_cv.preprocessing import BoundingBox, resize_frame
from canine_gait_cv.video import iter_video_frames
from canine_gait_cv.visualization import draw_bounding_box


video_path = Path("data/raw/dog_test.mp4")
output_dir = Path("outputs/debug")
output_dir.mkdir(parents=True, exist_ok=True)

frame_iterator = iter_video_frames(video_path)
first_frame = next(frame_iterator)

resized_frame = resize_frame(first_frame.image, target_width=1280)

box = BoundingBox(
    x_min=250,
    y_min=150,
    x_max=1000,
    y_max=650,
)

debug_frame = draw_bounding_box(
    resized_frame,
    box,
    label="debug ROI",
)

output_path = output_dir / "debug_box_preview.jpg"
cv2.imwrite(str(output_path), debug_frame)

print(f"Saved debug image to: {output_path}")