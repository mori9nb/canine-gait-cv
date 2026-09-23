from pathlib import Path

import cv2

from canine_gait_cv.pose import load_superanimal_multi_json
from canine_gait_cv.tracking import MultiDogTracker


video_path = Path("data/raw/multi_dog_side_test.mp4")

json_path = next(
    Path("outputs/deeplabcut/multi_dog_side_before_adapt")
    .glob("*_before_adapt.json")
)

output_path = Path(
    "outputs/tracking/multi_dog_side_tracked_final.mp4"
)
output_path.parent.mkdir(parents=True, exist_ok=True)

pose_frames = load_superanimal_multi_json(json_path)

tracked_frames = MultiDogTracker(
    max_missed_frames=15,
    max_normalized_distance=2.5,
    iou_weight=0.35,
    max_active_tracks=3,
).track_frames(pose_frames)

colors = [
    (0, 255, 0),
    (255, 0, 0),
    (0, 0, 255),
]

capture = cv2.VideoCapture(str(video_path))

if not capture.isOpened():
    raise SystemExit(f"Could not open video: {video_path}")

fps = capture.get(cv2.CAP_PROP_FPS)
width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

writer = cv2.VideoWriter(
    str(output_path),
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height),
)

if not writer.isOpened():
    raise SystemExit(f"Could not create video: {output_path}")

frame_index = 0

while True:
    ok, image = capture.read()

    if not ok:
        break

    if frame_index < len(tracked_frames):
        tracked_frame = tracked_frames[frame_index]

        for item in tracked_frame.individuals:
            box = item.detection.dog_box
            track_id = item.track_id
            confidence = item.detection.bbox_confidence
            color = colors[track_id % len(colors)]

            cv2.rectangle(
                image,
                (box.x_min, box.y_min),
                (box.x_max, box.y_max),
                color,
                2,
            )

            cv2.putText(
                image,
                f"Dog {track_id} ({confidence:.2f})",
                (box.x_min, max(20, box.y_min - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )

    writer.write(image)
    frame_index += 1

capture.release()
writer.release()

print(f"Saved: {output_path}")
print(f"Frames written: {frame_index}")