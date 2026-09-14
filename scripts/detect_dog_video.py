from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from canine_gait_cv.detection import YOLODetector, select_largest_detection
from canine_gait_cv.preprocessing import resize_frame
from canine_gait_cv.video import iter_video_frames, read_video_metadata
from canine_gait_cv.visualization import draw_bounding_box


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect the largest dog in each video frame with YOLO.",
    )
    parser.add_argument("input", type=Path, help="Path to the input video.")
    parser.add_argument("output", type=Path, help="Path for the annotated video.")
    parser.add_argument("--weights", default="yolo11n.pt", help="YOLO weights path.")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--device", help="Inference device, for example cpu, 0, or mps.")
    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="Resize frames to this width before detection; use 0 for original size.",
    )
    parser.add_argument("--max-frames", type=int, help="Optional debug frame limit.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.input.resolve() == args.output.resolve():
        raise ValueError("Input and output video paths must be different.")
    if args.width < 0:
        raise ValueError("--width cannot be negative.")
    if args.max_frames is not None and args.max_frames <= 0:
        raise ValueError("--max-frames must be positive.")

    metadata = read_video_metadata(args.input)
    detector = YOLODetector(
        model_path=args.weights,
        min_confidence=args.confidence,
        device=args.device,
        image_size=args.image_size,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer: cv2.VideoWriter | None = None
    processed_frames = 0
    detected_frames = 0

    try:
        for video_frame in iter_video_frames(args.input):
            frame = video_frame.image
            if args.width:
                frame = resize_frame(frame, target_width=args.width)

            detection = select_largest_detection(detector.detect(frame))
            annotated_frame = frame
            if detection is not None:
                detected_frames += 1
                annotated_frame = draw_bounding_box(
                    frame,
                    detection.box,
                    label=f"{detection.class_name} {detection.confidence:.2f}",
                )

            if writer is None:
                height, width = annotated_frame.shape[:2]
                fps = metadata.fps if metadata.fps > 0 else 30.0
                writer = cv2.VideoWriter(
                    str(args.output),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps,
                    (width, height),
                )
                if not writer.isOpened():
                    raise RuntimeError(f"Could not create output video: {args.output}")

            writer.write(annotated_frame)
            processed_frames += 1
            if args.max_frames is not None and processed_frames >= args.max_frames:
                break
    finally:
        if writer is not None:
            writer.release()

    if processed_frames == 0:
        raise RuntimeError(f"Input video contains no readable frames: {args.input}")

    detection_rate = detected_frames / processed_frames
    print(f"Saved annotated video to: {args.output}")
    print(f"Frames processed: {processed_frames}")
    print(f"Frames with a dog detection: {detected_frames}")
    print(f"Detection rate: {detection_rate:.1%}")


if __name__ == "__main__":
    main()
