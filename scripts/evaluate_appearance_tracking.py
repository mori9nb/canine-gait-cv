from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from time import perf_counter

import cv2
from PIL import Image

from canine_gait_cv.appearance import (
    AppearanceGallery,
    MegaDescriptorExtractor,
)
from canine_gait_cv.pose import load_superanimal_multi_json
from canine_gait_cv.quality import assess_detection_quality
from canine_gait_cv.tracking import MultiDogTracker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate appearance-aware multi-dog tracking."
    )
    parser.add_argument("video", type=Path)
    parser.add_argument("predictions", type=Path)
    parser.add_argument(
        "--max-missed-frames",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--max-distance",
        type=float,
        default=2.5,
    )
    parser.add_argument(
        "--appearance-weight",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--overlap-iou",
        type=float,
        default=0.20,
    )
    parser.add_argument(
        "--output-video",
        type=Path,
        default=None,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    pose_frames = load_superanimal_multi_json(
        args.predictions
    )

    extractor = MegaDescriptorExtractor.from_pretrained()
    gallery = AppearanceGallery(
        max_samples_per_track=20,
        min_quality_score=0.50,
    )
    tracker = MultiDogTracker(
        max_missed_frames=args.max_missed_frames,
        max_normalized_distance=args.max_distance,
        appearance_gallery=gallery,
        appearance_weight=args.appearance_weight,
    )

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise RuntimeError(
            f"Could not open video: {args.video}"
        )
        writer = None

    if args.output_video is not None:
        args.output_video.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        fps = capture.get(cv2.CAP_PROP_FPS)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        writer = cv2.VideoWriter(
            str(args.output_video),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )

        if not writer.isOpened():
            raise RuntimeError(
                f"Could not create video: {args.output_video}"
            )

    raw_detection_count = 0
    clean_embedding_count = 0
    tracked_detection_count = 0
    track_frames: dict[int, list[int]] = defaultdict(list)

    start_time = perf_counter()

    try:
        for pose_frame in pose_frames:
            success, bgr_frame = capture.read()
            if not success:
                raise RuntimeError(
                    f"Could not read video frame "
                    f"{pose_frame.frame_index}."
                )

            rgb_frame = cv2.cvtColor(
                bgr_frame,
                cv2.COLOR_BGR2RGB,
            )
            image = Image.fromarray(rgb_frame)

            assessments = assess_detection_quality(
                pose_frame,
                overlap_iou_threshold=args.overlap_iou,
            )
            quality_by_detection = {
                assessment.detection_index: assessment
                for assessment in assessments
            }

            embeddings = {}

            for individual in pose_frame.individuals:
                raw_detection_count += 1

                assessment = quality_by_detection[
                    individual.detection_index
                ]

                if not assessment.accepted_for_training:
                    continue

                try:
                    embeddings[
                        individual.detection_index
                    ] = extractor.extract(
                        image,
                        individual.dog_box,
                    )
                except ValueError:
                    continue

                clean_embedding_count += 1

            tracked_frame = tracker.update(
                pose_frame,
                appearance_embeddings=embeddings,
                quality_assessments=quality_by_detection,
            )
            colors = (
                (0, 255, 0),
                (255, 100, 0),
                (0, 80, 255),
                (255, 0, 255),
            )

            for tracked_individual in tracked_frame.individuals:
                detection = tracked_individual.detection
                box = detection.dog_box
                track_id = tracked_individual.track_id
                assessment = quality_by_detection[
                    detection.detection_index
                ]

                color = colors[track_id % len(colors)]
                status = (
                    "clean"
                    if assessment.accepted_for_training
                    else "uncertain"
                )

                cv2.rectangle(
                    bgr_frame,
                    (box.x_min, box.y_min),
                    (box.x_max, box.y_max),
                    color,
                    2,
                )

                cv2.putText(
                    bgr_frame,
                    f"Frame {pose_frame.frame_index}",
                    (25, 45),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.1,
                    (255, 255, 255),
                    3,
                    cv2.LINE_AA,
                )

            if writer is not None:
                writer.write(bgr_frame)

            tracked_detection_count += len(
                tracked_frame.individuals
            )

            for individual in tracked_frame.individuals:
                track_frames[individual.track_id].append(
                    tracked_frame.frame_index
                )
    finally:
        capture.release()

        if writer is not None:
            writer.release()

    elapsed = perf_counter() - start_time

    print("\nAppearance-aware tracking result")
    print("-" * 50)
    print(f"Frames: {len(pose_frames)}")
    print(f"Raw detections: {raw_detection_count}")
    print(f"Clean embeddings extracted: {clean_embedding_count}")
    print(f"Tracked detections: {tracked_detection_count}")
    print(f"Total track IDs: {len(track_frames)}")
    print(f"Runtime: {elapsed:.2f} seconds")

    for track_id, frames in sorted(
        track_frames.items(),
        key=lambda item: (-len(item[1]), item[0]),
    ):
        print(
            f"Track {track_id:2d}: "
            f"{len(frames):3d} detections, "
            f"frames {min(frames):3d}-{max(frames):3d}, "
            f"gallery samples={gallery.sample_count(track_id)}"
        )


if __name__ == "__main__":
    main()