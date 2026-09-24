from __future__ import annotations

import argparse
from pathlib import Path

from canine_gait_cv.dataset import (
    build_tracked_gait_csv_rows,
    write_tracked_gait_csv,
)
from canine_gait_cv.pose import load_superanimal_multi_json
from canine_gait_cv.quality import assess_detection_quality
from canine_gait_cv.tracking import MultiDogTracker
from canine_gait_cv.video.reader import read_video_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create raw and quality-filtered tracked gait datasets "
            "from DeepLabCut multi-animal JSON."
        )
    )

    parser.add_argument("video", type=Path)
    parser.add_argument("json", type=Path)

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/datasets"),
    )
    parser.add_argument(
        "--max-missed-frames",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--max-normalized-distance",
        type=float,
        default=1.5,
    )
    parser.add_argument(
        "--iou-weight",
        type=float,
        default=0.35,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    metadata = read_video_metadata(args.video)
    pose_frames = load_superanimal_multi_json(args.json)

    if len(pose_frames) != metadata.frame_count:
        raise ValueError(
            "Video and JSON frame counts do not match: "
            f"{metadata.frame_count} != {len(pose_frames)}."
        )

    quality_by_frame = {
        frame.frame_index: {
            assessment.detection_index: assessment
            for assessment in assess_detection_quality(frame)
        }
        for frame in pose_frames
    }

    tracked_frames = MultiDogTracker(
        max_missed_frames=args.max_missed_frames,
        max_normalized_distance=args.max_normalized_distance,
        iou_weight=args.iou_weight,
        max_active_tracks=None,
    ).track_frames(pose_frames)

    raw_rows: list[dict[str, object]] = []
    clean_rows: list[dict[str, object]] = []

    for frame in tracked_frames:
        timestamp = frame.frame_index / metadata.fps
        quality = quality_by_frame[frame.frame_index]

        raw_rows.extend(
            build_tracked_gait_csv_rows(
                frame,
                timestamp_seconds=timestamp,
                frame_width=metadata.width,
                frame_height=metadata.height,
                quality_by_detection_index=quality,
                accepted_only=False,
            )
        )

        clean_rows.extend(
            build_tracked_gait_csv_rows(
                frame,
                timestamp_seconds=timestamp,
                frame_width=metadata.width,
                frame_height=metadata.height,
                quality_by_detection_index=quality,
                accepted_only=True,
            )
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    raw_path = args.output_dir / "tracked_raw.csv"
    clean_path = args.output_dir / "tracked_clean.csv"

    write_tracked_gait_csv(raw_path, raw_rows)
    write_tracked_gait_csv(clean_path, clean_rows)

    raw_detection_count = sum(
        row["detected"] is True
        for row in raw_rows
    )
    clean_detection_count = sum(
        row["detected"] is True
        for row in clean_rows
    )

    print(f"Frames: {len(tracked_frames)}")
    print(f"Raw detections: {raw_detection_count}")
    print(f"Clean detections: {clean_detection_count}")
    print(
        "Rejected or uncertain: "
        f"{raw_detection_count - clean_detection_count}"
    )
    print(f"Raw CSV: {raw_path}")
    print(f"Clean CSV: {clean_path}")


if __name__ == "__main__":
    main()