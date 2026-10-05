from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from canine_gait_cv.gait import (
    build_hind_limb_angle_series,
    select_primary_individual,
)
from canine_gait_cv.pose import (
    KeypointName,
    load_superanimal_multi_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render single-dog hind-limb gait validation video."
    )
    parser.add_argument("video", type=Path)
    parser.add_argument("pose_json", type=Path)
    parser.add_argument("output_video", type=Path)
    parser.add_argument(
        "--side",
        choices=("left", "right"),
        default="right",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.30,
    )
    return parser.parse_args()


def draw_label(
    frame,
    text: str,
    position: tuple[int, int],
    color: tuple[int, int, int],
    *,
    scale: float = 0.8,
) -> None:
    x, y = position

    cv2.putText(
        frame,
        text,
        (x + 2, y + 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 0, 0),
        5,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        2,
        cv2.LINE_AA,
    )


def main() -> None:
    args = parse_args()

    pose_frames = load_superanimal_multi_json(
        args.pose_json
    )

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise SystemExit(
            f"Could not open video: {args.video}"
        )

    fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_frame_count = int(
        capture.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    if fps <= 0.0:
        raise SystemExit("Video FPS is invalid.")

    if video_frame_count != len(pose_frames):
        raise SystemExit(
            "Video and pose JSON frame counts differ: "
            f"{video_frame_count} != {len(pose_frames)}"
        )

    angle_series = build_hind_limb_angle_series(
        pose_frames,
        fps=fps,
        side=args.side,
        min_confidence=args.min_confidence,
    )

    args.output_video.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    writer = cv2.VideoWriter(
        str(args.output_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        capture.release()
        raise SystemExit(
            f"Could not create video: {args.output_video}"
        )

    if args.side == "right":
        names = (
            KeypointName.RIGHT_HIP,
            KeypointName.RIGHT_KNEE,
            KeypointName.RIGHT_BACK_PAW,
        )
    else:
        names = (
            KeypointName.LEFT_HIP,
            KeypointName.LEFT_KNEE,
            KeypointName.LEFT_BACK_PAW,
        )

    observed_color = (0, 220, 0)
    estimated_color = (0, 165, 255)
    joint_colors = (
        (255, 100, 0),
        (0, 255, 255),
        (255, 0, 255),
    )

    rendered_frames = 0

    for pose_frame, angle_frame in zip(
        pose_frames,
        angle_series,
    ):
        ok, image = capture.read()
        if not ok:
            break

        status_color = (
            observed_color
            if angle_frame.fully_observed
            else estimated_color
        )
        status_text = (
            "OBSERVED"
            if angle_frame.fully_observed
            else "ESTIMATED"
        )

        individual = select_primary_individual(
            pose_frame,
            min_keypoint_confidence=args.min_confidence,
        )

        if individual is not None:
            box = individual.dog_box

            cv2.rectangle(
                image,
                (box.x_min, box.y_min),
                (box.x_max, box.y_max),
                status_color,
                3,
            )

            points = [
                individual.pose.get(name)
                for name in names
            ]

            reliable = [
                point.confidence >= args.min_confidence
                for point in points
            ]

            pixel_points = [
                (round(point.x), round(point.y))
                for point in points
            ]

            # Draw a limb segment only when both endpoints
            # are reliable in the original DLC output.
            for index in range(2):
                if reliable[index] and reliable[index + 1]:
                    cv2.line(
                        image,
                        pixel_points[index],
                        pixel_points[index + 1],
                        status_color,
                        6,
                        cv2.LINE_AA,
                    )

            labels = ("HIP", "KNEE", "PAW")

            for point, pixel, is_reliable, label, color in zip(
                points,
                pixel_points,
                reliable,
                labels,
                joint_colors,
            ):
                if not is_reliable:
                    continue

                cv2.circle(
                    image,
                    pixel,
                    11,
                    (0, 0, 0),
                    -1,
                    cv2.LINE_AA,
                )
                cv2.circle(
                    image,
                    pixel,
                    7,
                    color,
                    -1,
                    cv2.LINE_AA,
                )

                draw_label(
                    image,
                    f"{label} {point.confidence:.2f}",
                    (pixel[0] + 12, pixel[1] - 10),
                    color,
                    scale=0.55,
                )

        panel_width = min(720, width - 40)

        overlay = image.copy()
        cv2.rectangle(
            overlay,
            (20, 20),
            (20 + panel_width, 190),
            (0, 0, 0),
            -1,
        )
        cv2.addWeighted(
            overlay,
            0.60,
            image,
            0.40,
            0.0,
            image,
        )

        draw_label(
            image,
            f"Frame: {angle_frame.frame_index}",
            (45, 65),
            (255, 255, 255),
        )
        draw_label(
            image,
            f"Time: {angle_frame.time_seconds:.2f} s",
            (45, 105),
            (255, 255, 255),
        )

        if angle_frame.angle_degrees is None:
            angle_text = "Angle: unavailable"
        else:
            angle_text = (
                f"Angle: {angle_frame.angle_degrees:.1f} deg"
            )

        draw_label(
            image,
            angle_text,
            (45, 145),
            status_color,
        )
        draw_label(
            image,
            status_text,
            (400, 65),
            status_color,
        )

        writer.write(image)
        rendered_frames += 1

    capture.release()
    writer.release()

    print("Gait validation video")
    print("-" * 50)
    print(f"Input video:     {args.video}")
    print(f"Pose frames:     {len(pose_frames)}")
    print(f"Rendered frames: {rendered_frames}")
    print(f"Resolution:      {width}x{height}")
    print(f"FPS:             {fps}")
    print(f"Output video:    {args.output_video}")


if __name__ == "__main__":
    main()