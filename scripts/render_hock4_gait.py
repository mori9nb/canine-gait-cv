from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import pandas as pd


BODY_PARTS = (
    "back_right_thai",
    "back_right_knee",
    "back_right_hock",
    "back_right_paw",
)

COLORS = {
    "back_right_thai": (255, 80, 80),
    "back_right_knee": (80, 255, 80),
    "back_right_hock": (0, 180, 255),
    "back_right_paw": (255, 80, 255),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render four-keypoint canine gait analysis."
    )
    parser.add_argument("video", type=Path)
    parser.add_argument("kinematics_csv", type=Path)
    parser.add_argument("output_video", type=Path)
    return parser.parse_args()


def as_bool(value: object) -> bool:
    return str(value).strip().lower() == "true"


def point_from_row(
    row: pd.Series,
    bodypart: str,
) -> tuple[int, int] | None:
    confidence = row.get(f"{bodypart}_confidence")
    x = row.get(f"{bodypart}_raw_x")
    y = row.get(f"{bodypart}_raw_y")

    if (
        pd.isna(confidence)
        or float(confidence) < 0.30
        or pd.isna(x)
        or pd.isna(y)
    ):
        return None

    return int(round(float(x))), int(round(float(y)))


def draw_text_box(
    frame,
    lines: list[str],
    *,
    x: int = 30,
    y: int = 45,
) -> None:
    line_height = 38
    width = 520
    height = line_height * len(lines) + 20

    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (x - 15, y - 35),
        (x + width, y - 35 + height),
        (0, 0, 0),
        -1,
    )
    cv2.addWeighted(
        overlay,
        0.60,
        frame,
        0.40,
        0,
        frame,
    )

    for index, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (x, y + index * line_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )


def main() -> None:
    args = parse_args()

    data = pd.read_csv(args.kinematics_csv)
    data_by_frame = {
        int(row["frame_index"]): row
        for _, row in data.iterrows()
    }

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise SystemExit(f"Cannot open video: {args.video}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps <= 0.0 or width <= 0 or height <= 0:
        raise SystemExit("Invalid video metadata.")

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
        raise SystemExit(
            f"Cannot create output: {args.output_video}"
        )

    frame_index = 0
    rendered = 0
    clean_frames = 0

    while True:
        ok, frame = capture.read()
        if not ok:
            break

        row = data_by_frame.get(frame_index)

        if row is not None:
            fully_observed = as_bool(
                row["fully_observed"]
            )

            points = {
                name: point_from_row(row, name)
                for name in BODY_PARTS
            }

            if fully_observed and all(
                point is not None
                for point in points.values()
            ):
                clean_frames += 1

                ordered_points = [
                    points[name]
                    for name in BODY_PARTS
                ]

                for first, second in zip(
                    ordered_points,
                    ordered_points[1:],
                ):
                    assert first is not None
                    assert second is not None
                    cv2.line(
                        frame,
                        first,
                        second,
                        (0, 255, 255),
                        6,
                        cv2.LINE_AA,
                    )

                for name, point in points.items():
                    assert point is not None
                    cv2.circle(
                        frame,
                        point,
                        11,
                        (0, 0, 0),
                        -1,
                        cv2.LINE_AA,
                    )
                    cv2.circle(
                        frame,
                        point,
                        8,
                        COLORS[name],
                        -1,
                        cv2.LINE_AA,
                    )

                knee_angle = float(
                    row["knee_angle_degrees"]
                )
                hock_angle = float(
                    row["hock_angle_degrees"]
                )
                confidence = float(
                    row["minimum_source_confidence"]
                )

                status = "VALID"
                status_color = (80, 255, 80)

            else:
                knee_angle = float("nan")
                hock_angle = float("nan")
                confidence = float(
                    row["minimum_source_confidence"]
                )
                status = "UNCERTAIN - excluded from clean data"
                status_color = (0, 100, 255)

            lines = [
                f"Frame: {frame_index}",
                (
                    f"Time: {frame_index / fps:.2f} s"
                ),
                (
                    "Knee angle: "
                    f"{knee_angle:.1f} deg"
                    if pd.notna(knee_angle)
                    else "Knee angle: unavailable"
                ),
                (
                    "Hock angle: "
                    f"{hock_angle:.1f} deg"
                    if pd.notna(hock_angle)
                    else "Hock angle: unavailable"
                ),
                f"Minimum confidence: {confidence:.3f}",
                f"Quality: {status}",
            ]

            draw_text_box(frame, lines)

            cv2.putText(
                frame,
                status,
                (30, height - 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                status_color,
                2,
                cv2.LINE_AA,
            )

        writer.write(frame)
        rendered += 1
        frame_index += 1

    capture.release()
    writer.release()

    print("Four-keypoint gait video rendered")
    print("-" * 50)
    print(f"Input frames:  {rendered}")
    print(f"Clean frames:  {clean_frames}")
    print(f"Output:        {args.output_video}")


if __name__ == "__main__":
    main()
    