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

KEYPOINT_COLORS = {
    "back_right_thai": (255, 80, 80),
    "back_right_knee": (80, 255, 80),
    "back_right_hock": (0, 180, 255),
    "back_right_paw": (255, 80, 255),
}

PHASE_COLORS = {
    "stance": (0, 165, 255),
    "swing": (255, 220, 0),
    "unknown": (140, 140, 140),
}

EVENT_COLORS = {
    "contact": (80, 255, 80),
    "toe_off": (50, 50, 255),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render four-keypoint gait angles, "
            "stance/swing phases and gait events."
        )
    )

    parser.add_argument(
        "video",
        type=Path,
    )
    parser.add_argument(
        "kinematics_csv",
        type=Path,
    )
    parser.add_argument(
        "phase_csv",
        type=Path,
    )
    parser.add_argument(
        "events_csv",
        type=Path,
    )
    parser.add_argument(
        "output_video",
        type=Path,
    )
    parser.add_argument(
        "--event-display-frames",
        type=int,
        default=6,
        help=(
            "Number of frames for which CONTACT or "
            "TOE-OFF remains visible."
        ),
    )

    return parser.parse_args()


def as_bool(value: object) -> bool:
    return (
        str(value)
        .strip()
        .lower()
        in {"true", "1"}
    )


def optional_integer(
    value: object,
) -> int | None:
    if pd.isna(value):
        return None

    return int(value)


def point_from_row(
    row: pd.Series,
    bodypart: str,
) -> tuple[int, int] | None:
    confidence = row.get(
        f"{bodypart}_confidence"
    )
    x = row.get(
        f"{bodypart}_raw_x"
    )
    y = row.get(
        f"{bodypart}_raw_y"
    )

    if (
        pd.isna(confidence)
        or float(confidence) < 0.30
        or pd.isna(x)
        or pd.isna(y)
    ):
        return None

    return (
        int(round(float(x))),
        int(round(float(y))),
    )


def draw_transparent_rectangle(
    frame,
    *,
    top_left: tuple[int, int],
    bottom_right: tuple[int, int],
    color: tuple[int, int, int],
    alpha: float,
) -> None:
    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        top_left,
        bottom_right,
        color,
        -1,
    )

    cv2.addWeighted(
        overlay,
        alpha,
        frame,
        1.0 - alpha,
        0,
        frame,
    )


def draw_text_box(
    frame,
    lines: list[str],
    *,
    x: int = 30,
    y: int = 45,
) -> None:
    line_height = 38
    width = 560
    height = (
        line_height * len(lines)
        + 20
    )

    draw_transparent_rectangle(
        frame,
        top_left=(
            x - 15,
            y - 35,
        ),
        bottom_right=(
            x + width,
            y - 35 + height,
        ),
        color=(0, 0, 0),
        alpha=0.62,
    )

    for index, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (
                x,
                y + index * line_height,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )


def draw_phase_badge(
    frame,
    *,
    phase: str,
    stride_index: int | None,
    frame_width: int,
) -> None:
    color = PHASE_COLORS.get(
        phase,
        PHASE_COLORS["unknown"],
    )

    label = phase.upper()

    if stride_index is not None:
        label += (
            f"  |  STRIDE {stride_index}"
        )
    else:
        label += "  |  PARTIAL CYCLE"

    box_width = 560
    box_height = 82
    x1 = frame_width - box_width - 30
    y1 = 30
    x2 = frame_width - 30
    y2 = y1 + box_height

    draw_transparent_rectangle(
        frame,
        top_left=(x1, y1),
        bottom_right=(x2, y2),
        color=color,
        alpha=0.78,
    )

    cv2.putText(
        frame,
        label,
        (x1 + 22, y1 + 53),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.05,
        (20, 20, 20),
        3,
        cv2.LINE_AA,
    )


def draw_event_banner(
    frame,
    *,
    event_type: str,
    frame_width: int,
    frame_height: int,
) -> None:
    color = EVENT_COLORS[event_type]

    if event_type == "contact":
        label = "CONTACT"
    else:
        label = "TOE-OFF"

    text_size, _ = cv2.getTextSize(
        label,
        cv2.FONT_HERSHEY_SIMPLEX,
        1.35,
        4,
    )

    text_width = text_size[0]
    box_width = text_width + 90
    box_height = 90

    x1 = (
        frame_width - box_width
    ) // 2
    y1 = frame_height - 135
    x2 = x1 + box_width
    y2 = y1 + box_height

    draw_transparent_rectangle(
        frame,
        top_left=(x1, y1),
        bottom_right=(x2, y2),
        color=color,
        alpha=0.78,
    )

    cv2.putText(
        frame,
        label,
        (
            x1 + 45,
            y1 + 60,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.35,
        (20, 20, 20),
        4,
        cv2.LINE_AA,
    )


def active_event(
    frame_index: int,
    events: list[dict[str, object]],
    *,
    display_frames: int,
) -> str | None:
    active: str | None = None

    for event in events:
        event_frame = int(
            event["frame_index"]
        )

        if (
            event_frame
            <= frame_index
            < event_frame + display_frames
        ):
            active = str(
                event["event_type"]
            )

    return active


def draw_skeleton(
    frame,
    *,
    points: dict[
        str,
        tuple[int, int] | None,
    ],
    phase: str,
) -> None:
    ordered_points = [
        points[name]
        for name in BODY_PARTS
    ]

    phase_color = PHASE_COLORS.get(
        phase,
        PHASE_COLORS["unknown"],
    )

    for first, second in zip(
        ordered_points,
        ordered_points[1:],
    ):
        if (
            first is None
            or second is None
        ):
            continue

        cv2.line(
            frame,
            first,
            second,
            phase_color,
            7,
            cv2.LINE_AA,
        )

    for name, point in points.items():
        if point is None:
            continue

        cv2.circle(
            frame,
            point,
            12,
            (0, 0, 0),
            -1,
            cv2.LINE_AA,
        )

        cv2.circle(
            frame,
            point,
            8,
            KEYPOINT_COLORS[name],
            -1,
            cv2.LINE_AA,
        )


def main() -> None:
    args = parse_args()

    if args.event_display_frames <= 0:
        raise ValueError(
            "event-display-frames must be positive."
        )

    kinematics = pd.read_csv(
        args.kinematics_csv
    )
    phases = pd.read_csv(
        args.phase_csv
    )
    events_data = pd.read_csv(
        args.events_csv
    )

    kinematics_by_frame = {
        int(row["frame_index"]): row
        for _, row in kinematics.iterrows()
    }

    phases_by_frame = {
        int(row["frame_index"]): row
        for _, row in phases.iterrows()
    }

    events = [
        {
            "frame_index": int(
                row["frame_index"]
            ),
            "event_type": str(
                row["event_type"]
            ),
        }
        for _, row in events_data.iterrows()
    ]

    capture = cv2.VideoCapture(
        str(args.video)
    )

    if not capture.isOpened():
        raise SystemExit(
            f"Cannot open video: {args.video}"
        )

    fps = capture.get(
        cv2.CAP_PROP_FPS
    )
    width = int(
        capture.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )
    height = int(
        capture.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    if (
        fps <= 0.0
        or width <= 0
        or height <= 0
    ):
        raise SystemExit(
            "Invalid video metadata."
        )

    args.output_video.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    writer = cv2.VideoWriter(
        str(args.output_video),
        cv2.VideoWriter_fourcc(
            *"mp4v"
        ),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        raise SystemExit(
            "Cannot create output video: "
            f"{args.output_video}"
        )

    frame_index = 0
    rendered_frames = 0
    clean_frames = 0
    stance_frames = 0
    swing_frames = 0
    unknown_frames = 0

    while True:
        ok, frame = capture.read()

        if not ok:
            break

        kinematics_row = (
            kinematics_by_frame.get(
                frame_index
            )
        )
        phase_row = (
            phases_by_frame.get(
                frame_index
            )
        )

        if phase_row is None:
            phase = "unknown"
            stride_index = None
        else:
            phase = str(
                phase_row["phase"]
            )

            stride_index = optional_integer(
                phase_row.get(
                    "stride_index"
                )
            )

        if phase == "stance":
            stance_frames += 1
        elif phase == "swing":
            swing_frames += 1
        else:
            unknown_frames += 1

        if kinematics_row is not None:
            fully_observed = as_bool(
                kinematics_row[
                    "fully_observed"
                ]
            )

            points = {
                name: point_from_row(
                    kinematics_row,
                    name,
                )
                for name in BODY_PARTS
            }

            if (
                fully_observed
                and all(
                    point is not None
                    for point in points.values()
                )
            ):
                clean_frames += 1

                draw_skeleton(
                    frame,
                    points=points,
                    phase=phase,
                )

                knee_angle = float(
                    kinematics_row[
                        "knee_angle_degrees"
                    ]
                )
                hock_angle = float(
                    kinematics_row[
                        "hock_angle_degrees"
                    ]
                )
                confidence = float(
                    kinematics_row[
                        "minimum_source_confidence"
                    ]
                )

                quality = "VALID"

            else:
                knee_angle = float("nan")
                hock_angle = float("nan")

                confidence = float(
                    kinematics_row.get(
                        "minimum_source_confidence",
                        0.0,
                    )
                )

                quality = (
                    "UNCERTAIN - excluded "
                    "from clean data"
                )

            lines = [
                f"Frame: {frame_index}",
                (
                    f"Time: "
                    f"{frame_index / fps:.2f} s"
                ),
                (
                    f"Knee angle: "
                    f"{knee_angle:.1f} deg"
                    if pd.notna(knee_angle)
                    else "Knee angle: unavailable"
                ),
                (
                    f"Hock angle: "
                    f"{hock_angle:.1f} deg"
                    if pd.notna(hock_angle)
                    else "Hock angle: unavailable"
                ),
                (
                    f"Minimum confidence: "
                    f"{confidence:.3f}"
                ),
                f"Quality: {quality}",
            ]

            draw_text_box(
                frame,
                lines,
            )

        draw_phase_badge(
            frame,
            phase=phase,
            stride_index=stride_index,
            frame_width=width,
        )

        event_type = active_event(
            frame_index,
            events,
            display_frames=(
                args.event_display_frames
            ),
        )

        if event_type is not None:
            draw_event_banner(
                frame,
                event_type=event_type,
                frame_width=width,
                frame_height=height,
            )

        writer.write(frame)

        rendered_frames += 1
        frame_index += 1

    capture.release()
    writer.release()

    print(
        "Four-keypoint stride video rendered"
    )
    print("-" * 55)
    print(
        f"Input frames:    "
        f"{rendered_frames}"
    )
    print(
        f"Clean frames:    "
        f"{clean_frames}"
    )
    print(
        f"Stance frames:   "
        f"{stance_frames}"
    )
    print(
        f"Swing frames:    "
        f"{swing_frames}"
    )
    print(
        f"Unknown frames:  "
        f"{unknown_frames}"
    )
    print(
        f"Events:          "
        f"{len(events)}"
    )
    print(
        f"Output:          "
        f"{args.output_video}"
    )


if __name__ == "__main__":
    main()
