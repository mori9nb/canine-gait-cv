from __future__ import annotations

import argparse
from math import acos, degrees, hypot
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from canine_gait_cv.pose import DogKeypoint
from canine_gait_cv.smoothing.kalman_rts import (
    smooth_keypoint_trajectory,
)
from canine_gait_cv.video.reader import read_video_metadata


BODY_PARTS = (
    "back_right_thai",
    "back_right_knee",
    "back_right_hock",
    "back_right_paw",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export right knee and hock angles from "
            "four-keypoint DeepLabCut output."
        )
    )
    parser.add_argument("video", type=Path)
    parser.add_argument("pose_h5", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/gait/single_dog_hock4"),
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.30,
    )
    parser.add_argument(
        "--max-estimated-gap",
        type=int,
        default=3,
        help="Maximum internal missing gap estimated by the smoother.",
    )
    return parser.parse_args()


def joint_angle_degrees(
    proximal: DogKeypoint,
    joint: DogKeypoint,
    distal: DogKeypoint,
) -> float | None:
    first_x = proximal.x - joint.x
    first_y = proximal.y - joint.y
    second_x = distal.x - joint.x
    second_y = distal.y - joint.y

    first_length = hypot(first_x, first_y)
    second_length = hypot(second_x, second_y)

    if first_length == 0.0 or second_length == 0.0:
        return None

    cosine = (
        first_x * second_x + first_y * second_y
    ) / (first_length * second_length)

    cosine = max(-1.0, min(1.0, cosine))
    return degrees(acos(cosine))


def load_raw_keypoints(
    path: Path,
) -> tuple[
    pd.DataFrame,
    dict[str, list[DogKeypoint | None]],
]:
    data = pd.read_hdf(path)

    expected_levels = {
        "scorer",
        "bodyparts",
        "coords",
    }
    if set(data.columns.names) != expected_levels:
        raise ValueError(
            "Expected DLC column levels: "
            "scorer, bodyparts and coords."
        )

    scorers = data.columns.get_level_values(
        "scorer"
    ).unique()

    if len(scorers) != 1:
        raise ValueError("Expected exactly one scorer.")

    scorer = scorers[0]
    available = set(
        data.columns.get_level_values("bodyparts")
    )

    missing = [
        bodypart
        for bodypart in BODY_PARTS
        if bodypart not in available
    ]
    if missing:
        raise ValueError(
            "Missing bodyparts: " + ", ".join(missing)
        )

    trajectories: dict[
        str,
        list[DogKeypoint | None],
    ] = {}

    for bodypart in BODY_PARTS:
        points: list[DogKeypoint | None] = []

        for _, row in data.iterrows():
            confidence = float(
                row[(scorer, bodypart, "likelihood")]
            )

            if confidence < 0.0:
                points.append(None)
                continue

            points.append(
                DogKeypoint(
                    x=float(row[(scorer, bodypart, "x")]),
                    y=float(row[(scorer, bodypart, "y")]),
                    confidence=confidence,
                )
            )

        trajectories[bodypart] = points

    return data, trajectories


def build_angle_measurements(
    trajectories: dict[
        str,
        list[DogKeypoint | None],
    ],
    *,
    min_confidence: float,
) -> tuple[
    list[DogKeypoint | None],
    list[DogKeypoint | None],
    list[bool],
    list[float],
]:
    frame_count = len(
        trajectories["back_right_thai"]
    )

    knee_measurements: list[
        DogKeypoint | None
    ] = []
    hock_measurements: list[
        DogKeypoint | None
    ] = []
    fully_observed: list[bool] = []
    minimum_confidences: list[float] = []

    for frame_index in range(frame_count):
        thigh = trajectories[
            "back_right_thai"
        ][frame_index]
        knee = trajectories[
            "back_right_knee"
        ][frame_index]
        hock = trajectories[
            "back_right_hock"
        ][frame_index]
        paw = trajectories[
            "back_right_paw"
        ][frame_index]

        points = (thigh, knee, hock, paw)

        confidences = [
            0.0 if point is None else point.confidence
            for point in points
        ]
        minimum_confidence = min(confidences)
        minimum_confidences.append(
            minimum_confidence
        )

        valid = (
            all(point is not None for point in points)
            and minimum_confidence >= min_confidence
        )

        if not valid:
            knee_measurements.append(None)
            hock_measurements.append(None)
            fully_observed.append(False)
            continue

        assert thigh is not None
        assert knee is not None
        assert hock is not None
        assert paw is not None

        knee_angle = joint_angle_degrees(
            thigh,
            knee,
            hock,
        )
        hock_angle = joint_angle_degrees(
            knee,
            hock,
            paw,
        )

        if knee_angle is None or hock_angle is None:
            knee_measurements.append(None)
            hock_measurements.append(None)
            fully_observed.append(False)
            continue

        knee_measurements.append(
            DogKeypoint(
                x=knee_angle,
                y=0.0,
                confidence=minimum_confidence,
            )
        )
        hock_measurements.append(
            DogKeypoint(
                x=hock_angle,
                y=0.0,
                confidence=minimum_confidence,
            )
        )
        fully_observed.append(True)

    return (
        knee_measurements,
        hock_measurements,
        fully_observed,
        minimum_confidences,
    )


def allowed_estimated_frames(
    observed: list[bool],
    *,
    max_gap: int,
) -> list[bool]:
    """Allow only short internal gaps bounded by real observations."""

    allowed = observed.copy()
    index = 0

    while index < len(observed):
        if observed[index]:
            index += 1
            continue

        start = index
        while (
            index < len(observed)
            and not observed[index]
        ):
            index += 1
        end = index

        gap_length = end - start
        bounded_left = start > 0 and observed[start - 1]
        bounded_right = (
            end < len(observed)
            and observed[end]
        )

        if (
            gap_length <= max_gap
            and bounded_left
            and bounded_right
        ):
            for gap_index in range(start, end):
                allowed[gap_index] = True

    return allowed


def build_rows(
    trajectories: dict[
        str,
        list[DogKeypoint | None],
    ],
    *,
    fps: float,
    min_confidence: float,
    max_estimated_gap: int,
) -> list[dict[str, object]]:
    (
        knee_measurements,
        hock_measurements,
        observed_flags,
        minimum_confidences,
    ) = build_angle_measurements(
        trajectories,
        min_confidence=min_confidence,
    )

    smoothed_knee = smooth_keypoint_trajectory(
        knee_measurements,
        min_confidence=min_confidence,
        process_variance=25.0,
        measurement_variance=4.0,
        outlier_mahalanobis_squared=100.0,
    )
    smoothed_hock = smooth_keypoint_trajectory(
        hock_measurements,
        min_confidence=min_confidence,
        process_variance=25.0,
        measurement_variance=4.0,
        outlier_mahalanobis_squared=100.0,
    )

    estimated_allowed = allowed_estimated_frames(
        observed_flags,
        max_gap=max_estimated_gap,
    )

    rows: list[dict[str, object]] = []

    for frame_index in range(len(observed_flags)):
        knee_raw = knee_measurements[frame_index]
        hock_raw = hock_measurements[frame_index]
        knee_smooth = smoothed_knee[frame_index]
        hock_smooth = smoothed_hock[frame_index]

        angle_available = (
            estimated_allowed[frame_index]
            and knee_smooth is not None
            and hock_smooth is not None
        )

        row: dict[str, object] = {
            "frame_index": frame_index,
            "time_seconds": frame_index / fps,
            "knee_angle_raw_degrees": (
                None if knee_raw is None else knee_raw.x
            ),
            "hock_angle_raw_degrees": (
                None if hock_raw is None else hock_raw.x
            ),
            "knee_angle_degrees": (
                None
                if not angle_available
                else max(
                    0.0,
                    min(180.0, knee_smooth.x),
                )
            ),
            "hock_angle_degrees": (
                None
                if not angle_available
                else max(
                    0.0,
                    min(180.0, hock_smooth.x),
                )
            ),
            "fully_observed": observed_flags[
                frame_index
            ],
            "temporally_estimated": (
                angle_available
                and not observed_flags[frame_index]
            ),
            "minimum_source_confidence": (
                minimum_confidences[frame_index]
            ),
        }

        for bodypart in BODY_PARTS:
            point = trajectories[bodypart][frame_index]

            row[f"{bodypart}_raw_x"] = (
                None if point is None else point.x
            )
            row[f"{bodypart}_raw_y"] = (
                None if point is None else point.y
            )
            row[f"{bodypart}_confidence"] = (
                0.0 if point is None else point.confidence
            )

        rows.append(row)

    return rows


def save_plot(
    path: Path,
    rows: list[dict[str, object]],
) -> None:
    times = np.array(
        [float(row["time_seconds"]) for row in rows]
    )

    knee = np.array(
        [
            (
                float(row["knee_angle_degrees"])
                if (
                    bool(row["fully_observed"])
                    and row["knee_angle_degrees"] is not None
                )
                else np.nan
            )
            for row in rows
        ]
    )
    hock = np.array(
        [
            (
                float(row["hock_angle_degrees"])
                if (
                    bool(row["fully_observed"])
                    and row["hock_angle_degrees"] is not None
                )
                else np.nan
            )
            for row in rows
        ]
    )

    plt.figure(figsize=(13, 6))
    plt.plot(
        times,
        knee,
        color="royalblue",
        linewidth=1.8,
        label="Knee: thigh–knee–hock",
    )
    plt.plot(
        times,
        hock,
        color="darkorange",
        linewidth=1.8,
        label="Hock: knee–hock–paw",
    )

    plt.xlabel("Time (seconds)")
    plt.ylabel("Internal 2D joint angle (degrees)")
    plt.title(
        "Right hind-limb knee and hock kinematics"
    )
    plt.ylim(0.0, 180.0)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def print_statistics(
    label: str,
    values: list[float],
) -> None:
    print(label)
    print(f"  Minimum:      {min(values):.2f}°")
    print(f"  Maximum:      {max(values):.2f}°")
    print(f"  Mean:         {mean(values):.2f}°")
    print(
        f"  Motion range: "
        f"{max(values) - min(values):.2f}°"
    )


def main() -> None:
    args = parse_args()

    if not 0.0 <= args.min_confidence <= 1.0:
        raise SystemExit(
            "--min-confidence must be between 0 and 1."
        )
    if args.max_estimated_gap < 0:
        raise SystemExit(
            "--max-estimated-gap cannot be negative."
        )

    metadata = read_video_metadata(args.video)
    _, trajectories = load_raw_keypoints(
        args.pose_h5
    )

    rows = build_rows(
        trajectories,
        fps=metadata.fps,
        min_confidence=args.min_confidence,
        max_estimated_gap=args.max_estimated_gap,
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    raw_path = (
        args.output_dir
        / "hind_limb_kinematics_raw.csv"
    )
    clean_path = (
        args.output_dir
        / "hind_limb_kinematics_clean.csv"
    )
    plot_path = (
        args.output_dir
        / "knee_hock_angle_plot.png"
    )

    raw_data = pd.DataFrame(rows)
    clean_data = raw_data[
        raw_data["fully_observed"]
    ].copy()

    raw_data.to_csv(raw_path, index=False)
    clean_data.to_csv(clean_path, index=False)
    save_plot(plot_path, rows)

    knee_values = (
        clean_data["knee_angle_degrees"]
        .dropna()
        .astype(float)
        .tolist()
    )
    hock_values = (
        clean_data["hock_angle_degrees"]
        .dropna()
        .astype(float)
        .tolist()
    )

    estimated_count = int(
        raw_data["temporally_estimated"].sum()
    )

    print("Four-keypoint gait analysis")
    print("-" * 50)
    print(f"Video frames:             {metadata.frame_count}")
    print(f"Pose frames:              {len(raw_data)}")
    print(f"Fully observed:           {len(clean_data)}")
    print(f"Short-gap estimates:      {estimated_count}")
    print(
        f"Rejected/incomplete:      "
        f"{len(raw_data) - len(clean_data)}"
    )
    print()
    print_statistics("Knee angle", knee_values)
    print()
    print_statistics("Hock angle", hock_values)
    print()
    print(f"Raw CSV:   {raw_path}")
    print(f"Clean CSV: {clean_path}")
    print(f"Plot:      {plot_path}")


if __name__ == "__main__":
    main()