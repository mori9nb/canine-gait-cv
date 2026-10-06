from __future__ import annotations

import argparse
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from canine_gait_cv.gait import (
    GaitEventType,
    GaitPhase,
    GaitPhaseResult,
    StrideInterval,
    detect_hind_paw_gait_phases,
)


REQUIRED_COLUMNS = {
    "frame_index",
    "time_seconds",
    "fully_observed",
    "knee_angle_degrees",
    "hock_angle_degrees",
    "back_right_thai_raw_x",
    "back_right_thai_raw_y",
    "back_right_paw_raw_x",
    "back_right_paw_raw_y",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Detect right hind-limb gait phases and "
            "export per-stride measurements."
        )
    )

    parser.add_argument(
        "kinematics_csv",
        type=Path,
        help=(
            "Raw kinematics CSV produced by "
            "export_hock4_gait_analysis.py."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "outputs/gait/single_dog_hock4/stride"
        ),
    )

    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
    )

    parser.add_argument(
        "--forward-direction",
        choices=("left", "right"),
        default="right",
    )

    parser.add_argument(
        "--max-interpolated-gap",
        type=int,
        default=3,
    )

    return parser.parse_args()


def load_kinematics_csv(
    path: Path,
) -> pd.DataFrame:
    data = pd.read_csv(path)

    missing = sorted(
        REQUIRED_COLUMNS - set(data.columns)
    )

    if missing:
        raise ValueError(
            "Missing CSV columns: "
            + ", ".join(missing)
        )

    if data.empty:
        raise ValueError(
            "Kinematics CSV contains no rows."
        )

    data = data.sort_values(
        "frame_index"
    ).reset_index(drop=True)

    frame_indices = (
        data["frame_index"]
        .astype(int)
        .tolist()
    )

    if any(
        current <= previous
        for previous, current in zip(
            frame_indices,
            frame_indices[1:],
        )
    ):
        raise ValueError(
            "frame_index must be strictly increasing."
        )

    return data


def as_boolean_series(
    values: pd.Series,
) -> pd.Series:
    if values.dtype == bool:
        return values

    normalized = (
        values
        .astype(str)
        .str.strip()
        .str.lower()
    )

    return normalized.map(
        {
            "true": True,
            "false": False,
            "1": True,
            "0": False,
        }
    ).fillna(False)


def detect_phases(
    data: pd.DataFrame,
    *,
    fps: float,
    forward_direction: str,
    max_interpolated_gap: int,
) -> GaitPhaseResult:
    valid = as_boolean_series(
        data["fully_observed"]
    )

    thigh_x = data[
        "back_right_thai_raw_x"
    ].where(valid).tolist()

    thigh_y = data[
        "back_right_thai_raw_y"
    ].where(valid).tolist()

    paw_x = data[
        "back_right_paw_raw_x"
    ].where(valid).tolist()

    paw_y = data[
        "back_right_paw_raw_y"
    ].where(valid).tolist()

    return detect_hind_paw_gait_phases(
        frame_indices=(
            data["frame_index"]
            .astype(int)
            .tolist()
        ),
        thigh_x=thigh_x,
        thigh_y=thigh_y,
        paw_x=paw_x,
        paw_y=paw_y,
        fps=fps,
        forward_direction=forward_direction,
        max_interpolated_gap=(
            max_interpolated_gap
        ),
    )


def find_stride_index(
    frame_index: int,
    strides: tuple[StrideInterval, ...],
) -> int | None:
    for stride in strides:
        if (
            stride.contact_frame
            <= frame_index
            < stride.next_contact_frame
        ):
            return stride.stride_index

    return None


def build_phase_rows(
    data: pd.DataFrame,
    result: GaitPhaseResult,
) -> list[dict[str, object]]:
    source_by_frame = {
        int(row["frame_index"]): row
        for _, row in data.iterrows()
    }

    rows: list[dict[str, object]] = []

    for phase_frame in result.frames:
        source = source_by_frame[
            phase_frame.frame_index
        ]

        rows.append(
            {
                "frame_index": (
                    phase_frame.frame_index
                ),
                "time_seconds": (
                    phase_frame.time_seconds
                ),
                "phase": (
                    phase_frame.phase.value
                ),
                "stride_index": (
                    find_stride_index(
                        phase_frame.frame_index,
                        result.strides,
                    )
                ),
                "relative_paw_x": (
                    phase_frame.relative_paw_x
                ),
                "relative_paw_y": (
                    phase_frame.relative_paw_y
                ),
                "forward_velocity": (
                    phase_frame.forward_velocity
                ),
                "fully_observed": bool(
                    source["fully_observed"]
                ),
                "minimum_source_confidence": (
                    source.get(
                        "minimum_source_confidence",
                        np.nan,
                    )
                ),
                "knee_angle_degrees": (
                    source["knee_angle_degrees"]
                ),
                "hock_angle_degrees": (
                    source["hock_angle_degrees"]
                ),
            }
        )

    return rows


def build_event_rows(
    result: GaitPhaseResult,
) -> list[dict[str, object]]:
    return [
        {
            "event_index": index,
            "frame_index": event.frame_index,
            "time_seconds": event.time_seconds,
            "event_type": event.event_type.value,
        }
        for index, event in enumerate(
            result.events
        )
    ]


def angle_statistics(
    values: pd.Series,
) -> dict[str, float | None]:
    clean = (
        pd.to_numeric(
            values,
            errors="coerce",
        )
        .dropna()
        .to_numpy(dtype=float)
    )

    if len(clean) == 0:
        return {
            "minimum": None,
            "maximum": None,
            "mean": None,
            "range": None,
        }

    minimum = float(np.min(clean))
    maximum = float(np.max(clean))

    return {
        "minimum": minimum,
        "maximum": maximum,
        "mean": float(np.mean(clean)),
        "range": maximum - minimum,
    }


def peak_angular_velocity(
    times: pd.Series,
    angles: pd.Series,
) -> float | None:
    values = pd.DataFrame(
        {
            "time": pd.to_numeric(
                times,
                errors="coerce",
            ),
            "angle": pd.to_numeric(
                angles,
                errors="coerce",
            ),
        }
    ).dropna()

    if len(values) < 2:
        return None

    time_values = values[
        "time"
    ].to_numpy(dtype=float)

    angle_values = values[
        "angle"
    ].to_numpy(dtype=float)

    delta_time = np.diff(time_values)
    delta_angle = np.diff(angle_values)

    valid = delta_time > 0.0

    if not valid.any():
        return None

    velocity = (
        delta_angle[valid]
        / delta_time[valid]
    )

    return float(
        np.max(np.abs(velocity))
    )


def build_stride_rows(
    data: pd.DataFrame,
    result: GaitPhaseResult,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    for stride in result.strides:
        subset = data[
            (
                data["frame_index"]
                >= stride.contact_frame
            )
            & (
                data["frame_index"]
                < stride.next_contact_frame
            )
        ].copy()

        knee = angle_statistics(
            subset["knee_angle_degrees"]
        )
        hock = angle_statistics(
            subset["hock_angle_degrees"]
        )

        valid_frames = int(
            subset[
                "knee_angle_degrees"
            ].notna().sum()
        )

        rows.append(
            {
                "stride_index": (
                    stride.stride_index
                ),
                "contact_frame": (
                    stride.contact_frame
                ),
                "toe_off_frame": (
                    stride.toe_off_frame
                ),
                "next_contact_frame": (
                    stride.next_contact_frame
                ),
                "stride_duration_seconds": (
                    stride.stride_duration_seconds
                ),
                "stance_duration_seconds": (
                    stride.stance_duration_seconds
                ),
                "swing_duration_seconds": (
                    stride.swing_duration_seconds
                ),
                "stance_percent": (
                    100.0
                    * stride.stance_fraction
                ),
                "swing_percent": (
                    100.0
                    * stride.swing_fraction
                ),
                "valid_angle_frames": (
                    valid_frames
                ),
                "knee_minimum_degrees": (
                    knee["minimum"]
                ),
                "knee_maximum_degrees": (
                    knee["maximum"]
                ),
                "knee_mean_degrees": (
                    knee["mean"]
                ),
                "knee_range_degrees": (
                    knee["range"]
                ),
                "knee_peak_angular_velocity_deg_s": (
                    peak_angular_velocity(
                        subset["time_seconds"],
                        subset[
                            "knee_angle_degrees"
                        ],
                    )
                ),
                "hock_minimum_degrees": (
                    hock["minimum"]
                ),
                "hock_maximum_degrees": (
                    hock["maximum"]
                ),
                "hock_mean_degrees": (
                    hock["mean"]
                ),
                "hock_range_degrees": (
                    hock["range"]
                ),
                "hock_peak_angular_velocity_deg_s": (
                    peak_angular_velocity(
                        subset["time_seconds"],
                        subset[
                            "hock_angle_degrees"
                        ],
                    )
                ),
            }
        )

    return rows


def phase_runs(
    result: GaitPhaseResult,
) -> list[
    tuple[GaitPhase, float, float]
]:
    if not result.frames:
        return []

    runs: list[
        tuple[GaitPhase, float, float]
    ] = []

    start = result.frames[0]
    previous = start

    for current in result.frames[1:]:
        if current.phase is previous.phase:
            previous = current
            continue

        runs.append(
            (
                start.phase,
                start.time_seconds,
                current.time_seconds,
            )
        )

        start = current
        previous = current

    frame_step = (
        result.frames[-1].time_seconds
        - result.frames[-2].time_seconds
        if len(result.frames) > 1
        else 0.0
    )

    runs.append(
        (
            start.phase,
            start.time_seconds,
            (
                result.frames[-1].time_seconds
                + frame_step
            ),
        )
    )

    return runs


def save_plot(
    path: Path,
    data: pd.DataFrame,
    result: GaitPhaseResult,
) -> None:
    figure, axes = plt.subplots(
        2,
        1,
        figsize=(14, 9),
        sharex=True,
    )

    angle_axis = axes[0]
    velocity_axis = axes[1]

    for phase, start, end in phase_runs(result):
        if phase is GaitPhase.STANCE:
            color = "tab:orange"
            alpha = 0.18
        elif phase is GaitPhase.SWING:
            color = "tab:cyan"
            alpha = 0.18
        else:
            color = "gray"
            alpha = 0.08

        for axis in axes:
            axis.axvspan(
                start,
                end,
                color=color,
                alpha=alpha,
            )

    angle_axis.plot(
        data["time_seconds"],
        data["knee_angle_degrees"],
        color="tab:blue",
        linewidth=1.8,
        label="Knee angle",
    )

    angle_axis.plot(
        data["time_seconds"],
        data["hock_angle_degrees"],
        color="tab:red",
        linewidth=1.8,
        label="Hock angle",
    )

    velocity_axis.plot(
        [
            frame.time_seconds
            for frame in result.frames
        ],
        [
            frame.forward_velocity
            for frame in result.frames
        ],
        color="tab:purple",
        linewidth=1.8,
        label="Relative paw velocity",
    )

    velocity_axis.axhline(
        0.0,
        color="black",
        linewidth=1.0,
    )

    contact_label_used = False
    toe_off_label_used = False

    for event in result.events:
        if (
            event.event_type
            is GaitEventType.CONTACT
        ):
            color = "darkgreen"
            label = (
                None
                if contact_label_used
                else "Contact"
            )
            contact_label_used = True
        else:
            color = "darkred"
            label = (
                None
                if toe_off_label_used
                else "Toe-off"
            )
            toe_off_label_used = True

        for axis in axes:
            axis.axvline(
                event.time_seconds,
                color=color,
                linestyle="--",
                linewidth=1.0,
                alpha=0.8,
                label=label,
            )

        # Only add the legend label to the first axis once.

    angle_axis.set_ylabel(
        "Joint angle (degrees)"
    )
    angle_axis.set_title(
        "Right Hind-Limb Gait Phases"
    )
    angle_axis.grid(alpha=0.25)
    angle_axis.legend(
        loc="upper right"
    )

    velocity_axis.set_xlabel(
        "Time (seconds)"
    )
    velocity_axis.set_ylabel(
        "Forward paw velocity\n(body lengths/s)"
    )
    velocity_axis.grid(alpha=0.25)

    figure.tight_layout()
    figure.savefig(
        path,
        dpi=180,
    )
    plt.close(figure)


def print_summary(
    result: GaitPhaseResult,
    stride_rows: list[dict[str, object]],
) -> None:
    print("Stride analysis")
    print("-" * 60)
    print(
        f"Frames:             "
        f"{len(result.frames)}"
    )
    print(
        f"Detected events:    "
        f"{len(result.events)}"
    )
    print(
        f"Complete strides:   "
        f"{len(result.strides)}"
    )
    print(
        f"Body scale:         "
        f"{result.body_scale_pixels:.2f} px"
    )
    print(
        f"Forward direction:  "
        f"{result.forward_direction}"
    )

    if not stride_rows:
        return

    print()
    print(
        "Mean stride duration: "
        f"{mean(float(row['stride_duration_seconds']) for row in stride_rows):.3f} s"
    )
    print(
        "Mean stance:          "
        f"{mean(float(row['stance_percent']) for row in stride_rows):.1f}%"
    )
    print(
        "Mean swing:           "
        f"{mean(float(row['swing_percent']) for row in stride_rows):.1f}%"
    )

    print("\nComplete strides")

    for row in stride_rows:
        print(
            f"  Stride {row['stride_index']}: "
            f"f{row['contact_frame']} -> "
            f"f{row['next_contact_frame']}, "
            f"toe-off=f{row['toe_off_frame']}, "
            f"duration="
            f"{float(row['stride_duration_seconds']):.3f} s, "
            f"stance="
            f"{float(row['stance_percent']):.1f}%, "
            f"swing="
            f"{float(row['swing_percent']):.1f}%"
        )


def main() -> None:
    args = parse_args()

    if args.fps <= 0.0:
        raise ValueError(
            "fps must be positive."
        )

    data = load_kinematics_csv(
        args.kinematics_csv
    )

    result = detect_phases(
        data,
        fps=args.fps,
        forward_direction=(
            args.forward_direction
        ),
        max_interpolated_gap=(
            args.max_interpolated_gap
        ),
    )

    phase_rows = build_phase_rows(
        data,
        result,
    )
    event_rows = build_event_rows(
        result
    )
    stride_rows = build_stride_rows(
        data,
        result,
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    phase_path = (
        args.output_dir
        / "gait_phase_frames.csv"
    )
    event_path = (
        args.output_dir
        / "gait_events.csv"
    )
    stride_path = (
        args.output_dir
        / "stride_summary.csv"
    )
    plot_path = (
        args.output_dir
        / "gait_phase_plot.png"
    )

    pd.DataFrame(
        phase_rows
    ).to_csv(
        phase_path,
        index=False,
    )

    pd.DataFrame(
        event_rows
    ).to_csv(
        event_path,
        index=False,
    )

    pd.DataFrame(
        stride_rows
    ).to_csv(
        stride_path,
        index=False,
    )

    save_plot(
        plot_path,
        data,
        result,
    )

    print_summary(
        result,
        stride_rows,
    )

    print()
    print(f"Phase CSV:  {phase_path}")
    print(f"Events CSV: {event_path}")
    print(f"Stride CSV: {stride_path}")
    print(f"Plot:       {plot_path}")


if __name__ == "__main__":
    main()
