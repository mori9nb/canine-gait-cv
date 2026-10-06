from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from canine_gait_cv.gait import (
    StrideInterval,
    build_mean_reference_trajectory,
    normalize_stride_profiles,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Normalize canine gait strides and build "
            "a mean prosthesis reference trajectory."
        )
    )

    parser.add_argument(
        "kinematics_csv",
        type=Path,
    )
    parser.add_argument(
        "stride_summary_csv",
        type=Path,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "outputs/gait/single_dog_hock4/"
            "normalized"
        ),
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=101,
    )
    parser.add_argument(
        "--min-source-coverage",
        type=float,
        default=0.80,
    )

    return parser.parse_args()


def load_strides(
    path: Path,
) -> list[StrideInterval]:
    data = pd.read_csv(path)

    required = {
        "stride_index",
        "contact_frame",
        "toe_off_frame",
        "next_contact_frame",
        "stride_duration_seconds",
        "stance_duration_seconds",
        "swing_duration_seconds",
        "stance_percent",
        "swing_percent",
    }

    missing = sorted(
        required - set(data.columns)
    )

    if missing:
        raise ValueError(
            "Missing stride columns: "
            + ", ".join(missing)
        )

    strides: list[StrideInterval] = []

    for _, row in data.iterrows():
        strides.append(
            StrideInterval(
                stride_index=int(
                    row["stride_index"]
                ),
                contact_frame=int(
                    row["contact_frame"]
                ),
                toe_off_frame=int(
                    row["toe_off_frame"]
                ),
                next_contact_frame=int(
                    row["next_contact_frame"]
                ),
                stride_duration_seconds=float(
                    row[
                        "stride_duration_seconds"
                    ]
                ),
                stance_duration_seconds=float(
                    row[
                        "stance_duration_seconds"
                    ]
                ),
                swing_duration_seconds=float(
                    row[
                        "swing_duration_seconds"
                    ]
                ),
                stance_fraction=float(
                    row["stance_percent"]
                ) / 100.0,
                swing_fraction=float(
                    row["swing_percent"]
                ) / 100.0,
            )
        )

    return strides


def phase_name(
    cycle_percent: float,
    *,
    toe_off_percent: float,
) -> str:
    if np.isclose(
        cycle_percent,
        0.0,
    ) or np.isclose(
        cycle_percent,
        100.0,
    ):
        return "contact"

    if np.isclose(
        cycle_percent,
        toe_off_percent,
        atol=0.5,
    ):
        return "toe_off"

    if cycle_percent < toe_off_percent:
        return "stance"

    return "swing"


def build_profile_rows(
    profiles,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    for profile in profiles:
        for (
            cycle_percent,
            knee_angle,
            hock_angle,
        ) in zip(
            profile.cycle_percent,
            profile.knee_angle_degrees,
            profile.hock_angle_degrees,
        ):
            rows.append(
                {
                    "stride_index": (
                        profile.stride_index
                    ),
                    "cycle_percent": (
                        cycle_percent
                    ),
                    "normalized_phase": (
                        cycle_percent / 100.0
                    ),
                    "time_seconds": (
                        cycle_percent
                        / 100.0
                        * profile.stride_duration_seconds
                    ),
                    "gait_phase": phase_name(
                        cycle_percent,
                        toe_off_percent=(
                            profile.toe_off_percent
                        ),
                    ),
                    "knee_angle_degrees": (
                        knee_angle
                    ),
                    "hock_angle_degrees": (
                        hock_angle
                    ),
                    "toe_off_percent": (
                        profile.toe_off_percent
                    ),
                    "stride_duration_seconds": (
                        profile.stride_duration_seconds
                    ),
                    "source_coverage": (
                        profile.source_coverage
                    ),
                }
            )

    return rows


def build_reference_rows(
    reference,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    for index, cycle_percent in enumerate(
        reference.cycle_percent
    ):
        knee_mean = (
            reference.knee_mean_degrees[
                index
            ]
        )
        knee_std = (
            reference.knee_std_degrees[
                index
            ]
        )
        hock_mean = (
            reference.hock_mean_degrees[
                index
            ]
        )
        hock_std = (
            reference.hock_std_degrees[
                index
            ]
        )

        rows.append(
            {
                "cycle_percent": (
                    cycle_percent
                ),
                "normalized_phase": (
                    cycle_percent / 100.0
                ),
                "mean_time_seconds": (
                    cycle_percent
                    / 100.0
                    * reference.mean_stride_duration_seconds
                ),
                "gait_phase": phase_name(
                    cycle_percent,
                    toe_off_percent=(
                        reference.mean_toe_off_percent
                    ),
                ),
                "knee_reference_degrees": (
                    knee_mean
                ),
                "knee_std_degrees": (
                    knee_std
                ),
                "knee_lower_degrees": (
                    knee_mean - knee_std
                ),
                "knee_upper_degrees": (
                    knee_mean + knee_std
                ),
                "knee_velocity_deg_s": (
                    reference
                    .knee_velocity_degrees_per_second[
                        index
                    ]
                ),
                "hock_reference_degrees": (
                    hock_mean
                ),
                "hock_std_degrees": (
                    hock_std
                ),
                "hock_lower_degrees": (
                    hock_mean - hock_std
                ),
                "hock_upper_degrees": (
                    hock_mean + hock_std
                ),
                "hock_velocity_deg_s": (
                    reference
                    .hock_velocity_degrees_per_second[
                        index
                    ]
                ),
                "mean_toe_off_percent": (
                    reference.mean_toe_off_percent
                ),
                "mean_stride_duration_seconds": (
                    reference.mean_stride_duration_seconds
                ),
                "source_stride_count": (
                    reference.stride_count
                ),
            }
        )

    return rows


def save_plot(
    path: Path,
    *,
    profiles,
    reference,
) -> None:
    figure, axes = plt.subplots(
        2,
        1,
        figsize=(13, 9),
        sharex=True,
    )

    knee_axis = axes[0]
    hock_axis = axes[1]

    toe_off = (
        reference.mean_toe_off_percent
    )

    for axis in axes:
        axis.axvspan(
            0.0,
            toe_off,
            color="tab:orange",
            alpha=0.12,
            label="Mean stance",
        )
        axis.axvspan(
            toe_off,
            100.0,
            color="tab:cyan",
            alpha=0.12,
            label="Mean swing",
        )
        axis.axvline(
            toe_off,
            color="darkred",
            linestyle="--",
            linewidth=1.5,
            label="Mean toe-off",
        )
        axis.grid(alpha=0.25)

    for profile in profiles:
        knee_axis.plot(
            profile.cycle_percent,
            profile.knee_angle_degrees,
            color="tab:blue",
            alpha=0.23,
            linewidth=1.1,
        )

        hock_axis.plot(
            profile.cycle_percent,
            profile.hock_angle_degrees,
            color="tab:red",
            alpha=0.23,
            linewidth=1.1,
        )

    cycle = np.asarray(
        reference.cycle_percent,
        dtype=float,
    )

    knee_mean = np.asarray(
        reference.knee_mean_degrees,
        dtype=float,
    )
    knee_std = np.asarray(
        reference.knee_std_degrees,
        dtype=float,
    )

    hock_mean = np.asarray(
        reference.hock_mean_degrees,
        dtype=float,
    )
    hock_std = np.asarray(
        reference.hock_std_degrees,
        dtype=float,
    )

    knee_axis.fill_between(
        cycle,
        knee_mean - knee_std,
        knee_mean + knee_std,
        color="tab:blue",
        alpha=0.18,
        label="Knee ±1 SD",
    )

    knee_axis.plot(
        cycle,
        knee_mean,
        color="tab:blue",
        linewidth=3.0,
        label="Mean knee",
    )

    hock_axis.fill_between(
        cycle,
        hock_mean - hock_std,
        hock_mean + hock_std,
        color="tab:red",
        alpha=0.18,
        label="Hock ±1 SD",
    )

    hock_axis.plot(
        cycle,
        hock_mean,
        color="tab:red",
        linewidth=3.0,
        label="Mean hock",
    )

    knee_axis.set_title(
        "Normalized Right Hind-Limb Reference Trajectory"
    )
    knee_axis.set_ylabel(
        "Knee angle (degrees)"
    )
    hock_axis.set_ylabel(
        "Hock angle (degrees)"
    )
    hock_axis.set_xlabel(
        "Normalized gait cycle (%)"
    )

    knee_axis.legend(
        loc="upper right",
        ncol=2,
    )
    hock_axis.legend(
        loc="upper right",
        ncol=2,
    )

    figure.tight_layout()

    figure.savefig(
        path,
        dpi=180,
    )

    plt.close(figure)


def main() -> None:
    args = parse_args()

    kinematics = pd.read_csv(
        args.kinematics_csv
    )

    required = {
        "frame_index",
        "knee_angle_degrees",
        "hock_angle_degrees",
    }

    missing = sorted(
        required
        - set(kinematics.columns)
    )

    if missing:
        raise ValueError(
            "Missing kinematics columns: "
            + ", ".join(missing)
        )

    strides = load_strides(
        args.stride_summary_csv
    )

    profiles = normalize_stride_profiles(
        frame_indices=(
            kinematics["frame_index"]
            .astype(int)
            .tolist()
        ),
        knee_angles=(
            kinematics[
                "knee_angle_degrees"
            ].tolist()
        ),
        hock_angles=(
            kinematics[
                "hock_angle_degrees"
            ].tolist()
        ),
        strides=strides,
        sample_count=args.sample_count,
        min_source_coverage=(
            args.min_source_coverage
        ),
    )

    if not profiles:
        raise RuntimeError(
            "No stride passed normalization quality checks."
        )

    reference = (
        build_mean_reference_trajectory(
            profiles
        )
    )

    profile_rows = build_profile_rows(
        profiles
    )
    reference_rows = build_reference_rows(
        reference
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    profiles_path = (
        args.output_dir
        / "normalized_stride_profiles.csv"
    )
    reference_path = (
        args.output_dir
        / "mean_reference_trajectory.csv"
    )
    plot_path = (
        args.output_dir
        / "normalized_knee_hock_plot.png"
    )

    pd.DataFrame(
        profile_rows
    ).to_csv(
        profiles_path,
        index=False,
    )

    pd.DataFrame(
        reference_rows
    ).to_csv(
        reference_path,
        index=False,
    )

    save_plot(
        plot_path,
        profiles=profiles,
        reference=reference,
    )

    print(
        "Normalized gait reference exported"
    )
    print("-" * 60)
    print(
        f"Source strides:       "
        f"{len(strides)}"
    )
    print(
        f"Accepted strides:     "
        f"{len(profiles)}"
    )
    print(
        f"Samples per stride:   "
        f"{args.sample_count}"
    )
    print(
        f"Mean stride duration: "
        f"{reference.mean_stride_duration_seconds:.3f} s"
    )
    print(
        f"Mean toe-off:         "
        f"{reference.mean_toe_off_percent:.2f}%"
    )

    print("\nAccepted stride coverage")

    for profile in profiles:
        print(
            f"  Stride "
            f"{profile.stride_index}: "
            f"{100 * profile.source_coverage:.1f}%"
        )

    print()
    print(
        f"Profiles CSV:  {profiles_path}"
    )
    print(
        f"Reference CSV: {reference_path}"
    )
    print(
        f"Plot:          {plot_path}"
    )


if __name__ == "__main__":
    main()
