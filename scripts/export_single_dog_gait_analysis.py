from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt

from canine_gait_cv.gait import build_hind_limb_angle_series
from canine_gait_cv.pose import load_superanimal_multi_json
from canine_gait_cv.video.reader import read_video_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export single-dog hind-limb angle analysis."
    )
    parser.add_argument("video", type=Path)
    parser.add_argument("pose_json", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/gait/single_dog_side"),
    )
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


def write_csv(path: Path, rows, *, clean_only: bool) -> int:
    written = 0

    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "frame_index",
                "time_seconds",
                "angle_degrees",
                "fully_observed",
                "minimum_source_confidence",
            ]
        )

        for row in rows:
            if row.angle_degrees is None:
                continue

            if clean_only and not row.fully_observed:
                continue

            writer.writerow(
                [
                    row.frame_index,
                    f"{row.time_seconds:.6f}",
                    f"{row.angle_degrees:.6f}",
                    row.fully_observed,
                    f"{row.minimum_source_confidence:.6f}",
                ]
            )
            written += 1

    return written


def save_plot(path: Path, rows, side: str) -> None:
    usable = [
        row for row in rows
        if row.angle_degrees is not None
    ]

    observed = [
        row for row in usable
        if row.fully_observed
    ]
    estimated = [
        row for row in usable
        if not row.fully_observed
    ]

    plt.figure(figsize=(12, 5))

    plt.plot(
        [row.time_seconds for row in usable],
        [row.angle_degrees for row in usable],
        color="royalblue",
        linewidth=1.6,
        label="Kalman–RTS smoothed angle",
    )

    plt.scatter(
        [row.time_seconds for row in observed],
        [row.angle_degrees for row in observed],
        color="green",
        s=12,
        label="Fully observed",
        zorder=3,
    )

    if estimated:
        plt.scatter(
            [row.time_seconds for row in estimated],
            [row.angle_degrees for row in estimated],
            color="orange",
            marker="x",
            s=25,
            label="Estimated / incomplete",
            zorder=4,
        )

    plt.xlabel("Time (seconds)")
    plt.ylabel("2D angle (degrees)")
    plt.title(
        f"Smoothed {side} hind-limb angle "
        "(hip–knee–paw)"
    )
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def main() -> None:
    args = parse_args()

    if not 0.0 <= args.min_confidence <= 1.0:
        raise SystemExit("--min-confidence must be between 0 and 1.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    metadata = read_video_metadata(args.video)
    pose_frames = load_superanimal_multi_json(args.pose_json)

    angle_rows = build_hind_limb_angle_series(
        pose_frames,
        fps=metadata.fps,
        side=args.side,
        min_confidence=args.min_confidence,
    )

    raw_csv = args.output_dir / "hind_limb_angles_raw.csv"
    clean_csv = args.output_dir / "hind_limb_angles_clean.csv"
    plot_path = args.output_dir / "hind_limb_angle_plot.png"

    raw_count = write_csv(
        raw_csv,
        angle_rows,
        clean_only=False,
    )
    clean_count = write_csv(
        clean_csv,
        angle_rows,
        clean_only=True,
    )

    save_plot(plot_path, angle_rows, args.side)

    angles = [
        row.angle_degrees
        for row in angle_rows
        if row.angle_degrees is not None
    ]
    observed_count = sum(
        row.angle_degrees is not None and row.fully_observed
        for row in angle_rows
    )
    estimated_count = sum(
        row.angle_degrees is not None and not row.fully_observed
        for row in angle_rows
    )

    print("Single-dog gait analysis")
    print("-" * 50)
    print(f"Video frames:             {metadata.frame_count}")
    print(f"Pose frames:              {len(pose_frames)}")
    print(f"Angle available:          {len(angles)}")
    print(f"Fully observed:           {observed_count}")
    print(f"Estimated or incomplete:  {estimated_count}")
    print(f"Raw CSV rows:             {raw_count}")
    print(f"Clean CSV rows:           {clean_count}")

    if angles:
        print(f"Minimum angle:            {min(angles):.2f}°")
        print(f"Maximum angle:            {max(angles):.2f}°")
        print(f"Mean angle:               {mean(angles):.2f}°")
        print(f"Motion range:             {max(angles) - min(angles):.2f}°")

    print()
    print(f"Raw CSV:   {raw_csv}")
    print(f"Clean CSV: {clean_csv}")
    print(f"Plot:      {plot_path}")


if __name__ == "__main__":
    main()