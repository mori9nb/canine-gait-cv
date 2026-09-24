from __future__ import annotations

import argparse
from pathlib import Path

from canine_gait_cv.video.reader import read_video_metadata


SUPPORTED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect video metadata."
    )
    parser.add_argument(
        "video",
        nargs="?",
        type=Path,
        help="Video path. If omitted, the first video in data/raw is used.",
    )
    return parser.parse_args()


def find_default_video() -> Path:
    raw_dir = Path("data/raw")
    videos = sorted(
        path
        for path in raw_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if not videos:
        raise SystemExit("No video file found in data/raw")

    return videos[0]


def main() -> None:
    args = parse_args()
    video_path = args.video if args.video is not None else find_default_video()

    if not video_path.exists():
        raise SystemExit(f"Video not found: {video_path}")

    metadata = read_video_metadata(video_path)

    print("Video metadata")
    print("-" * 40)
    print(f"Path:      {metadata.path}")
    print(f"FPS:       {metadata.fps}")
    print(f"Frames:    {metadata.frame_count}")
    print(f"Width:     {metadata.width}")
    print(f"Height:    {metadata.height}")
    print(f"Duration:  {metadata.duration_seconds:.2f} seconds")


if __name__ == "__main__":
    main()