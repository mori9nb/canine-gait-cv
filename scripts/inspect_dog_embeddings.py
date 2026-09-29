from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from PIL import Image

from canine_gait_cv.appearance import MegaDescriptorExtractor
from canine_gait_cv.pose import load_superanimal_multi_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect MegaDescriptor similarities between dog detections."
    )
    parser.add_argument("video", type=Path)
    parser.add_argument("predictions", type=Path)
    parser.add_argument(
        "--frames",
        type=int,
        nargs="+",
        default=[52, 68, 109],
    )
    return parser.parse_args()


def read_frame(
    capture: cv2.VideoCapture,
    frame_index: int,
) -> Image.Image:
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    success, frame = capture.read()

    if not success:
        raise RuntimeError(f"Could not read frame {frame_index}.")

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def main() -> None:
    args = parse_args()

    pose_frames = load_superanimal_multi_json(args.predictions)
    extractor = MegaDescriptorExtractor.from_pretrained()

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")

    embeddings = {}

    try:
        for frame_index in args.frames:
            if frame_index < 0 or frame_index >= len(pose_frames):
                print(f"\nFrame {frame_index}: outside prediction range")
                continue

            image = read_frame(capture, frame_index)
            pose_frame = pose_frames[frame_index]

            print(f"\nFrame {frame_index}")
            print(f"Detections: {len(pose_frame.individuals)}")

            frame_embeddings = []

            for individual in pose_frame.individuals:
                try:
                    embedding = extractor.extract(
                        image,
                        individual.dog_box,
                    )
                except ValueError as error:
                    print(
                        f"  Detection {individual.detection_index}: "
                        f"skipped ({error})"
                    )
                    continue

                key = (frame_index, individual.detection_index)
                embeddings[key] = embedding
                frame_embeddings.append(
                    (individual.detection_index, embedding)
                )

                print(
                    f"  Detection {individual.detection_index}: "
                    f"dimension={embedding.dimension}, "
                    f"box={embedding.crop_box}"
                )

            if len(frame_embeddings) > 1:
                print("  Within-frame cosine similarities:")

            for first_index in range(len(frame_embeddings)):
                for second_index in range(
                    first_index + 1,
                    len(frame_embeddings),
                ):
                    first_id, first_embedding = frame_embeddings[first_index]
                    second_id, second_embedding = frame_embeddings[second_index]

                    similarity = extractor.cosine_similarity(
                        first_embedding,
                        second_embedding,
                    )

                    print(
                        f"    detection {first_id}-{second_id}: "
                        f"{similarity:.4f}"
                    )

        print("\nCross-frame cosine similarities")

        keys = sorted(embeddings)
        for first_position, first_key in enumerate(keys):
            for second_key in keys[first_position + 1 :]:
                if first_key[0] == second_key[0]:
                    continue

                similarity = extractor.cosine_similarity(
                    embeddings[first_key],
                    embeddings[second_key],
                )

                print(
                    f"  f{first_key[0]}d{first_key[1]} ↔ "
                    f"f{second_key[0]}d{second_key[1]}: "
                    f"{similarity:.4f}"
                )
    finally:
        capture.release()


if __name__ == "__main__":
    main()