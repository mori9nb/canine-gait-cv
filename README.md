# Canine Gait CV

A modular computer vision pipeline for canine hind-limb gait analysis.

## Project Goal

The goal of this project is to build a computer vision pipeline that can process dog gait videos and extract biomechanical information from the hind limb.

The final objective is to:

- Detect the dog in video frames
- Extract the hind-limb region of interest
- Estimate hind-limb skeletal keypoints
- Track keypoints across frames
- Calculate joint angles such as stifle and hock angles
- Identify gait phases such as stance and swing
- Generate structured data for biomechanical and prosthesis-related analysis

## Current Status

The current version contains the foundational pipeline and real YOLO-based dog
detection. Pose estimation is not integrated yet.

Implemented modules:

```text
src/canine_gait_cv/
├── video/
├── preprocessing/
├── visualization/
└── detection/
```

## Implemented Features

### Video Module

- Read video metadata
- Iterate through video frames
- Store frame index and timestamp

### Preprocessing Module

- Resize frames
- Crop frames
- Define bounding boxes

### Visualization Module

- Draw bounding boxes for debugging

### Detection Module

- Define a standard detection data structure
- Define a detector interface
- Add a dummy fixed-box detector for pipeline testing
- Add an Ultralytics YOLO adapter for real dog detection
- Preserve the common `detect(frame) -> list[Detection]` interface

## Testing

The project uses `pytest`.

Run tests with:

```bash
pytest -v
```

## Installation

Create and activate the Conda environment:

```bash
conda create -n canine-gait-cv python=3.10 -y
conda activate canine-gait-cv
```

Install the project in editable mode:

```bash
python -m pip install -e ".[dev]"
```

Install the optional YOLO dependency when running real dog detection:

```bash
python -m pip install -e ".[dev,yolo]"
```

## Real Dog Detection

Run YOLO on a gait video and save an annotated preview:

```bash
python scripts/detect_dog_video.py \
    data/raw/dog_test.mp4 \
    outputs/detection/dog_test_detected.mp4 \
    --weights yolo11n.pt \
    --confidence 0.25 \
    --width 1280
```

The first run downloads the requested pretrained weights if they are not already
available. The detector keeps only the COCO `dog` class, clips coordinates to the
frame, and the demo selects the largest detected dog as the gait subject.

## Project Structure

```text
canine-gait-cv/
├── src/canine_gait_cv/
│   ├── video/
│   ├── preprocessing/
│   ├── visualization/
│   └── detection/
├── tests/
├── scripts/
├── reports/
├── pyproject.toml
├── README.md
└── .gitignore
```

## Next Steps

Planned next steps:

1. Validate YOLO dog detection on representative gait videos.
2. Use dog detection to define a hind-limb region of interest.
3. Add keypoint structures for canine hind-limb joints:
   - hip
   - stifle
   - hock
   - paw
4. Implement joint-angle calculation.
5. Add gait phase detection logic.
6. Integrate Kalman filtering for smoother ROI tracking.

## Notes

Raw videos, processed data, model weights, and output files are excluded from GitHub using `.gitignore`.
