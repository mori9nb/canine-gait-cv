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

The current version contains the foundational structure of the pipeline. Real YOLO detection and pose estimation are not integrated yet.

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

## Testing

The project uses `pytest`.

Current test status:

```text
15 passed
```

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

1. Add a real dog detector, likely YOLO-based.
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