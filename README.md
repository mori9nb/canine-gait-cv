# Canine Gait Computer Vision Pipeline

A computer-vision pipeline for markerless canine hind-limb gait analysis from side-view videos.

This project detects dogs, estimates anatomical keypoints, tracks individuals, evaluates detection quality, extracts knee and hock kinematics, identifies gait phases and events, and generates normalized reference trajectories for later canine prosthesis simulation and variable-stiffness control.

## Project Information

- **Developer:** Morteza Nabipour Pashaki
- **Project type:** Computer Vision Internship Project
- **Supervisor:** Prof. Francesco La Rosa
- **Institution:** University of Messina
- **Python:** 3.10
- **DeepLabCut:** 3.0.1

## Project Objective

The objective is to transform canine side-view videos into reliable gait information that can later support the design and control of a personalized adaptive hind-limb prosthesis.

The computer-vision pipeline produces:

- dog detections and bounding boxes;
- anatomical keypoints;
- persistent multi-dog track IDs;
- confidence and quality metadata;
- right hind-limb knee and hock angles;
- angular velocities;
- stance and swing phases;
- contact and toe-off events;
- per-stride measurements;
- normalized gait reference trajectories.

The vision output is intended to provide offline reference kinematics and training labels. Future prosthesis stages will combine this reference with IMU and FSR measurements, mechanical simulation, and variable-stiffness optimization.

## Pipeline Overview

```text
Input video
    |
    v
Dog detection and bounding boxes
    |
    v
DeepLabCut keypoint estimation
    |
    +-----------------------------+
    |                             |
    v                             v
Multi-dog tracking          Detection quality assessment
    |                             |
    +-------------+---------------+
                  |
                  v
Four-keypoint hind-limb model
(thigh/hip anchor, knee, hock, paw)
                  |
                  v
Confidence-aware temporal processing
                  |
                  v
Knee and hock kinematics
                  |
                  v
Stance / swing classification
                  |
                  v
Contact and toe-off detection
                  |
                  v
Complete stride extraction
                  |
                  v
0-100% phase normalization
                  |
                  v
Mean gait reference trajectory
```

## Main Components

### 1. Dog Detection

The project supports dog detection and bounding-box extraction. DeepLabCut SuperAnimal uses a Faster R-CNN detector together with an HRNet pose model.

### 2. DeepLabCut Adapter

The DeepLabCut adapter parses both single-animal and multi-animal SuperAnimal output.

It handles:

- padded unused detection slots;
- multiple dogs in one frame;
- DeepLabCut `xywh` bounding boxes;
- confidence values;
- native-to-project keypoint mapping;
- frames without valid detections.

### 3. Multi-Dog Tracking

The multi-dog tracker combines:

- bounding-box motion;
- predicted track position;
- intersection over union;
- appearance embeddings;
- configurable missed-frame tolerance;
- active-track limits.

Appearance embeddings are extracted using MegaDescriptor. They reduce identity switches, although severe overlap and visually similar dogs remain challenging.

### 4. Detection Quality Assessment

Each detection is evaluated using:

- bounding-box confidence;
- reliable keypoint coverage;
- overlap with other detections;
- pose quality.

Raw detections are preserved, while uncertain detections can be excluded from clean training datasets.

### 5. Four-Keypoint Hind-Limb Model

The general SuperAnimal model does not provide a dedicated canine hock keypoint. A specialized model was therefore fine-tuned for:

```text
back_right_thai
back_right_knee
back_right_hock
back_right_paw
```

The custom model uses 60 manually annotated frames from a sagittal-view canine gait video.

Available annotations:

| Keypoint | Labeled frames |
|---|---:|
| Right thigh/hip anchor | 52/60 |
| Right knee/stifle | 56/60 |
| Right hock | 55/60 |
| Right paw | 55/60 |

Preliminary internal validation results:

| Metric | Result |
|---|---:|
| RMSE | 7.84 px |
| mAP | 100.00% |
| mAR | 100.00% |

These metrics are based on a small internal validation split from the same video source and must not be interpreted as independent-video generalization performance.

### 6. Temporal Processing

The project contains confidence-aware Kalman filtering and Rauch-Tung-Striebel smoothing.

Temporal processing is used conservatively to:

- reduce frame-to-frame keypoint jitter;
- reject implausible measurements;
- estimate very short internal gaps;
- preserve raw measurements for inspection.

Long or unreliable missing intervals are not reconstructed as valid clean data.

### 7. Joint Kinematics

The knee angle is calculated from:

```text
thigh/hip anchor -> knee -> hock
```

The hock angle is calculated from:

```text
knee -> hock -> paw
```

Angular velocity is the time derivative of the corresponding joint angle:

```text
knee_velocity_deg_s
hock_velocity_deg_s
```

It represents joint flexion/extension rate, not the translational velocity of the dog or paw.

Recorded four-keypoint analysis:

| Measurement | Result |
|---|---:|
| Video frames | 187 |
| Fully observed frames | 159 |
| Short-gap estimates | 1 |
| Rejected/incomplete frames | 28 |
| Knee motion range | 53.33 degrees |
| Hock motion range | 51.20 degrees |

### 8. Gait Phase and Event Detection

Stance and swing are estimated using paw motion relative to the proximal limb rather than raw image displacement.

The motion is normalized by the estimated thigh-to-paw scale to reduce dependence on image resolution and dog size.

Definitions:

- **Contact:** transition from swing to stance
- **Toe-off:** transition from stance to swing
- **Stride:** contact-to-next-contact interval of the same paw
- **Stance duration:** contact-to-toe-off interval
- **Swing duration:** toe-off-to-next-contact interval

### 9. Stride Analysis

Five complete strides were extracted from the evaluated video.

| Stride | Contact | Toe-off | Next contact | Duration | Stance | Swing |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 26 | 43 | 52 | 0.867 s | 65.4% | 34.6% |
| 1 | 52 | 68 | 77 | 0.833 s | 64.0% | 36.0% |
| 2 | 77 | 93 | 102 | 0.833 s | 64.0% | 36.0% |
| 3 | 102 | 120 | 130 | 0.933 s | 64.3% | 35.7% |
| 4 | 130 | 146 | 156 | 0.867 s | 61.5% | 38.5% |

Aggregate results:

| Metric | Mean | Coefficient of variation |
|---|---:|---:|
| Stride duration | 0.867 s | 4.71% |
| Stance fraction | 63.84% | 2.21% |
| Swing fraction | 36.16% | 3.89% |
| Knee range of motion | 44.36 degrees | 19.15% |
| Hock range of motion | 41.08 degrees | 10.42% |

### 10. Normalized Gait Reference

Each complete stride is interpolated onto a common gait-cycle axis containing 101 samples from 0% to 100%.

Generated reference data:

| Output | Size |
|---|---:|
| Complete source strides | 5 |
| Samples per stride | 101 |
| Normalized per-stride rows | 505 |
| Mean reference rows | 101 |
| Mean stride duration | 0.8667 s |
| Mean toe-off phase | 63.84% |

The reference trajectory contains:

- normalized phase;
- gait phase;
- knee reference angle;
- knee angular velocity;
- hock reference angle;
- hock angular velocity;
- standard deviation;
- lower and upper variability bounds;
- mean toe-off percentage;
- source-stride count.

## Repository Structure

```text
canine-gait-cv/
├── scripts/
│   ├── inspect_video.py
│   ├── export_single_dog_gait_analysis.py
│   ├── export_hock4_gait_analysis.py
│   ├── render_hock4_gait.py
│   ├── export_stride_analysis.py
│   ├── render_hock4_stride_video.py
│   ├── export_normalized_stride_reference.py
│   ├── inspect_dog_embeddings.py
│   └── evaluate_appearance_tracking.py
├── src/canine_gait_cv/
│   ├── appearance/
│   ├── dataset/
│   ├── detection/
│   ├── gait/
│   ├── pose/
│   ├── preprocessing/
│   ├── quality/
│   ├── roi/
│   ├── smoothing/
│   ├── tracking/
│   ├── video/
│   └── visualization/
├── tests/
├── pyproject.toml
├── README.md
└── .gitignore
```

## Installation

Clone the repository:

```bash
git clone https://github.com/mori9nb/canine-gait-cv.git
cd canine-gait-cv
```

Create and activate a Python environment:

```bash
conda create -n canine-gait-dlc python=3.10
conda activate canine-gait-dlc
```

Install the project in editable mode:

```bash
python -m pip install -e .
```

DeepLabCut, PyTorch, CUDA, OpenCV, Pandas, NumPy, Matplotlib and the optional appearance-model dependencies must be installed separately when their related pipelines are used.

## Running the Tests

Run the full test suite:

```bash
python -m pytest -q
```

Latest recorded result:

```text
125 passed, 13 warnings
```

The recorded warnings are third-party Matplotlib/PyParsing deprecation warnings and do not indicate test failures.

## Example Commands

Inspect video metadata:

```bash
python scripts/inspect_video.py data/raw/video.mp4
```

Inspect the command-line options of each processing stage:

```bash
python scripts/export_hock4_gait_analysis.py --help
python scripts/render_hock4_gait.py --help
python scripts/export_stride_analysis.py --help
python scripts/render_hock4_stride_video.py --help
python scripts/export_normalized_stride_reference.py --help
```

Example stride analysis:

```bash
python scripts/export_stride_analysis.py \
  outputs/gait/single_dog_hock4/hind_limb_kinematics_raw.csv \
  --fps 30 \
  --forward-direction right \
  --output-dir outputs/gait/single_dog_hock4/stride
```

## Generated Outputs

The pipeline can generate:

```text
hind_limb_kinematics_raw.csv
hind_limb_kinematics_clean.csv
gait_phase_frames.csv
gait_events.csv
stride_summary.csv
normalized_stride_profiles.csv
mean_reference_trajectory.csv
knee_hock_angle_plot.png
gait_phase_plot.png
normalized_knee_hock_plot.png
four_keypoint_gait_analysis.mp4
gait_phase_stride_analysis.mp4
```

Large datasets, raw videos, trained model weights and generated outputs are not committed to Git.

## Current Limitations

- The specialized four-keypoint model was trained from one main video source.
- Independent-video generalization has not yet been fully validated.
- Only the right hind limb is included in the specialized gait model.
- Severe overlap can still cause identity switches in multi-dog tracking.
- The current analysis is two-dimensional and assumes a suitable sagittal view.
- Monocular pixel coordinates do not provide direct 3-D or metric kinematics.
- The normalized reference still requires cyclic endpoint preparation before continuous online control.
- No synchronized IMU/FSR and video dataset is currently available.
- Computer vision alone cannot provide optimal prosthesis stiffness labels.
- Optimal stiffness must be generated through mechanical simulation, optimization, or synchronized physical experiments.

## Relationship to the Prosthesis Controller

The computer-vision stage provides target gait information:

```text
normalized phase
gait phase
knee reference angle
knee reference angular velocity
hock reference angle
hock reference angular velocity
reference variability
quality/confidence
```

The future online system will use IMU and FSR measurements to estimate the current prosthesis state.

The controller will compare the measured state with the vision-derived reference and determine bounded changes to:

- `k_calf`: rotational stiffness around the prosthetic hock;
- `k_paw`: linear/compressive stiffness in the paw-ground region.

The computer-vision reference is therefore an input to the later simulation and learning pipeline, not a direct stiffness prediction.

## Next Steps

1. Make the normalized reference cyclic at the 100%-to-0% boundary.
2. Validate the four-keypoint model on an independent side-view video.
3. Freeze the final computer-vision output schema.
4. Define the IMU/FSR-to-CV data interface.
5. Implement the 2-D prosthesis mechanical simulation.
6. Generate stiffness-response training data.
7. Train and evaluate the adaptive stiffness-selection model.

## Data and Model Availability

Raw videos, DeepLabCut projects, annotations, trained snapshots and generated analysis outputs are stored separately because of file size, licensing and reproducibility considerations.

Only source code, tests and lightweight documentation are maintained in this repository.

## License and Usage

This repository was developed for academic and internship research.

Third-party datasets, pretrained models, videos and libraries remain subject to their respective licenses.