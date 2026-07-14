# Posture Detection and Injury Dataset Builder

This project now focuses on building an unbiased labeled dataset from MediaPipe pose features. The rule-based injury scorer has been removed.

## What it does

- Captures webcam frames in `main.py` for live pose preview only.
- Extracts MediaPipe pose landmarks and body angles without any rule-based injury scoring.
- Samples a selected video clip in `dataset_builder.py` using a start time, stop time, and target frames per second.
- Shows each sampled frame beside its extracted pose features so you can assign a label frame by frame.
- Saves the labeled pose features to `pose_data.csv` for later model training and testing.

## Requirements

- Python 3.10 or newer.
- A webcam.
- A laptop that can run MediaPipe inference on CPU. An i5-class laptop should work for this prototype at 640x480.

## Install

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the live preview

Preview mode is the default:

```bash
python main.py
```

To force terminal-only headless mode:

```bash
python main.py --no-preview
```

Useful options:

- `--camera 0` selects the webcam index.
- `--width 640 --height 480` requests a lower capture resolution.
- `--max-frames 30` stops after a fixed number of frames in headless mode.

## On the second laptop

1. Clone the repository.
2. Create the virtual environment and install the requirements.
3. Run `python main.py` to test the webcam preview.
4. If the laptop is slow, drop the capture size to `640x480` or run headless first.

## Dataset Creation Tool

Use `dataset_builder.py` to turn a clip into labeled pose-feature rows.

### Features

- Loads one video file.
- Accepts a clip start time, stop time, and target sampling FPS.
- Extracts per-frame pose features with MediaPipe.
- Shows the video frame and pose measurements side by side.
- Lets you label each frame as low, medium, high, or custom.
- Saves the frame-level label and pose features directly to CSV.

### Run

```bash
python dataset_builder.py --video /path/to/badminton_video.mp4 --start-time 12 --end-time 24 --sample-fps 5
```

If you want prompts instead of command-line values, omit the optional inputs:

```bash
python dataset_builder.py --video /path/to/badminton_video.mp4
```

Useful options:

- `--output-csv pose_data.csv` to choose the CSV file.
- `--append` to keep adding rows to an existing CSV.

### Labeling flow

1. The script loads the selected clip range from the video.
2. It samples frames at the requested frames per second.
3. Each sampled frame is shown beside the MediaPipe angles and torso lean.
4. In terminal, choose a label for that frame:
	- `1` low
	- `2` medium
	- `3` high
	- `4` custom

Output CSV columns:

- `video_name`, `clip_start_sec`, `clip_end_sec`, `source_fps`, `sample_fps`
- `frame_index`, `timestamp_sec`, `label`
- `left_knee_angle`, `right_knee_angle`, `knee_angle`
- `left_elbow_angle`, `right_elbow_angle`, `elbow_angle`, `torso_lean`
- `pose_detected`

## Training direction

The next model should be trained on the labeled pose features from `pose_data.csv`, not on the old rule-based injury score. That keeps the dataset unbiased and lets us compare model choices cleanly after labeling is complete.
- `label`
