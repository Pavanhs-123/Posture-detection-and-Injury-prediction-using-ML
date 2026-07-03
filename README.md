# Posture Detection and Injury Risk Prediction

This project uses a webcam, MediaPipe pose landmarks, and angle-based features to detect posture and generate a real-time injury-risk score and feedback.

## What it does

- Captures webcam frames and detects the body pose in real time.
- Measures knee and shoulder angles.
- Calculates a simple real-time injury risk score with LOW, MEDIUM, and HIGH levels.
- Draws the detected landmarks on the preview window when enabled.
- Appends the angles, risk score, and feedback to `pose_data.csv` for later analysis or model training.

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

## Run

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

Use `dataset_builder.py` to convert one badminton video into labeled training data.

### Features

- Loads one video file.
- Auto-detects movement segments from landmark motion, or lets you enter manual frame ranges.
- Extracts per-frame measurements: knee angle, elbow angle, torso lean, and movement score.
- Plays each segment with overlayed measurements.
- Lets you manually label each segment (low, medium, high, or custom).
- Saves all labeled rows to CSV automatically.

### Run

```bash
python dataset_builder.py --video /path/to/badminton_video.mp4 --output-csv movement_dataset.csv
```

Manual segment mode:

```bash
python dataset_builder.py --video /path/to/badminton_video.mp4 --manual-select
```

Useful options:

- `--playback-speed 0.75` to slow preview playback.
- `--output-csv data/movement_dataset.csv` to save into a data folder.

### Labeling flow

1. The script processes the video and computes frame-level metrics.
2. It previews each movement segment in a window.
3. In terminal, choose label options:
	- `1` low
	- `2` medium
	- `3` high
	- `c` custom label
	- `e` edit frame range
	- `s` skip segment
	- `q` quit and save what is done

Output CSV columns:

- `video_name`, `segment_id`, `segment_start_frame`, `segment_end_frame`
- `frame_index`, `timestamp_sec`
- `knee_angle`, `elbow_angle`, `torso_lean`, `movement_score`
- `label`
