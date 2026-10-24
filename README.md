# Badminton Posture Detection: Movement Events and Biomechanical Observations

This repository contains a local video analysis prototype. It proposes badminton movement events, extracts 2D pose measurements, and provides an interface for human review. It is not a medical device, a clinical injury predictor, or a validated risk classifier.

## Active movement categories

The active categories are `LUNGE`, `SMASH_LANDING`, and `CHANGE_OF_DIRECTION`. `SCISSOR_JUMP` and `FOOTWORK_SPLIT_STEP` remain experimental and disabled by default. Ambiguous or unsupported observations remain `UNCERTAIN` or `NO_EVENT` as appropriate.

## Workflow

1. Upload a video in the Streamlit app or call the video pipeline.
2. The pipeline validates the video, extracts one MediaPipe pose per frame, filters low-confidence measurements, and proposes candidate event intervals.
3. The movement classifier reports its source and keeps detector labels separate from classifier predictions. Heuristic output is identified as heuristic and does not include fabricated probabilities.
4. Review candidates and manifests before adding labels to training data.
5. Use grouped evaluation by player or source video. Training stops when the available data cannot support independent partitions.

The pipeline currently tracks one player. Use footage with one clearly visible player. Active joint measurements are image-plane 2D estimates and depend on camera view; they are not 3D clinical measurements.

## Setup

The verified environment for this code is Windows with Python 3.11.0. Linux development is intended but has not been verified in this audit.

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
```

Linux/macOS:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install -r requirements-dev.txt
```

The MediaPipe Pose Landmarker task asset is under `assets/pose_landmarker_lite.task`. It is the same content already tracked at the root of the public `main` branch, moved to match the local project layout.

## Run the app

```bash
.\.venv\Scripts\python.exe -m streamlit run app.py
```

On Linux/macOS, use `.venv/bin/python -m streamlit run app.py`.

## Tests and checks

```bash
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m pip check
```

Use `.venv/bin/python` on Linux/macOS. Tests create temporary fixtures. The audit environment passed 85 tests; this result does not establish Linux compatibility or movement-classification accuracy.

## Data and model artifacts

This branch intentionally does not include user videos, processed clips, review history, the local labeled dataset, the history database, or serialized classifier artifacts. These files may contain private footage, user labels, or local-only outputs. Supply independently reviewed data and trusted model artifacts locally when needed. The current labeled dataset is too small to support strong generalization claims.

Pickle model files can execute code when loaded. Only load artifacts from a trusted source. The dashboard may use the documented heuristic baseline when a trained movement model is absent. Automated `LOW`/`MODERATE`/`HIGH` injury-risk prediction remains disabled; rubric output is provisional, uncertain, or unassessed.

## Project layout

- `app/`, `app.py`: Streamlit application.
- `src/video/`, `src/pose/`, `src/features/`, `src/detection/`: video and pose processing.
- `src/classification/`, `src/risk/`: movement classification and conservative rubric observations.
- `src/detection/manifest.py`: event manifest schema and provenance.
- `scripts/`: human review and reviewed-data utilities.
- `training/`: grouped movement-model evaluation and training utilities.
- `tests/`: unit and integration tests.
- `docs/rubric_v0.3.md`: current rubric description and limitations.

`requirements-lock.txt` from the local folder was not migrated because its pins do not match the verified environment. `requirements.txt` contains the pinned direct runtime dependencies; `requirements-dev.txt` adds the test runner. A clean cross-platform lock has not been generated or tested.
