# Data Directory Organization

This directory stores video assets, candidate proposals, manifests, and dataset extracts.

## Directory Structure
- `raw/`: Unprocessed court video files (`.mp4`, `.avi`, `.mov`). Kept outside Git tracking.
- `processed/`: Processed videos with skeletal tracking overlays and cut clips.
- `annotations/`: Human-reviewed decisions, corrections, and verified labels (`candidate_decisions.csv`).
- `manifests/`: Structured event manifests (`event_manifest.json`) output by the video pipeline.
- `clips/`: Extracted candidate and event video segments.
- `landmarks/`: Raw landmark coordinates per sequence (`.json`).
- `candidates.csv`: Working candidate detector output.
- `dataset.csv`: Approved sequence dataset assembled from accepted human-reviewed candidates.

## Data Protection Notice
Video binaries, candidate clips, and CSV exports are excluded from Git commits via `.gitignore` to prevent repository bloat and respect athlete privacy.
