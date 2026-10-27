# Badminton Posture Detection and Injury Risk Assessment Using Machine Learning

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![Tests](https://img.shields.io/badge/tests-121%20passing-brightgreen.svg)]()
[![Rubric](https://img.shields.io/badge/rubric-v0.3%20deterministic-orange.svg)]()
[![License](https://img.shields.io/badge/license-Academic%20Prototype-lightgrey.svg)]()

> **Final-Year B.Tech Capstone Project**  
> An automated computer-vision and machine-learning pipeline for athletic badminton movement detection, kinematic joint analysis, and deterministic biomechanical posture-risk screening.

---

## ⚠️ Academic Screening Disclaimer

> **IMPORTANT**: This software is an **academic engineering screening prototype** developed for research, sports engineering, and athletic movement review.
>
> 1. It is **NOT** a certified medical diagnostic tool, clinical device, or predictive model of medical pathology.
> 2. The **Biomechanical Risk Score (0–100)** represents a deterministic kinematic deviation score derived from published sports science literature (e.g., knee deceleration angles, trunk flexion, bilateral asymmetry).
> 3. A score of **70 / 100** indicates **elevated kinematic posture deviation** from reference mechanics; it does **NOT** represent a 70% probability of suffering a ligament rupture, tendon tear, or clinical injury.
> 4. Under **Rubric v0.3**, low-quality tracking frames (< 0.45 landmark confidence), missing joint angles, or movements outside the target specification are labeled **`UNASSESSED`**. Missing evidence is **never** substituted with zero risk or arbitrary guesses.

---

## 🎯 Target Movement Classes

The pipeline is calibrated for three primary high-demand athletic maneuvers in badminton:

1. **`LUNGE`** — Front-court lunges requiring deep knee flexion, rapid deceleration cushioning, and torso stability.
2. **`SMASH_LANDING`** — Rear-court overhead smash landings involving bilateral impact shock absorption and rapid trunk re-alignment.
3. **`CHANGE_OF_DIRECTION`** — Mid-court cutting, split-recovery, and lateral deceleration maneuvers.

*Non-Target Outcomes:*
- **`UNCERTAIN`** — Ambiguous or occluded movement candidates where tracking quality or kinematic patterns do not meet confirmation thresholds.
- **`NO_EVENT`** — Background idle frames, court walking, or service preparation.
- *Deferred Classes:* `SCISSOR_JUMP` and `FOOTWORK_SPLIT_STEP` are tracked experimentally in detection proposals but remain formally deferred from automated risk scoring in Rubric v0.3.

---

## 🔬 System Architecture

The end-to-end pipeline operates in five tightly coupled phases:

```
Video Input (MP4/MOV)
       │
       ▼
1. Validation & Pose Tracking (MediaPipe Pose + Confidence Gating ≥ 0.45)
       │
       ▼
2. Temporal Signal Extraction & Candidate Window Detection
       │
       ▼
3. Movement Classification (Random Forest / Group-KFold Model)
       │
       ▼
4. Biomechanical Risk Evaluation (Rubric v0.3: 0–100 Deterministic Scoring)
       │
       ▼
5. Interactive Web Dashboard (Streamlit UI) + Structured Manifest Export (JSON / CSV)
```

### 1. Pose Estimation & Confidence Filtering
- Tracks 33 anatomical landmarks using Google MediaPipe Pose.
- Computes planar sagittal joint angles (left/right knee flexion, left/right elbow angle, and trunk lean from vertical).
- **Zero Synthetic Neutral Imputation**: Frames where pose is lost (`has_pose=False`) emit `None` for joint coordinates and angles rather than synthetic neutral assumptions (`180.0°` or `0.5`). On tracking reacquisition, differential hip velocity is reset to prevent artificial kinetic acceleration spikes.

### 2. Candidate Detection & Kinematic Peak Selection
- Scans temporal knee flexion and hip velocity signals for dynamic deceleration events.
- Extracts candidate bounding intervals `[start_time, end_time]`.
- Identifies representative keyframes using a smoothed local median (±0.12s) around the kinematic peak.

### 3. Movement Classification & Group-KFold Validation
- Classifier: Random Forest ensemble (`models/movement_classifier_reviewed_161_rf.pkl`). Tree ensemble outputs represent uncalibrated class vote fractions rather than posterior probabilities.
- Training Dataset: 161 independently reviewed and verified badminton movement events (`data/dataset_v20261025_reviewed.csv`).
- Cross-Validation: **3-fold GroupKFold grouped strictly by `player_id`** across six distinct athletes (`player_01`–`player_06`; test partitions hold 1, 2, and 3 players respectively) to prevent subject and background leakage across train and test partitions.
- Cross-Validation Metrics: **Macro F1 = 0.5797**, **Accuracy = 0.6211**.

### 4. Deterministic Rubric v0.3 Biomechanical Risk Scoring
The posture score combines three literature-grounded risk indicators into a normalized 0–100 scale:

$$	ext{Score} = 	ext{KneeFlexionPts} + 	ext{TorsoLeanPts} + 	ext{BilateralAsymmetryPts}$$

| Biomechanical Indicator | Max Points | Optimal Range (0 pts) | Elevated Concern Threshold | Biomechanical Rationale |
| :--- | :---: | :---: | :---: | :--- |
| **Knee Flexion Cushioning** | **40 pts** | $95.0^\circ \le 	heta \le 115.0^\circ$ | Stiff ($> 115^\circ$) or Acute ($< 95^\circ$) | Stiff landings (>115°) increase ground reaction forces; acute hyperflexion (<95°) elevates patellofemoral compressive stress. |
| **Torso Lean Alignment** | **35 pts** | $\le 15.0^\circ$ | Excess Lean ($> 15.0^\circ$) | Forward/lateral trunk displacement shifts center of mass outside base of support, increasing knee abduction moments. |
| **Bilateral Knee Asymmetry** | **25 pts** | $\le 10.0^\circ$ | Excess Asymmetry ($> 10.0^\circ$) | Asymmetric loading indicates unilateral shock bias and compensatory limb mechanics. |

**Screening Category Bands:**
- **`LOW RISK`**: $[0.0, 30.0)$ — Kinematic measurements align with reference cushioning mechanics.
- **`MODERATE RISK`**: $[30.0, 60.0)$ — Mild-to-moderate deviation in deceleration cushioning or torso posture.
- **`HIGH RISK`**: $[60.0, 100.0]$ — Significant postural deviation indicating stiff impact absorption or excessive lean.
- **`UNASSESSED`**: Score withheld if lower-body landmark quality $< 0.45$, key angles are occluded/missing, or movement class is outside active scope.

---

## 📁 Repository Map

```
badminton_posture_ai/
├── app/
│   └── streamlit_app.py        # Streamlit web dashboard (Redesigned for sports biomechanics)
├── assets/
│   └── pose_landmarker_lite.task # MediaPipe model asset
├── configs/
│   └── pipeline_config.json    # Central pipeline, rubric, and model paths configuration
├── data/
│   ├── ASSET_MANIFEST.md       # SHA-256 provenance manifest of all datasets and models
│   ├── dataset_v20261025_reviewed.csv # 161-row reviewed feature dataset
│   ├── finalized_review_manifest.csv  # 386-candidate review manifest
│   ├── accepted_target_candidates.csv # Accepted target candidates
│   ├── history.db              # SQLite database storing analysis history
│   ├── processed/              # Output directory for annotated videos, clips, and manifests
│   └── test_videos/            # Sample test videos (Test_Lunge.mp4, Test_Lunge_2.mp4)
├── models/
│   ├── model_registry.json     # Active model registry and cross-validation metadata
│   ├── movement_classifier_reviewed_161_rf.pkl # Primary 161-row Random Forest classifier
│   └── movement_classifier.pkl # Baseline 34-sample legacy model
├── reports/
│   └── evaluation/
│       └── movement_group_cv_report.json # Group-KFold cross-validation report
├── src/
│   ├── classification/         # Movement classifiers and category schemas
│   ├── detection/              # Candidate interval detector and manifest schemas
│   ├── pose/                   # MediaPipe pose extraction and 2D/3D angle calculators
│   ├── risk/                   # Rubric v0.3 deterministic biomechanical score evaluator
│   └── video/                  # Video validator, clip extractor, and pipeline coordinator
└── tests/                      # 121 comprehensive pytest unit & integration tests
```

---

## 🚀 Quickstart & Setup Guide

### 1. Environment Requirements
- **Operating System:** Linux (Ubuntu 20.04+, Arch, Debian) or macOS / Windows.
- **Python Version:** Python 3.11 (tested on Python 3.11.15).

### 2. Virtual Environment Setup
```bash
# Clone or navigate to the repository
cd /home/pavan/projects/badminton_posture_ai

# Create virtual environment with Python 3.11
python3.11 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### 3. Run the Regression Test Suite
Run the 121 unit and integration tests across pose estimation, candidate detection, model splitting, and rubric evaluation:
```bash
PYTHONPATH=. .venv/bin/pytest tests
```
*Expected result: 121 passed.*

### 4. Launch the Streamlit Web Application
Start the interactive dashboard locally:
```bash
PYTHONPATH=. .venv/bin/streamlit run app/streamlit_app.py --server.port 8501
```
Open your browser at `http://localhost:8501`.

---

## 💻 Video Processing from Python CLI

You can also run video analysis directly through the Python API:

```python
from src.video.video_pipeline import VideoProcessingPipeline

pipeline = VideoProcessingPipeline(model_path="models/movement_classifier_reviewed_161_rf.pkl")

results = pipeline.process_video(
    video_path="data/test_videos/Test_Lunge.mp4",
    output_dir="data/processed/lunge_run",
    sample_fps=15.0,
)

print(f"Detected {results['total_events']} movement event(s).")
for ev in results["events"]:
    print(f"Event: {ev['movement_category']} | Risk: {ev.get('risk_category')} (Score: {ev.get('risk_score')})")
```

---

## 📜 Citations & Biomechanical References

The kinematic thresholds in Rubric v0.3 are grounded in published sports engineering and badminton injury literature:
- **Krosshaug et al. (2007)** — Mechanisms of ACL injuries in sports: trunk lean, knee valgus collapse, and stiff landing angles ($> 115^\circ$ knee extension).
- **Huang et al. (2014)** — Biomechanical analysis of badminton lunges: knee flexion cushion windows and deceleration ground reaction forces.
- **Lin et al. (2015)** — Lower-extremity kinematics and bilateral asymmetry in overhead smash landings.
