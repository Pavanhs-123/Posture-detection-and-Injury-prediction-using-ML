# TECHNICAL BRIEF — Badminton Posture Detection & Injury-Risk Assessment System

This document outlines the architecture, pipeline design, current status, and deployment strategy for the **AI-Based Badminton Posture Detection & Injury-Risk Assessment System**. It is designed to introduce potential industry, academic, or corporate collaborators to the project's technical foundations and outline area requirements for future development.

---

## 1. PROJECT OVERVIEW

*   **Project Title**: AI-Based Badminton Posture Detection & Injury-Risk Assessment System
*   **Problem Statement**: Badminton is a high-speed sport characterized by repetitive, explosive movements (smashes, lunges, rapid changes of direction). Improper mechanics, joint landing angles, and torso lean increase joint loading and muscle strain, leading to common injuries like ACL tears, patellar tendinopathy, and shoulder impingement.
*   **Main Objective**: To build an automated, non-invasive biomechanical evaluation system that flags posture safety margins and identifies high-risk movements from standard video feeds.
*   **Intended AI Functionality**: The system extracts player skeletal landmarks, calculates joint angles and trunk alignment in real-time, and feeds sequence-level aggregated statistics to a machine learning classifier to categorize risk levels.
*   **Current Prototype Capability**: The system is fully structured end-to-end. It features standard side-angle recording video loading, MediaPipe pose tracking, 2D angle and torso lean feature computations, and a CLI-driven sequence annotation/saving pipeline. A complete Streamlit dashboard allows video uploads, slices clips by time, renders skeletal overlays, visualizes biomechanical metrics, provides dynamic explanations, logs history via local SQLite, and runs predictions using a rule-based Integration Test Model.
*   **Final System Scope**: Once a complete labeled dataset is compiled, the application will load a trained supervised ML pipeline to predict risk categories on new badminton videos uploaded by players and coaches.

### ⚠️ Biomechanical Definition: Injury-Risk vs. Injury Prediction
The system evaluates **movement-related injury-risk indicators** present in posture mechanics. It classifies movement execution as **LOW**, **MODERATE**, or **HIGH** risk based on anatomical boundaries (e.g. knee flexion angles, torso lean). It does **not** predict if or when a specific player will suffer an injury in the future, as injury occurrence depends on external variables (load history, muscle fatigue, shoe friction, tissue health) that computer vision cannot measure.

---

## 2. CURRENT PROJECT STATUS

The table below summarizes the implementation status of all key layers in the repository:

| Layer / Component | Status | Description |
| :--- | :---: | :--- |
| **Computer Vision / Pose Detection** | ✅ | Pre-trained MediaPipe Pose Landmarker tracks joints frame-by-frame. |
| **Biomechanical Feature Extraction**| ✅ | Computes 2D knee flexions, elbow extensions, and torso tilts. |
| **Dataset Builder Script** | ✅ | Console-driven clipping tool saves sequence-level metrics. |
| **Sequence-Based Dataset CSV** | ✅ | Structure shifted: one row represents one movement sequence. |
| **Expert Labeling Workflow** | ✅/🟡 | Implemented sequence-level prompts, but dataset is currently empty. |
| **ML Training Pipeline** | 🟡 | Code implemented in `train.py`, but awaits real training data. |
| **Final Injury-Risk Model** | ⏳ | Pending completion of dataset collection and model training. |
| **Inference Pipeline** | ✅/🟡 | Pipeline extracts, drawing overlays and predicting (Demo Mode fallback). |
| **Web Interface** | ✅ | Streamlit dashboard built, loading configurations and playing overlays. |
| **Model Integration** | ✅/🟡 | Pipeline loads model from `models/` or falls back to integration model. |
| **Testing Suite** | ✅ | Unit test suite verifies math, rules, and mock video processing. |
| **Deployment** | ⏳ | Code runs locally; cloud containerization (Docker) is planned. |

**Legend**:  
*   ✅ **Implemented and working**  
*   🟡 **Implemented but dependent on future data/model**  
*   ⏳ **Planned / remaining**  
*   ❌ **Not implemented**

---

## 3. COMPLETE TECHNOLOGY STACK

All technologies currently integrated and configured in the project are outlined below:

| Layer | Technology | Purpose | Current Usage |
| :--- | :--- | :--- | :--- |
| **Language** | Python 3.10+ | Core language environment | Applied across all scripts and application codes. |
| **Computer Vision** | MediaPipe (0.10.35) | Skeletal joint extraction | Loads `pose_landmarker_lite.task` to extract 33 landmarks. |
| **Video Decoding** | OpenCV (4.13.0) | Frame loading & drawing | Handled via `cv2.VideoCapture` and `cv2.line`/`cv2.circle`. |
| **Video Compression** | FFmpeg | Browser video rendering | Runs subprocess to compress processed videos into H.264 MP4. |
| **Data Processing** | NumPy & Pandas | Matrix math & dataframe logs | Handles coordinate geometry and splits dataset values. |
| **Machine Learning** | Scikit-learn (1.9.0) | Classification framework | Prepares Random Forest pipeline and scales values (`StandardScaler`). |
| **Storage (CSV)** | CSV Library | Dataset cataloging | `dataset_builder.py` writes sequence statistics to `data/dataset.csv`. |
| **Storage (JSON)** | JSON Library | Raw landmark preservation | Saves frame coordinates to `data/landmarks/{sequence_id}.json`. |
| **Storage (Database)**| SQLite | Local application logging | Writes analyses metadata to `data/history.db` for the UI history tab. |
| **Web Application** | Streamlit (1.61.1) | Frontend GUI interface | Launches the main web app at `http://localhost:8501`. |
| **Testing** | unittest | Code testing framework | Runs verification cases located in [`tests/test_pipeline.py`](file:///home/pavan/Programs/Posture-detection-and-Injury-prediction-using-ML/tests/test_pipeline.py). |

---

## 4. HIGH-LEVEL ARCHITECTURE

The diagram below shows how the current application components connect. Solid borders represent operational workflows, while dashed borders indicate the pipeline's future connection to the trained model.

```text
                  🏸 INPUT BADMINTON VIDEO
                             │
                             ▼
                    [ OpenCV VideoCapture ]
                             │
                             ▼
               [ MediaPipe Pose Landmarker ]
                             │
                             ▼
                [ 2D Coordinates (x, y) ]
                             │
                             ▼
              [ Biomechanical feature math ]
               (Knee/Elbow Flexion, Torso Tilt)
                             │
              ┌──────────────┴──────────────┐
              ▼                             ▼
       [ DATA COLLECTION ]          [ INFERENCE RUNNER ]
      (dataset_builder.py)            (pipeline.py)
              │                             │
              ▼                             ▼
     data/landmarks/*.json          [ Model Abstraction ]
     data/clips/*.mp4               (src/model/interface.py)
     data/dataset.csv                       │
            :                               ├ - - - - - - - - - - - - - - -┐
            : (Future Train)                ▼                              ▼
    ┌ - - - - - - - - - - ┐      [ Integration Test Model ]      [ Trained ML Model ]
    :  Scikit-learn       :      (Deterministic rules check)    (models/injury_risk_model.pkl)
    :  (train.py)         :                 │                              │
    └ - - - - - - - - - - ┘                 └──────────────┬───────────────┘
                                                           ▼
                                               [ Prediction Results ]
                                               (LOW / MODERATE / HIGH)
                                                           │
                                                           ▼
                                                [ Streamlit Dashboard ]
                                                (app/streamlit_app.py)
                                                           │
                                                           ├─► Annotated video playback
                                                           ├─► Feature gauges & explanations
                                                           └─► sqlite3 local log history
```

---

## 5. COMPLETE WORKFLOW A — DATA COLLECTION

Data collection relies on a standardized filming protocol to produce consistent biomechanical metrics:

```text
Record Side-View Video (fixed camera, side orientation, 1.2-1.5m height)
   ↓
Select Movement Sequence (e.g. identify lunge sequence start/end seconds)
   ↓
Run dataset_builder.py (extracts clips and runs landmark processing)
   ↓
Biomechanical Extraction (aggregates minimum knees/elbows and maximum lean)
   ↓
Expert Review / Labeling (expert reviews loop playback and assigns type and risk)
   ↓
Write to data/dataset.csv (records sequence row and writes landmarks to JSON)
```

1.  **Recording**: Videos are shot under fixed conditions.
2.  **Slicing**: Start/end seconds are selected to isolate the movement.
3.  **Extraction**: The system runs MediaPipe Pose Landmarker on matching frames.
4.  **Math calculation**: Feature coordinates are processed frame-by-frame.
5.  **Labeling**: The annotator inputs the category and risk label in the terminal.
6.  **Storage**: Sequence features append to `dataset.csv` and raw coordinate timelines write to JSON.

---

## 6. COMPLETE WORKFLOW B — DATASET BUILDER

The file [`dataset_builder.py`](file:///home/pavan/Programs/Posture-detection-and-Injury-prediction-using-ML/dataset_builder.py) executes the training data generation pipeline:

1.  **Input**: Receives video path, start time, end time, and target sample FPS.
2.  **Processing**: Loads video frames using `cv2.VideoCapture` and crops the clip matching the time boundaries.
3.  **Skeletal Extraction**: Feeds frames to MediaPipe landmarker, tracking joint paths.
4.  **Coordinate Logging**: Captures normalized coordinates and confidence values for the 33 joints.
5.  **Biomechanical Feature Math**: Calculates angles on every frame matching the target FPS intervals.
6.  **Loop Preview Playback**: Opens an OpenCV window playing the annotated sequence frame loop (rendering skeletons and angles).
7.  **Terminal Prompts**: Prompts the user for:
    *   **Movement Type**: Selected from category presets.
    *   **Injury-Risk Label**: LOW, MODERATE, or HIGH.
8.  **Output Export**:
    *   Appends aggregated features (min/max angles) to `data/dataset.csv`.
    *   Dumps raw landmark coordinate lists to `data/landmarks/{sequence_id}.json`.
    *   Saves the extracted video clip to `data/clips/{sequence_id}.mp4`.

### 💡 Design Decision: Why One Row = One Movement Sequence (Not One Frame)
Treating individual frames as independent samples creates massive temporal dependency and ignores context. A player might reach a deep knee angle at the bottom of a lunge, which is normal, but holding that posture too long or showing unstable torso lean over the landing sequence indicates risk. Grouping frames into a sequence and summarizing metrics (min/max bounds) captures the **dynamic movement context** in a single record, aligning with how sports scientists evaluate posture.

---

## 7. MOVEMENT CATEGORIES

The system supports five movement categories defined in [`dataset_builder.py:L31-37`](file:///home/pavan/Programs/Posture-detection-and-Injury-prediction-using-ML/dataset_builder.py#L31-L37):

1.  **`LUNGE`**: Forehand or backhand low court lunges. Used to analyze knee flexion loading and hip stabilization.
2.  **`SMASH_LANDING`**: The recovery phase after an overhead smash. Analyzes impact absorption forces on knees and ankles.
3.  **`SCISSOR_JUMP`**: Mid-air foot rotation during overhead shots. Analyzes torso rotation symmetry and core balance.
4.  **`CHANGE_OF_DIRECTION`**: Lateral transitions or court recovery steps. Analyzes side-swaying, trunk lean, and knee alignment.
5.  **`FOOTWORK_SPLIT_STEP`**: The initial preparatory bounce step. Evaluates base support width and knee pre-tension angles.

---

## 8. CAMERA AND VIDEO DATA

To eliminate perspective distortion in 2D angle calculations, recording parameters are standardized:

*   **Angle**: Fixed side-view orientation relative to the court baseline/sideline.
*   **Camera Height**: 1.2 – 1.5 meters (matches player torso height to minimize parallax errors).
*   **Framing**: The entire court lane must be visible; the player's full body must remain inside the image bounds to extract lower leg and wrist points.
*   **MediaPipe Coordinate System**: Coordinates are returned in normalized space:
    *   `x`: Horizontal position relative to width, `0.0` (left) to `1.0` (right).
    *   `y`: Vertical position relative to height, `0.0` (top) to `1.0` (bottom).
    *   `z`: Depth coordinate representing distance relative to hips center.
    *   `visibility`: Confidence score indicating point occlusion probability.
*   **Preservation**: Raw coordinates $(x,y,z)$ and visibility values are exported to the landmarks folder for every frame in the sequence.

---

## 9. POSE ESTIMATION WORKFLOW

Pose tracking processes images frame-by-frame:

```text
BGR Video Frame ──► RGB Convert ──► MediaPipe Task ──► Joint Coordinates ──► Angle Calculations
```

*   **MediaPipe Model**: Google Tasks Pose Landmarker (Lite version, configured via `pose_landmarker_lite.task`).
*   **Tracking Mode**: Running in `VIDEO` mode with tracking confidence set to `0.5`.
*   **Capacity**: Restrained to track a single player (`num_poses=1`).
*   **Anatomical Joints Tracked**: Hips, knees, ankles, shoulders, elbows, and wrists.
*   **Data Stored**: Biomechanical feature calculations are stored in the CSV. Raw landmarks are exported to JSON. Individual frame image dumps are **not** stored to keep memory requirements low.

---

## 10. BIOMECHANICAL FEATURE EXTRACTION

The system calculates 2D joint angles inside the camera plane. Calculations are defined in [`pose_analysis.py`](file:///home/pavan/Programs/Posture-detection-and-Injury-prediction-using-ML/pose_analysis.py):

| Feature | Input Landmarks | Biomechanical Meaning | Application |
| :--- | :--- | :--- | :--- |
| **Knee Flexion (Left/Right)** | Hip (23/24), Knee (25/26), Ankle (27/28) | Degree of leg bending. | Measures joint loading during landing and lunges. |
| **Elbow Extension (Left/Right)** | Shoulder (11/12), Elbow (13/14), Wrist (15/16) | Degree of elbow opening. | Evaluates swing extension during clears and smashes. |
| **Torso Lean (Trunk Tilt)** | Shoulder mid-point, Hip mid-point | Degrees of upper body deviation from vertical axis. | Identifies spinal load and core stabilization issues. |

### Mathematical Formula
The angle is calculated using the dot product vectors $\vec{ba}$ (joint to start point) and $\vec{bc}$ (joint to end point):

$$\theta = \left| \text{atan2}(c_y - b_y, c_x - b_x) - \text{atan2}(a_y - b_y, a_x - b_x) \right| \times \frac{180}{\pi}$$

Angles exceeding $180^\circ$ are normalized to $360^\circ - \theta$.

---

## 11. RAW LANDMARK STORAGE

*   **Purpose**: Raw coordinates $(x, y, z)$ and visibility values are saved to enable recalculating features later. If a collaborator wants to test new metrics (e.g., hip-to-shoulder alignment ratios), they can compute them from the JSON files without running MediaPipe inference again.
*   **Directory**: `data/landmarks/{sequence_id}.json`
*   **Format**: JSON dictionary capturing sequence metadata and frame arrays.

```json
{
  "sequence_id": "seq_lunge_test_0.0to3.2",
  "video_name": "lunge_test.mp4",
  "start_time": 0.0,
  "end_time": 3.2,
  "frames": [
    {
      "frame_index": 12,
      "timestamp_sec": 0.4,
      "landmarks": [
        { "x": 0.54, "y": 0.32, "z": -0.12, "visibility": 0.99, "presence": 0.99 },
        ...
      ]
    }
  ]
}
```

---

## 12. DATASET.CSV

The database layout in `data/dataset.csv` catalogs sequence-level summaries:

| Column | Data Type | Description | Example |
| :--- | :--- | :--- | :--- |
| **`sequence_id`** | Text | Unique identifier | `seq_lunge_1_0.0to3.2_178712` |
| **`video_id`** | Text | Source video file | `lunge_1.mp4` |
| **`movement_type`** | Text | Preset class category | `LUNGE` |
| **`frame_start`** | Integer | Starting frame index | `0` |
| **`frame_end`** | Integer | Ending frame index | `96` |
| **`duration`** | Real | Length of sequence in seconds | `3.20` |
| **`left_knee_min`** | Real | Minimum flexion reached | `84.50` |
| **`left_knee_max`** | Real | Maximum extension reached | `168.20` |
| **`right_knee_min`** | Real | Minimum flexion reached | `112.40` |
| **`right_knee_max`** | Real | Maximum extension reached | `162.30` |
| **`left_elbow_min`** | Real | Minimum flexion reached | `142.10` |
| **`left_elbow_max`** | Real | Maximum extension reached | `175.50` |
| **`right_elbow_min`** | Real | Minimum flexion reached | `98.30` |
| **`right_elbow_max`** | Real | Maximum extension reached | `168.40` |
| **`trunk_angle_min`** | Real | Minimum torso lean tilt | `3.20` |
| **`trunk_angle_max`** | Real | Maximum torso lean tilt | `32.40` |
| **`risk_label`** | Text | Assigned target class label | `MODERATE` |

---

## 13. LABELING WORKFLOW

To keep the labeling process simple, labels represent a single consensus score:

```text
Movement Sequence Clip ──► Experts Video Review ──► Joint Angle Inspection ──► Consensus Agreement ──► Save Risk Label (LOW/MODERATE/HIGH)
```

*   **Risk Scores**: Labeling reflects posture-related risk metrics:
    *   **`LOW`**: Nominal range movements showing aligned knees and trunk.
    *   **`MODERATE`**: Shows deep knee loads or trunk tilts.
    *   **`HIGH`**: Severe posture deviations (e.g. knee flexions $< 80^\circ$, torso lean $> 40^\circ$).
*   The label is logged in `dataset.csv` as `risk_label`.

---

## 14. MACHINE LEARNING TRAINING WORKFLOW

The training pipeline in [`train.py`](file:///home/pavan/Programs/Posture-detection-and-Injury-prediction-using-ML/train.py) processes the dataset as follows:

```text
data/dataset.csv ──► Missing-value Drop ──► Group-based Split (by video_id) ──► StandardScaler ──► Random Forest ──► Export Pickle
```

*   **Model Type**: Random Forest Classifier (n_estimators=100, max_depth=6).
*   **Rationale**: Random Forest trains fast, handles non-linear angle boundary combinations, performs well on tabular datasets, and runs locally on standard CPUs.
*   **Target Label**: `risk_label` (LOW, MODERATE, HIGH).
*   **Features Used**: All 10 joint min/max statistics.
*   **Imbalance Strategy**: Uses `class_weight="balanced"` to adjust for class distributions during training.
*   **Model Export**: Outputs a unified pipeline (Scaler + Model) to `models/injury_risk_model.pkl`.
*   **Status**: 🟡 **The pipeline code is complete**, but training awaits complete dataset generation.

---

## 15. DATA SPLITTING AND LEAKAGE PREVENTION

*   **Data Leakage Risk**: Sequential frames in video recordings are highly correlated. If individual sequences from the same recording session (e.g., Lunge 1 and Lunge 2 from the same player video) are split randomly into training and testing sets, the model will evaluate on data it has already seen, inflating evaluation scores.
*   **Prevention**: The script [`train.py`](file:///home/pavan/Programs/Posture-detection-and-Injury-prediction-using-ML/train.py) implements **Group-Based Splitting**. It groups rows by `video_id`. Splitting splits the *videos* (70% train, 15% val, 15% test) rather than individual rows. This ensures that no movement sequence from a training video file ever appears in the test set.

---

## 16. MODEL INTERFACE

The script [`src/model/interface.py`](file:///home/pavan/Programs/Posture-detection-and-Injury-prediction-using-ML/src/model/interface.py) acts as an abstraction wrapper:

*   **Decoupling**: The interface exposes loading (`load`), prediction (`predict`), and probabilities (`predict_proba`) wrappers. The frontend does not contain hardcoded ML models or input parsing logic.
*   **Flexibility**: The model class reads features as standard dictionary entries, scaling and aligning them internally.
*   **Pipeline Plug-in**: When the real model is exported to `models/injury_risk_model.pkl`, the app loads it automatically, replacing the Demo Mode fallback.

---

## 17. INTEGRATION TEST MODEL

*   **Purpose**: Enables testing the pipeline end-to-end (video upload $\rightarrow$ pose extraction $\rightarrow$ UI gauges $\rightarrow$ SQLite logs) without needing a pre-trained model file.
*   **Implementation**: Written in [`src/model/integration_test_model.py`](file:///home/pavan/Programs/Posture-detection-and-Injury-prediction-using-ML/src/model/integration_test_model.py). It uses rule-based biomechanical checks to categorize risk:
    *   **HIGH**: If minimum knee angle $< 80^\circ$ or torso lean $> 40^\circ$.
    *   **MODERATE**: If knee angle $< 110^\circ$ or torso lean $> 25^\circ$.
    *   **LOW**: Default case.
*   **Visual Warning**: Renders `Demo Mode (Integration Test Model)` in the UI, separating it from the final trained model.

---

## 18. INFERENCE PIPELINE

The file [`src/inference/pipeline.py`](file:///home/pavan/Programs/Posture-detection-and-Injury-prediction-using-ML/src/inference/pipeline.py) runs the end-to-end analysis on new video feeds:

```text
Raw video path ──► OpenCV frame loop ──► MediaPipe task (VIDEO mode) ──► Skeleton overlay ──► Feature Min/Max Aggregations ──► Interface Prediction ──► FFmpeg H.264 Conversion ──► Streamlit Video Render
```

1.  **Read Video**: Opens the video and extracts frames between `start_time` and `end_time`.
2.  **Tracking**: Runs MediaPipe landmarker on frames, tracking joint locations.
3.  **Visual Overlays**: Draws green joint circles and blue skeletal lines on the frames.
4.  **Feature Math**: Calculates knee, elbow, and torso lean values on frames.
5.  **Summary Aggregation**: Computes min/max metrics over the sequence timeline.
6.  **Execute Prediction**: Passes metrics to `InjuryRiskModel` to compute risk classification and probabilities.
7.  **Video Compression**: Calls `ffmpeg` in a background subprocess to convert the processed video into H.264 MP4 format.
8.  **Output API**: Returns a dictionary containing success flags, error codes, metrics, risk prediction outputs, and the H.264 video path.

---

## 19. WEB APPLICATION

The dashboard is built using Streamlit ([`app/streamlit_app.py`](file:///home/pavan/Programs/Posture-detection-and-Injury-prediction-using-ML/app/streamlit_app.py)):

*   **Upload Panel**: File uploader accepting standard formats (`mp4`, `avi`, `mov`).
*   **Configurations**: Input fields for Start/End seconds, target FPS, and movement types.
*   **Visual Risk Alert**: Shows the risk classification score in highlighted cards (Green for LOW, Yellow for MODERATE, Red for HIGH).
*   **Angle Gauges Grid**: Displays minimum knee, minimum elbow, and maximum torso lean metrics.
*   **Dynamic Explanations**: Lists biomechanical reasons explaining the risk score (e.g., flagging knee load limits or torso tilts).
*   **Model Status Indicator**: Shows `✅ Real Model Loaded` if a trained model exists, or displays `⚠️ Real Model Not Installed` with instructions to use Demo Mode.
*   **History Logs**: Displays a table of previous analyses pulled from SQLite (`data/history.db`).
*   **Input Validation**: Catches and displays errors for invalid timing ranges or cases where no person is detected.

---

## 20. WEB UI USER JOURNEY

```text
1. Open Streamlit App ──► 2. Upload video file ──► 3. Set timing boundaries (start/end)
                                                                 │
                                                                 ▼
6. View skeletal overlays ◄── 5. Run inference pipeline ◄── 4. Trigger "Analyze Video"
           │
           ▼
7. Read risk level alert (Green/Yellow/Red) ──► 8. Inspect biomechanical gauges 
                                                                 │
                                                                 ▼
10. View database history log ◄── 9. Read dynamic posture warnings
```

---

## 21. DATABASE / HISTORY

*   **Technology**: SQLite (locally stored in `data/history.db`).
*   **Purpose**: Log analysis results so players and coaches can compare metrics across sessions.
*   **Columns stored**: Timestamp, video name, movement type, risk prediction score, and joint angle metrics.
*   **Streamlit Integration**: Query results are loaded directly into a Pandas dataframe and rendered in the sidebar.

---

## 22. VIDEO OUTPUT

*   **Drawing Engine**: OpenCV rendering engine.
*   **Skeletal Overlays**: Draws key joints (hips, knees, ankles, shoulders, elbows, wrists) and links them with bone lines.
*   **H.264 Codec Encoding**: The output is written to a temporary video file using `mp4v` codec, then converted to H.264 format using FFmpeg:
    ```bash
    ffmpeg -y -i input.mp4 -vcodec libx264 -pix_fmt yuv420p -profile:v baseline -level 3.0 output.mp4
    ```
*   **Rationale**: H.264 is widely supported by modern browsers, enabling direct playback in the Streamlit UI.

---

## 23. TESTING

The verification test suite was executed in the workspace:

```bash
.venv/bin/python -m unittest discover -s tests
```

### Ran Test Cases
1.  **`test_angle_calculation`**: Verified the mathematical angle calculation formulas against $90^\circ$ and $180^\circ$ vector baselines. **PASSED**.
2.  **`test_integration_test_model`**: Verified the Demo Mode rule transitions (LOW, MODERATE, HIGH) and output distributions. **PASSED**.
3.  **`test_pipeline_on_dummy_video`**: Generates a mock black video, runs the inference pipeline, and verifies that it handles the absence of pose detection gracefully. **PASSED**.

```text
Ran 3 tests in 0.689s
OK
```

### CLI Verification Tests
*   `dataset_builder.py --help`: CLI arguments and inputs parsed successfully. **PASSED**.
*   `train.py --help`: Training pipeline CLI parameters initialized successfully. **PASSED**.
*   `app/streamlit_app.py`: Launches the server successfully at `http://localhost:8501`. **PASSED**.

---

## 24. CURRENT PROJECT FILE STRUCTURE

```text
Posture-detection-and-Injury-prediction-using-ML/
├── app/
│   └── streamlit_app.py        # Streamlit web application dashboard interface
│                               # Depends on: src/inference/pipeline.py
│                               # Outputs: data/history.db
├── data/
│   ├── clips/                  # Generated movement clip files (H.264)
│   ├── landmarks/              # Raw landmarks coordinates files (JSON)
│   └── dataset.csv             # Primary training CSV dataset
├── models/
│   └── injury_risk_model.pkl   # Serialized trained model (Future deployment target)
├── src/
│   ├── inference/
│   │   └── pipeline.py         # Primary video processing and overlays runner
│   │                           # Depends on: pose_analysis.py, src/model/interface.py
│   └── model/
│       ├── integration_test_model.py # Demo Mode threshold classifier
│       └── interface.py        # Model loading abstraction wrapper
├── tests/
│   └── test_pipeline.py        # Unit test suite
├── dataset_builder.py          # Sequence annotation script
│                               # Depends on: pose_analysis.py
├── pose_analysis.py            # Landmark geometry and angle mathematics
├── train.py                    # Training template (Random Forest Classifier)
└── requirements.txt            # Python environment requirements
```

---

## 25. DEPENDENCY FLOW

The diagram below outlines the dependency relationships within the project.

### Pipeline Dependency Flow
```text
[ OpenCV / Video Capture ]
            │
            ▼
[ MediaPipe Pose Landmarker ]
            │
            ▼
  [ pose_analysis.py ] (Angle geometry calculations)
            │
      ┌─────┴───────────────────────────────────┐
      ▼                                         ▼
[ dataset_builder.py ] (Data collection)  [ src/inference/pipeline.py ] (Inference pipeline)
```

### Web Application Dependency Flow
```text
          [ app/streamlit_app.py ] (Streamlit Dashboard)
                        │
                        ▼
          [ src/inference/pipeline.py ] (Inference Pipeline)
                        │
                        ▼
          [ src/model/interface.py ] (Model Abstraction Loader)
                        │
            ┌───────────┴───────────┐
            ▼                       ▼
  [ IntegrationTestModel ]  [ Trained ML Model ] (models/injury_risk_model.pkl)
```

---

## 26. WHAT IS COMPLETE RIGHT NOW?

| Component | Status | Explanation |
| :--- | :---: | :--- |
| **Pose Detection** | ✅ Complete | Uses MediaPipe to track joints frame-by-frame. |
| **Feature Extraction** | ✅ Complete | Computes knees/elbows flexion and torso lean. |
| **Dataset Builder** | ✅ Complete | Sequence-based clipping tool is operational. |
| **Data Storage** | ✅ Complete | Saves aggregated features to CSV and raw coordinates to JSON. |
| **Training Pipeline** | 🟡 Partially ready | `train.py` code is complete, but awaits dataset entries. |
| **Real Model** | ⏳ Pending | Awaiting data collection and training. |
| **Inference** | ✅ Complete | `pipeline.py` executes tracking and inference (Demo Mode fallback). |
| **Web UI** | ✅ Complete | Streamlit application dashboard is operational. |
| **History Logs** | ✅ Complete | SQLite logging is configured and integrated. |
| **Testing Suite** | ✅ Complete | Math, model, and pipeline unit tests run and pass. |
| **Deployment** | ⏳ Pending | Runs locally; containerization (Docker) is planned. |

---

## 27. WHAT REMAINS?

The following roadmap outlines the remaining steps to move from prototype to final deployment:

1.  **Video Collection**: Record badminton clips (Smash Landing, Lunge, Split Step) following the side-view recording protocol.
2.  **Dataset Labeling**: Run `dataset_builder.py` on the recorded videos to compile a baseline dataset in `data/dataset.csv`.
3.  **Verify Dataset**: Check feature distributions in the CSV for outliers or errors.
4.  **Train the Model**: Run `train.py` to train the Random Forest Classifier on the labeled dataset.
5.  **Evaluate Performance**: Review validation and testing metrics (accuracy, precision, recall, confusion matrix) in the training logs.
6.  **Deploy the Model**: Place the exported `injury_risk_model.pkl` in the `models/` directory.
7.  **Run the App**: Launch Streamlit to load the trained model.
8.  **Final Verification**: Test the web interface using new, unseen videos.

---

## 28. FINAL END-TO-END ARCHITECTURE

The diagram below shows the complete system architecture, mapping the data path from raw video input to final prediction output.

```text
                           🏸 Raw Video (.mp4)
                                    │
                                    ▼
                         [ OpenCV Video Capture ]
                                    │
                                    ▼
                        [ MediaPipe Pose Task ]
                                    │
                                    ▼
                        [ 33 Skeletal Joints ]
                                    │
                                    ▼
                         [ Biomechanical Math ]
                   (Flexion angles & torso tilt)
                                    │
                                    ▼
                        [ Sequence Aggregations ]
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
            [ DATA COLLECTION ]             [ INFERENCE RUNNER ]
           (dataset_builder.py)           (src/inference/pipeline.py)
                    │                               │
                    ▼                               ▼
            [ dataset.csv ]                [ Model Abstraction ]
                    │                               │
                    ▼                               ▼
            [ training.py ]               [ Trained ML Pipeline ]
                    │                      (injury_risk_model.pkl)
                    │                               │
                    └───────────────┬───────────────┘
                                    ▼
                          [ Predict Risk Label ]
                         (LOW / MODERATE / HIGH)
                                    │
                                    ▼
                         [ Streamlit Dashboard ]
                                    │
                        ┌───────────┴───────────┐
                        ▼                       ▼
            [ Skeletal Video Overlay ]  [ Metrics & explanations ]
```

---

## 29. INDUSTRY COLLABORATION SUMMARY

To advance the project from prototype to final deployment, collaboration with industry partners in the following areas would be beneficial:

*   **Video Data Collection**: Access to players to record high-quality badminton movement sequences.
*   **Professional Courts**: Access to indoor courts with controlled lighting to record standardized videos.
*   **Coaching & Clinical Domain Experts**: Collaboration with coaches and sports physiotherapists to establish labeling guidelines and validate biomechanical risk thresholds.
*   **Camera Equipment**: Access to recording equipment (tripods, stands, high-speed cameras) to capture clean, high-framerate clips.

---

## 30. FINAL 30-SECOND EXPLANATION

> "We are building an AI-based badminton posture and injury-risk assessment system. It uses computer vision (MediaPipe) to track a player's body landmarks from side-angle court videos, extracts key joint angles like knee flexion and torso lean, and aggregates them into sequence-level metrics. These biomechanical features are classified as Low, Moderate, or High injury-risk using a machine learning model. The full-stack Streamlit web app, local SQLite history logging, unit tests, and training code are complete and running. We are currently seeking collaborators (coaches, players, and sports experts) to help record and annotate movement sequences so we can train and evaluate the final machine learning model."
