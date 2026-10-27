"""
Badminton AI - Video Movement Analysis & Biomechanical Injury-Risk Dashboard
Streamlit-based sports biomechanics interface supporting full video-to-clips analysis,
3-class movement classification (Lunge, Smash Landing, Change of Direction),
Rubric v0.3 deterministic injury-risk screening, and structured manifest exports.
"""

import os
import sys
import json
import hashlib
import math
import sqlite3
import tempfile
import time
from datetime import datetime
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.video.validator import validate_video
from src.video.video_pipeline import VideoProcessingPipeline
from src.classification.categories import (
    MOVEMENT_CATEGORIES,
    ACTIVE_RISK_RUBRIC_CATEGORIES,
    DEFERRED_RISK_RUBRIC_CATEGORIES,
    CATEGORY_METADATA,
)
from src.classification.predictor import MovementClassifier
from src.risk.rubric import RUBRIC_VERSION

# Database & file paths
BASE_DIR = str(root_dir)
DB_PATH = os.path.join(BASE_DIR, "data", "history.db")
MODEL_REGISTRY_PATH = os.path.join(BASE_DIR, "models", "model_registry.json")

# Candidate model paths in priority order
_mv_reviewed_161 = os.path.join(BASE_DIR, "models", "movement_classifier_reviewed_161_rf.pkl")
_mv_baseline = os.path.join(BASE_DIR, "models", "movement_classifier.pkl")
_inj_model = os.path.join(BASE_DIR, "models", "injury_risk_model.pkl")

_configured_model = os.environ.get("BADMINTON_MOVEMENT_MODEL_PATH", "").strip()
CUSTOM_MODEL_PATH_ACTIVE = bool(_configured_model)

# Resolve default model
if CUSTOM_MODEL_PATH_ACTIVE:
    PRIMARY_MODEL_PATH = _configured_model
elif os.path.exists(_mv_reviewed_161):
    PRIMARY_MODEL_PATH = _mv_reviewed_161
elif os.path.exists(_mv_baseline):
    PRIMARY_MODEL_PATH = _mv_baseline
else:
    PRIMARY_MODEL_PATH = _inj_model


def format_angle(value) -> str:
    """Formats a numeric angle in degrees or returns N/A."""
    try:
        angle = float(value)
    except (TypeError, ValueError):
        return "N/A"
    return f"{angle:.1f}°" if math.isfinite(angle) else "N/A"


def init_db():
    """Initializes the local SQLite database for run history."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            video_name TEXT,
            movement_type TEXT,
            risk_label TEXT,
            left_knee_min REAL,
            right_knee_min REAL,
            left_elbow_min REAL,
            right_elbow_min REAL,
            trunk_angle_max REAL,
            manifest_json TEXT
        )
    """)
    conn.commit()
    conn.close()


def save_run_to_history(video_name: str, events: list):
    """Saves run summary and serialized manifest to SQLite history."""
    if not events:
        return
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    ev0 = events[0]
    feats = ev0.get("features", {}) or {}

    cursor.execute("""
        INSERT INTO history (
            timestamp, video_name, movement_type, risk_label,
            left_knee_min, right_knee_min, left_elbow_min, right_elbow_min, trunk_angle_max,
            manifest_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        now,
        video_name,
        ev0.get("movement_category", "UNCERTAIN"),
        (f"{ev0.get('risk_category')} ({ev0.get('risk_score'):.1f})"
         if ev0.get("risk_score") is not None and ev0.get("risk_category") and ev0.get("risk_category") != "UNASSESSED"
         else ev0.get("risk_level", "UNASSESSED")),
        feats.get("left_knee_min", 180.0),
        feats.get("right_knee_min", 180.0),
        feats.get("left_elbow_min", 180.0),
        feats.get("right_elbow_min", 180.0),
        feats.get("trunk_angle_max"),
        json.dumps(events),
    ))
    conn.commit()
    conn.close()


def get_history_df() -> pd.DataFrame:
    """Retrieves recent run history."""
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query(
            "SELECT id, timestamp, video_name, movement_type, risk_label, manifest_json FROM history ORDER BY id DESC LIMIT 15",
            conn,
        )
    except Exception:
        df = pd.DataFrame()
    conn.close()
    return df


def clear_history():
    """Clears run history database table."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM history")
    conn.commit()
    conn.close()


# Initialize database
init_db()

# Streamlit Page Config
st.set_page_config(
    page_title="Badminton AI | Biomechanical Risk Screening",
    page_icon=":material/sports_tennis:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------
# Sidebar: System Controls & Configuration
# ---------------------------------------------------------
st.sidebar.title(":material/tune: System Control")

# Model Registry Status
has_registry = os.path.exists(MODEL_REGISTRY_PATH)
reg_info = {}
if has_registry:
    try:
        with open(MODEL_REGISTRY_PATH, "r") as rf:
            reg_info = json.load(rf)
    except Exception:
        pass

# Model selection options
model_choices = {}
if os.path.exists(_mv_reviewed_161):
    active_metrics = reg_info.get("active_metrics", {})
    f1 = active_metrics.get("group_kfold_macro_f1", reg_info.get("validation_macro_f1", 0.58))
    model_choices[f"Reviewed RF (161 Events, F1: {f1:.2f})"] = _mv_reviewed_161
if os.path.exists(_mv_baseline):
    model_choices["Baseline RF (34 Samples)"] = _mv_baseline
model_choices["Heuristic Kinematic Baseline (No ML)"] = None

selected_model_label = st.sidebar.selectbox(
    "Active Movement Model",
    options=list(model_choices.keys()),
    index=0,
    help="Select the trained movement classifier or use heuristic kinematics rules.",
)
active_model_path = model_choices[selected_model_label]

if active_model_path and os.path.exists(active_model_path):
    st.sidebar.success(f":material/check_circle: Loaded: **{Path(active_model_path).name}**")
else:
    st.sidebar.info(":material/info: Using Heuristic Kinematic Rules")

st.sidebar.markdown("---")
st.sidebar.subheader(":material/settings: Pipeline Settings")

sample_fps = st.sidebar.slider(
    "Sampling Rate (FPS)",
    min_value=5.0,
    max_value=30.0,
    value=15.0,
    step=1.0,
    help="Higher FPS gives finer temporal resolution but takes longer to process.",
)

camera_viewpoint = st.sidebar.selectbox(
    "Camera Viewpoint (Metadata)",
    ["side_90deg (Standard Lateral)", "diagonal_45deg", "front_view", "rear_view", "broadcast_free"],
    help="Lateral 90° view is optimal for planar sagittal joint kinematics. Oblique angles suffer perspective foreshortening.",
)

include_extended = st.sidebar.checkbox(
    "Include Deferred Classes (Disabled)",
    value=False,
    disabled=True,
    help="Detection of retired classes (SCISSOR_JUMP, FOOTWORK_SPLIT_STEP) is disabled under Rubric v0.3.",
)

# Run History in Sidebar
st.sidebar.markdown("---")
st.sidebar.subheader(":material/history: Recent Runs")
hist_df = get_history_df()
if not hist_df.empty:
    st.sidebar.dataframe(
        hist_df[["timestamp", "video_name", "movement_type", "risk_label"]],
        hide_index=True,
    )
    if st.sidebar.button("Clear History", icon=":material/delete:"):
        clear_history()
        st.rerun()
else:
    st.sidebar.caption("No previous analysis records found.")


# ---------------------------------------------------------
# Main Page: Header & Academic Framing
# ---------------------------------------------------------
st.title("🏸 Badminton AI — Biomechanical Movement Analysis & Injury-Risk Screening")
st.caption(
    "Computer-vision movement proposal and deterministic biomechanical posture screening "
    "for athletic badminton maneuvers (Lunge, Smash Landing, Change of Direction) based on Rubric v0.3."
)

# Academic Research Prototype Disclaimer Banner
st.info(
    ":material/shield_with_heart: **Academic Research Prototype**: "
    "This system is an engineering screening prototype based on published sports biomechanics literature. "
    "It evaluates kinematic posture deviations (knee flexion, torso lean, bilateral asymmetry). "
    "It is **not** a clinical diagnostic tool or medical injury predictor. "
    "A risk score of 70 indicates elevated postural deviation from reference mechanics, not a 70% probability of injury.",
    icon=None,
)

# ---------------------------------------------------------
# Step 1: Video Upload & Inspection
# ---------------------------------------------------------
st.markdown("### 📤 1. Video Input & Inspection")

with st.container(border=True):
    col_upload, col_meta = st.columns([3, 2])

    with col_upload:
        uploaded_file = st.file_uploader(
            "Upload badminton movement footage (lateral ~90° view recommended)",
            type=["mp4", "avi", "mov", "mkv"],
            help="Supported formats: MP4, AVI, MOV, MKV. For accurate sagittal joint angles, use lateral tripod footage.",
        )

    saved_video_path = None
    if uploaded_file is not None:
        uploaded_bytes = uploaded_file.getvalue()
        upload_key = (uploaded_file.name, hashlib.sha256(uploaded_bytes).hexdigest())
        if st.session_state.get("_active_upload_key") != upload_key:
            st.session_state.pop("pipeline_result", None)
            st.session_state.pop("video_name", None)
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uploaded_file.name}")
            tfile.write(uploaded_bytes)
            tfile.close()
            st.session_state["_saved_video_path"] = tfile.name
            st.session_state["_active_upload_key"] = upload_key
        saved_video_path = st.session_state["_saved_video_path"]

        v_meta = validate_video(saved_video_path)
        with col_meta:
            st.markdown("**Video Inspection Summary:**")
            if v_meta["is_valid"]:
                st.badge("Format Valid", icon=":material/check_circle:", color="green")
                st.write(f"• **Duration:** `{v_meta['duration']:.2f} s` ({v_meta['frame_count']} frames)")
                st.write(f"• **Resolution:** `{v_meta['width']} × {v_meta['height']}` at `{v_meta['fps']} FPS`")
                st.write(f"• **Codec:** `{v_meta['codec']}` ({v_meta['file_size_mb']:.1f} MB)")
            else:
                st.error(f"❌ Invalid Video File: {v_meta['error']}")

        if v_meta["is_valid"]:
            analyze_col1, analyze_col2 = st.columns([2, 3])
            with analyze_col1:
                run_btn = st.button(
                    "Analyze Movement Biomechanics",
                    type="primary",
                    icon=":material/analytics:",
                )
            if run_btn:
                st.session_state.pop("pipeline_result", None)
                progress_bar = st.progress(0.0)
                status_text = st.empty()

                def update_cb(pct, text):
                    progress_bar.progress(pct)
                    status_text.text(f":material/hourglass_empty: {text}")

                output_dir = os.path.join(BASE_DIR, "data", "processed", Path(uploaded_file.name).stem)
                pipeline = VideoProcessingPipeline(model_path=active_model_path)

                with st.spinner("Processing video through pose estimation, candidate detection, and biomechanical evaluation..."):
                    t_start = time.time()
                    pipeline_result = pipeline.process_video(
                        video_path=saved_video_path,
                        output_dir=output_dir,
                        sample_fps=sample_fps,
                        include_deferred_categories=include_extended,
                        progress_callback=update_cb,
                    )
                    t_elapsed = time.time() - t_start

                if pipeline_result["success"]:
                    st.success(f"Analysis complete in {t_elapsed:.1f}s. Detected {pipeline_result['total_events']} movement event(s).")
                    st.session_state["pipeline_result"] = pipeline_result
                    st.session_state["video_name"] = uploaded_file.name
                    save_run_to_history(uploaded_file.name, pipeline_result["events"])
                    st.rerun()
                else:
                    st.error(f"Pipeline processing failed: {pipeline_result.get('error', 'Unknown error')}")

    elif st.session_state.get("_active_upload_key") is not None:
        st.session_state.pop("pipeline_result", None)
        st.session_state.pop("video_name", None)
        st.session_state.pop("_active_upload_key", None)
        st.session_state.pop("_saved_video_path", None)


# ---------------------------------------------------------
# Step 2: Actionable Screening Results (Primary View)
# ---------------------------------------------------------
if "pipeline_result" in st.session_state and st.session_state["pipeline_result"].get("success"):
    res = st.session_state["pipeline_result"]
    events = res.get("events", [])

    st.markdown("### 📊 2. Biomechanical Screening Results")

    if not events:
        st.warning(
            ":material/warning: No athletic movement candidates met detection thresholds in this video. "
            "Ensure the subject is clearly visible with full body landmarks tracking in a lateral perspective."
        )
    else:
        # Event Selector
        if len(events) > 1:
            event_labels = [
                f"Event {i+1}: {ev.get('movement_category', 'UNCERTAIN')} [{ev.get('start_time', 0.0):.2f}s – {ev.get('end_time', 0.0):.2f}s]"
                for i, ev in enumerate(events)
            ]
            selected_idx = st.selectbox(
                "Select Detected Movement Event:",
                range(len(events)),
                format_func=lambda i: event_labels[i],
            )
        else:
            selected_idx = 0

        active_ev = events[selected_idx]
        mv_cat = active_ev.get("movement_category", "UNCERTAIN")
        risk_lvl = active_ev.get("risk_level", "UNASSESSED")
        score_val = active_ev.get("risk_score")
        risk_cat = active_ev.get("risk_category", "UNASSESSED")
        risk_contribs = active_ev.get("risk_contributions", [])
        feats = active_ev.get("features", {}) or {}

        # -------------------------------------------------
        # Actionable Overview Cards: Movement Class & Risk Score
        # -------------------------------------------------
        col_mv, col_risk = st.columns(2)

        with col_mv:
            with st.container(border=True):
                st.markdown("#### :material/directions_run: Movement Identification")
                st.header(mv_cat.replace("_", " "))

                # Status & Detection Metadata
                if mv_cat in ("LUNGE", "SMASH_LANDING", "CHANGE_OF_DIRECTION"):
                    st.badge("Target Athletic Maneuver", icon=":material/verified:", color="green")
                else:
                    st.badge("Unconfirmed / Non-Target Event", icon=":material/help:", color="orange")

                st.write(
                    f"• **Time Span:** `{active_ev.get('start_time', 0.0):.2f}s – {active_ev.get('end_time', 0.0):.2f}s` "
                    f"(`{active_ev.get('end_time', 0.0) - active_ev.get('start_time', 0.0):.2f}s` duration)"
                )
                st.write(f"• **Localization Mode:** `{active_ev.get('localization_method', 'Candidate Detector')}`")
                st.write(f"• **Classification Source:** `{active_ev.get('prediction_source', 'ML Model')}`")

                cls_note = active_ev.get("classification_note")
                if cls_note:
                    st.caption(f":material/info: {cls_note}")

        with col_risk:
            with st.container(border=True):
                st.markdown("#### :material/health_and_safety: Biomechanical Screening Risk")

                if risk_lvl == "PROVISIONAL" and score_val is not None:
                    score_col, badge_col = st.columns([1, 1])
                    with score_col:
                        st.metric(
                            label="Posture Risk Score",
                            value=f"{score_val:.1f} / 100",
                        )
                    with badge_col:
                        if risk_cat == "LOW":
                            st.badge("LOW RISK (< 30.0)", icon=":material/check_circle:", color="green")
                            st.caption("Postural indicators fall within reference biomechanical ranges.")
                        elif risk_cat == "MODERATE":
                            st.badge("MODERATE RISK (30.0 – 59.9)", icon=":material/warning:", color="orange")
                            st.caption("Mild-to-moderate kinematic deviation in knee cushioning or torso lean.")
                        else:
                            st.badge("HIGH RISK (≥ 60.0)", icon=":material/dangerous:", color="red")
                            st.caption("Elevated joint stress indicators: stiff deceleration or excessive torso lean.")

                    # Score bar
                    norm_score = min(1.0, max(0.0, score_val / 100.0))
                    st.progress(norm_score)
                    st.caption("Risk Spectrum: 0–29.9 Low Concern | 30–59.9 Moderate Concern | 60–100 High Concern")

                else:
                    st.metric(label="Posture Risk Score", value="UNASSESSED")
                    st.badge("Assessment: UNASSESSED", icon=":material/info:", color="gray")
                    st.caption(
                        "**Honest Science Notice:** A numerical screening score is withheld when landmark "
                        "tracking quality is low (< 0.45), joint measurements are incomplete, or the maneuver is outside "
                        "the active Rubric v0.3 categories. Missing data is **never** substituted with zero risk."
                    )

        # -------------------------------------------------
        # Visual Evidence: Keyframe & Video Previews
        # -------------------------------------------------
        with st.container(border=True):
            st.markdown("#### :material/videocam: Visual Evidence & Annotated Keyframe")

            tab_keyframe, tab_clip, tab_full = st.tabs([
                ":material/photo_camera: Representative Keyframe",
                ":material/movie: Event Sub-Clip",
                ":material/video_library: Full Tracking Video",
            ])

            with tab_keyframe:
                key_path = active_ev.get("representative_frame")
                if key_path and os.path.exists(key_path):
                    rep_t = active_ev.get("representative_timestamp")
                    rep_idx = active_ev.get("representative_frame_index")
                    cap_text = f"Keyframe at source timestamp {rep_t:.2f}s (frame {rep_idx})" if rep_t is not None else "Representative posture keyframe"
                    st.image(key_path, caption=cap_text)
                    reason = active_ev.get("keyframe_selection_reason")
                    if reason:
                        st.caption(f":material/search: Selection Criteria: {reason}")
                else:
                    st.info("Representative keyframe image not generated for this event.")

            with tab_clip:
                clip_path = active_ev.get("clip_path")
                if clip_path and os.path.exists(clip_path):
                    st.video(clip_path)
                else:
                    st.caption("Isolated event sub-clip unavailable.")

            with tab_full:
                ann_vid = res.get("annotated_video_path")
                if ann_vid and os.path.exists(ann_vid):
                    st.video(ann_vid)
                else:
                    st.caption("Full tracking video overlay unavailable.")

        # -------------------------------------------------
        # Contributing Biomechanical Indicators Breakdown
        # -------------------------------------------------
        with st.container(border=True):
            st.markdown("#### :material/table_chart: Contributing Biomechanical Indicators (Rubric v0.3)")

            if risk_contribs and risk_lvl == "PROVISIONAL":
                rows = []
                for c in risk_contribs:
                    rows.append({
                        "Biomechanical Indicator": c.get("name", "Unknown"),
                        "Measured Value": f"{c.get('measured_value', 0.0):.1f}{c.get('unit', '°')}",
                        "Contribution": f"{c.get('points', 0.0):.1f} / {c.get('max_points', 0.0):.0f} pts",
                        "Biomechanical Rationale & Clinical Direction": c.get("explanation", c.get("direction_of_concern", "")),
                    })
                st.dataframe(pd.DataFrame(rows), hide_index=True)
            else:
                st.info(
                    "Individual point contributions are unavailable because this event is UNASSESSED. "
                    "Review kinematics and video quality below."
                )

            # Measured Joint Angles Grid
            st.markdown("##### Measured Joint Kinematics")
            lk = feats.get("left_knee_min")
            rk = feats.get("right_knee_min")
            le = feats.get("left_elbow_min")
            re = feats.get("right_elbow_min")
            tm = feats.get("trunk_angle_max")

            def finite_or_none(v):
                try:
                    num = float(v)
                    return num if math.isfinite(num) else None
                except (TypeError, ValueError):
                    return None

            valid_knees = [v for v in (finite_or_none(lk), finite_or_none(rk)) if v is not None]
            valid_elbows = [v for v in (finite_or_none(le), finite_or_none(re)) if v is not None]

            asym = abs(finite_or_none(lk) - finite_or_none(rk)) if finite_or_none(lk) is not None and finite_or_none(rk) is not None else None

            k1, k2, k3 = st.columns(3)
            with k1:
                with st.container(border=True):
                    st.metric("Min Knee Flexion", format_angle(min(valid_knees) if valid_knees else None))
                    st.caption(f"Left: {format_angle(lk)} | Right: {format_angle(rk)}")
                    st.caption(f"Bilateral Knee Asymmetry: {format_angle(asym)}")

            with k2:
                with st.container(border=True):
                    st.metric("Min Elbow Angle", format_angle(min(valid_elbows) if valid_elbows else None))
                    st.caption(f"Left: {format_angle(le)} | Right: {format_angle(re)}")

            with k3:
                with st.container(border=True):
                    st.metric("Max Torso Lean", format_angle(tm))
                    st.caption(f"Min Torso Angle: {format_angle(feats.get('trunk_angle_min'))}")

            # Kinematic Observations & Evidence Limitations
            obs_list = active_ev.get("risk_observations", [])
            valid_obs = [ob for ob in obs_list if not ob.startswith("ℹ️ Notice:")]
            if valid_obs:
                st.markdown("**Kinematic Observations:**")
                for ob in valid_obs:
                    st.write(f"• {ob}")

            missing_ev = active_ev.get("risk_missing_evidence")
            if missing_ev:
                st.caption(f":material/info: **Limitations / Missing Evidence:** {missing_ev}")

        # -------------------------------------------------
        # Step 3: Collapsed Diagnostics & Telemetry
        # -------------------------------------------------
        with st.expander(":material/bug_report: Diagnostics, Classifier Probabilities & Pipeline Telemetry", expanded=False):
            diag_col1, diag_col2 = st.columns(2)

            with diag_col1:
                st.markdown("##### Classifier Probability Distribution")
                probs = active_ev.get("probabilities", {}) or {}
                if probs:
                    for c_name, p_val in sorted(probs.items(), key=lambda x: -x[1]):
                        st.progress(float(p_val), text=f"{c_name}: {p_val*100:.1f}%")
                else:
                    st.caption("No class probability distribution available for this event.")

                det_metrics = active_ev.get("detector_metrics", {})
                if det_metrics:
                    st.markdown("##### Candidate Detector Telemetry")
                    for k, v in det_metrics.items():
                        st.write(f"• **{k.replace('_', ' ').title()}:** `{v}`")

            with diag_col2:
                st.markdown("##### Model & Tracking Quality Telemetry")
                st.write(f"• **Pose Quality Confidence:** `{active_ev.get('pose_quality', 0.0):.3f}`")
                st.write(f"• **Lower-Body / Trunk Quality:** `{active_ev.get('lower_body_quality', 0.0):.3f}`")
                st.write(f"• **Active Rubric Version:** `{active_ev.get('rubric_version', 'v0.3')}`")
                st.write(f"• **Model Selected:** `{selected_model_label}`")

                if has_registry and reg_info:
                    st.write(f"• **Registered Model Macro F1:** `{reg_info.get('validation_macro_f1', 0.0):.4f}`")
                    st.write(f"• **Target Column:** `{reg_info.get('target_column', 'risk_label')}`")

            st.markdown("---")
            st.markdown("##### Export Structured Results")
            export_c1, export_c2 = st.columns(2)

            with export_c1:
                manifest_p = res.get("manifest_path")
                if manifest_p and os.path.exists(manifest_p):
                    with open(manifest_p, "r") as mf:
                        m_str = mf.read()
                    st.download_button(
                        label="Download Full Manifest (JSON)",
                        icon=":material/download:",
                        data=m_str,
                        file_name=f"manifest_{Path(st.session_state.get('video_name', 'video')).stem}.json",
                        mime="application/json",
                    )

            with export_c2:
                if events:
                    ev_df = pd.DataFrame(events)
                    st.download_button(
                        label="Download Events Summary (CSV)",
                        icon=":material/download:",
                        data=ev_df.to_csv(index=False),
                        file_name=f"events_{Path(st.session_state.get('video_name', 'video')).stem}.csv",
                        mime="text/csv",
                    )
