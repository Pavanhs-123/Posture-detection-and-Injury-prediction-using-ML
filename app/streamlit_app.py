"""
Badminton AI - Video Movement Analysis & Biomechanical Injury-Risk Dashboard
Streamlit-based sports biomechanics interface supporting full video-to-clips analysis,
3-class movement classification (Lunge, Smash Landing, Change of Direction), Rubric v0.3 injury-risk screening, and manifest exports.
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

# Database setup
BASE_DIR = str(root_dir)
DB_PATH = os.path.join(BASE_DIR, "data", "history.db")
MODEL_REGISTRY_PATH = os.path.join(BASE_DIR, "models", "model_registry.json")
# Primary model is the movement category classifier; injury_risk_model.pkl is kept as a compatibility alias
_mv_model = os.path.join(BASE_DIR, "models", "movement_classifier.pkl")
_inj_model = os.path.join(BASE_DIR, "models", "injury_risk_model.pkl")
PRIMARY_MODEL_PATH = _mv_model if os.path.exists(_mv_model) else _inj_model


def format_angle(value):
    try:
        angle = float(value)
    except (TypeError, ValueError):
        return "N/A"
    return f"{angle:.1f}°" if math.isfinite(angle) else "N/A"


def init_db():
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
    if not events:
        return
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Primary summary stats from first or most significant event
    ev0 = events[0]
    feats = ev0.get("features", {})

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
        ev0.get("risk_level", "UNASSESSED"),
        feats.get("left_knee_min", 180.0),
        feats.get("right_knee_min", 180.0),
        feats.get("left_elbow_min", 180.0),
        feats.get("right_elbow_min", 180.0),
        feats.get("trunk_angle_max"),
        json.dumps(events),
    ))
    conn.commit()
    conn.close()


def get_history_df():
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
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM history")
    conn.commit()
    conn.close()


# Initialize database
init_db()

# Streamlit Page Config
st.set_page_config(
    page_title="Badminton AI | Biomechanics & Movement Analysis",
    page_icon="🏸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for research dashboard styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .risk-badge-low {
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
        padding: 6px 12px;
        border-radius: 6px;
        font-weight: 600;
        display: inline-block;
    }
    .risk-badge-mod {
        background-color: #fff3cd;
        color: #856404;
        border: 1px solid #ffeeba;
        padding: 6px 12px;
        border-radius: 6px;
        font-weight: 600;
        display: inline-block;
    }
    .risk-badge-high {
        background-color: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
        padding: 6px 12px;
        border-radius: 6px;
        font-weight: 600;
        display: inline-block;
    }
    .risk-badge-deferred {
        background-color: #e2e3e5;
        color: #383d41;
        border: 1px solid #d6d8db;
        padding: 6px 12px;
        border-radius: 6px;
        font-weight: 600;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)


# Sidebar Configuration
st.sidebar.title("🏸 System Control")

# Check model status
has_registry = os.path.exists(MODEL_REGISTRY_PATH)
has_primary_model = os.path.exists(PRIMARY_MODEL_PATH)

if has_registry:
    with open(MODEL_REGISTRY_PATH, "r") as rf:
        reg_info = json.load(rf)
    selected_name = reg_info.get("selected_model", "Trained Model")
    val_macro = reg_info.get("validation_macro_f1", 0.0)
    st.sidebar.success(f"✅ Active Model: **{selected_name}**")
    st.sidebar.caption(f"Val Macro F1: {val_macro:.4f} | Target: {reg_info.get('target_column', 'risk_label')}")
elif has_primary_model:
    st.sidebar.success("✅ Primary Model Loaded")
else:
    st.sidebar.warning("⚠️ No Trained Model Registered")
    st.sidebar.caption("Using Heuristic Kinematic Baseline. Run `training/compare_models.py` to train & register models.")

# Pipeline execution options
st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Processing Options")
sample_fps = st.sidebar.slider("Analysis Sampling Rate (FPS)", min_value=5.0, max_value=30.0, value=15.0, step=1.0)

include_extended = st.sidebar.checkbox(
    "Include Deferred Movement Categories (Experimental)",
    value=False,
    help="Enables experimental SCISSOR_JUMP and FOOTWORK_SPLIT_STEP detection. These categories are outside the active Rubric v0.3 scope.",
)

camera_viewpoint = st.sidebar.selectbox(
    "Camera Viewpoint (Metadata)",
    ["side_90deg (Standard Lateral)", "diagonal_45deg", "front_view", "rear_view", "broadcast_free"],
    help="Lateral 90° view is required for planar sagittal joint kinematics. Non-perpendicular angles suffer perspective foreshortening.",
)

# Previous Runs History
st.sidebar.markdown("---")
st.sidebar.subheader("📜 Run History")
hist_df = get_history_df()
if not hist_df.empty:
    st.sidebar.dataframe(hist_df[["timestamp", "video_name", "movement_type", "risk_label"]], hide_index=True)
    if st.sidebar.button("Clear History"):
        clear_history()
        st.rerun()
else:
    st.sidebar.caption("No previously analyzed videos.")


# Main Dashboard Header
st.title("🏸 AI Badminton Movement Analysis & Injury-Risk Dashboard")
st.markdown(
    "Automated video-to-clips pipeline for badminton athletic movement detection, "
    "movement classification (Lunge, Smash Landing, Change of Direction), and **Rubric v0.3** biomechanical posture-risk screening."
)

# Video Upload & Validation Section
st.markdown("### 📤 1. Video Upload & Inspection")
upload_col, info_col = st.columns([2, 1])

with upload_col:
    uploaded_file = st.file_uploader(
        "Upload badminton match or training footage",
        type=["mp4", "avi", "mov", "mkv"],
        help="Upload MP4/AVI/MOV video. Standard lateral view (~90°) on tripod recommended.",
    )

saved_video_path = None
if uploaded_file is not None:
    uploaded_bytes = uploaded_file.getvalue()
    upload_key = (uploaded_file.name, hashlib.sha256(uploaded_bytes).hexdigest())
    if st.session_state.get("_active_upload_key") != upload_key:
        # Do not show the previous upload's events while a different file is active.
        st.session_state.pop("pipeline_result", None)
        st.session_state.pop("video_name", None)
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uploaded_file.name}")
        tfile.write(uploaded_bytes)
        tfile.close()
        st.session_state["_saved_video_path"] = tfile.name
        st.session_state["_active_upload_key"] = upload_key
    saved_video_path = st.session_state["_saved_video_path"]

    # Validate video
    v_meta = validate_video(saved_video_path)
    with info_col:
        st.markdown("**Video Inspection Metadata:**")
        if v_meta["is_valid"]:
            st.success("✅ Video file validated successfully")
            st.write(f"• **Duration:** {v_meta['duration']:.2f} s")
            st.write(f"• **Resolution:** {v_meta['width']} × {v_meta['height']}")
            st.write(f"• **Frame Rate:** {v_meta['fps']} FPS ({v_meta['frame_count']} frames)")
            st.write(f"• **Codec:** {v_meta['codec']} ({v_meta['file_size_mb']} MB)")
        else:
            st.error(f"❌ Invalid Video: {v_meta['error']}")

    # Process Video Button
    if v_meta["is_valid"]:
        process_btn = st.button("🚀 Process Video Pipeline (Detect Events & Analyze Biomechanics)")
        if process_btn:
            st.session_state.pop("pipeline_result", None)
            progress_bar = st.progress(0.0)
            status_text = st.empty()

            def update_cb(pct, text):
                progress_bar.progress(pct)
                status_text.text(text)

            output_dir = os.path.join(BASE_DIR, "data", "processed", Path(uploaded_file.name).stem)
            pipeline = VideoProcessingPipeline(model_path=PRIMARY_MODEL_PATH if has_primary_model else None)

            with st.spinner("Processing video through pose estimation & event detection..."):
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
                st.success(f"🎉 Pipeline completed in {t_elapsed:.1f}s! Found {pipeline_result['total_events']} movement events.")
                st.session_state["pipeline_result"] = pipeline_result
                st.session_state["video_name"] = uploaded_file.name
                save_run_to_history(uploaded_file.name, pipeline_result["events"])
            else:
                st.error(f"Processing failed: {pipeline_result.get('error', 'Unknown error')}")
elif st.session_state.get("_active_upload_key") is not None:
    st.session_state.pop("pipeline_result", None)
    st.session_state.pop("video_name", None)
    st.session_state.pop("_active_upload_key", None)
    st.session_state.pop("_saved_video_path", None)


# Render Results Workspace
if "pipeline_result" in st.session_state and st.session_state["pipeline_result"].get("success"):
    res = st.session_state["pipeline_result"]
    events = res.get("events", [])

    st.markdown("---")
    st.markdown("### 📊 2. Main Analysis Workspace")

    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.subheader("🎥 Video & Detected Movement Events")

        tab_annotated, tab_events = st.tabs(["Full Tracking Video", "Detected Movement Clips"])

        with tab_annotated:
            ann_vid = res.get("annotated_video_path")
            if ann_vid and os.path.exists(ann_vid):
                st.video(ann_vid)
            else:
                st.info("Full tracking overlay video rendering.")

        with tab_events:
            if events:
                event_options = [
                    f"{ev['event_id']} — {ev['movement_category']} ({ev['start_time']}s - {ev['end_time']}s)"
                    for ev in events
                ]
                selected_event_idx = st.selectbox(
                    "Select Detected Event to Inspect:",
                    range(len(events)),
                    format_func=lambda i: event_options[i],
                )
                active_event = events[selected_event_idx]

                st.markdown(f"**Event Clip Preview:** `{active_event['event_id']}`")
                clip_p = active_event.get("clip_path")
                if clip_p and os.path.exists(clip_p):
                    st.video(clip_p)
                else:
                    st.caption("Sub-clip video preview unavailable.")

                key_p = active_event.get("representative_frame")
                if key_p and os.path.exists(key_p):
                    st.image(key_p, caption=f"Representative Keyframe ({active_event['start_time']}s - {active_event['end_time']}s)")
            else:
                st.warning("No candidate movement events were detected in this video segment.")
                active_event = None

    with col_right:
        st.subheader("🔍 Event Movement Analysis & Biomechanics")

        if events and active_event:
            cat = active_event.get("movement_category", "UNCERTAIN")
            meta = CATEGORY_METADATA.get(cat, CATEGORY_METADATA["UNCERTAIN"])

            st.markdown(f"#### Movement Category: **{meta['icon']} {cat}**")
            st.caption(f"{meta['description']} — *{meta['rubric_status']}*")

            # Classification Output & Probabilities
            probs = active_event.get("probabilities", None)
            pred_src = active_event.get("prediction_source", "heuristic")

            if pred_src == "trained_model" and probs:
                st.markdown("##### 📈 Model Classification Probabilities")
                for c_name, p_val in sorted(probs.items(), key=lambda x: -x[1]):
                    st.progress(float(p_val), text=f"{c_name}: {p_val*100:.1f}%")
            else:
                st.markdown("##### ⚙️ Movement Classification Source")
                st.info(
                    "ℹ️ **Heuristic Kinematic Proposal**: Detected using biomechanical rules. "
                    "No trained ML classifier is loaded; uncalibrated probabilities are withheld."
                )

            st.markdown("---")
            # Risk Assessment Box
            risk = active_event.get("risk_level", "UNASSESSED")
            st.markdown("##### 🛡️ Biomechanical Injury-Risk (Rubric v0.3)")

            if risk == "PROVISIONAL":
                badge_html = "<div class='risk-badge-mod'>🟡 Assessment: PROVISIONAL (Unvalidated Cutoffs)</div>"
            elif risk == "UNASSESSED":
                badge_html = "<div class='risk-badge-deferred'>⚪ Assessment: UNASSESSED (Deferred in Rubric v0.3)</div>"
            elif risk == "UNCERTAIN":
                badge_html = "<div class='risk-badge-deferred'>⚪ Assessment: UNCERTAIN (Insufficient / Low Quality Data)</div>"
            else:
                badge_html = f"<div class='risk-badge-deferred'>⚪ Assessment: {risk}</div>"

            st.markdown(badge_html, unsafe_allow_html=True)

            obs_list = active_event.get("risk_observations", [])
            for ob in obs_list:
                st.write(ob)

            if cat in DEFERRED_RISK_RUBRIC_CATEGORIES:
                st.info(
                    f"ℹ️ **Rubric v0.3 Scope Notice**: `{cat}` is a valid movement category, "
                    "but injury-risk scoring is deferred in Rubric v0.3. "
                    "Risk labels are not automatically assigned."
                )

            st.markdown("---")
            # Joint Angle Metrics
            st.markdown("##### 📐 Measured Joint Angles")
            feats = active_event.get("features", {})
            left_knee = feats.get("left_knee_min")
            right_knee = feats.get("right_knee_min")
            left_elbow = feats.get("left_elbow_min")
            right_elbow = feats.get("right_elbow_min")
            def finite_angle(value):
                try:
                    number = float(value)
                except (TypeError, ValueError):
                    return None
                return number if math.isfinite(number) else None
            valid_knees = [v for v in (finite_angle(left_knee), finite_angle(right_knee)) if v is not None]
            valid_elbows = [v for v in (finite_angle(left_elbow), finite_angle(right_elbow)) if v is not None]
            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("Min Knee Flexion", format_angle(min(valid_knees) if valid_knees else None))
                st.caption(f"L: {format_angle(left_knee)} | R: {format_angle(right_knee)}")
            with m2:
                st.metric("Min Elbow Angle", format_angle(min(valid_elbows) if valid_elbows else None))
                st.caption(f"L: {format_angle(left_elbow)} | R: {format_angle(right_elbow)}")
            with m3:
                st.metric("Max Torso Lean", format_angle(feats.get("trunk_angle_max")))
                st.caption(f"Min: {format_angle(feats.get('trunk_angle_min'))}")

            asym = abs(finite_angle(left_knee) - finite_angle(right_knee)) if finite_angle(left_knee) is not None and finite_angle(right_knee) is not None else None
            st.write(f"\u2022 **Bilateral Knee Asymmetry:** `{format_angle(asym)}`")
            st.write(f"• **Pose Quality Confidence:** `{active_event.get('pose_quality', 1.0):.2f}`")

    # Lower Section: Summary Analytics & Downloads
    st.markdown("---")
    st.markdown("### 📈 3. Summary Analytics & Manifest Export")

    sum_col1, sum_col2, sum_col3 = st.columns([2, 2, 1])

    with sum_col1:
        st.markdown("**Movement Category Distribution:**")
        cat_counts = {}
        for ev in events:
            c = ev.get("movement_category", "UNCERTAIN")
            cat_counts[c] = cat_counts.get(c, 0) + 1
        cat_df = pd.DataFrame(list(cat_counts.items()), columns=["Category", "Count"])
        st.bar_chart(cat_df.set_index("Category"))

    with sum_col2:
        st.markdown("**Biomechanical Assessment Distribution:**")
        risk_counts = {"PROVISIONAL": 0, "UNASSESSED": 0, "UNCERTAIN": 0}
        for ev in events:
            r = ev.get("risk_level", "UNASSESSED")
            risk_counts[r] = risk_counts.get(r, 0) + 1
        risk_df = pd.DataFrame(list(risk_counts.items()), columns=["Status", "Count"])
        st.bar_chart(risk_df.set_index("Status"))

    with sum_col3:
        st.markdown("**Export Artifacts:**")
        manifest_p = res.get("manifest_path")
        if manifest_p and os.path.exists(manifest_p):
            with open(manifest_p, "r") as mf:
                manifest_str = mf.read()
            st.download_button(
                label="📥 Download Event Manifest (JSON)",
                data=manifest_str,
                file_name=f"manifest_{Path(st.session_state.get('video_name', 'video')).stem}.json",
                mime="application/json",
            )

        # CSV table export
        if events:
            events_df = pd.DataFrame(events)
            st.download_button(
                label="📥 Download Event Summary (CSV)",
                data=events_df.to_csv(index=False),
                file_name=f"events_{Path(st.session_state.get('video_name', 'video')).stem}.csv",
                mime="text/csv",
            )

    # Prototype Disclaimer
    st.markdown("---")
    st.info(
        "⚠️ **Engineering Prototype Disclaimer**: This software is an academic engineering prototype "
        "designed for computer-vision movement proposal and biomechanical posture screening. "
        "It is **NOT** a certified medical diagnostic device or clinical injury predictor."
    )
