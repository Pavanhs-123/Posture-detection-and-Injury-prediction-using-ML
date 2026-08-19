import os
import sys
import sqlite3
import tempfile
from datetime import datetime
import streamlit as st
import pandas as pd

# Add the project root to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.inference.pipeline import analyze_video

# Define default paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "data", "history.db")
MODEL_PATH = os.path.join(BASE_DIR, "models", "injury_risk_model.pkl")

# Initialize database
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
            trunk_angle_max REAL
        )
    """)
    conn.commit()
    conn.close()

def save_to_history(video_name, movement_type, risk_label, features):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("""
        INSERT INTO history (
            timestamp, video_name, movement_type, risk_label,
            left_knee_min, right_knee_min, left_elbow_min, right_elbow_min, trunk_angle_max
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        now, video_name, movement_type, risk_label,
        features.get("left_knee_min", 180.0),
        features.get("right_knee_min", 180.0),
        features.get("left_elbow_min", 180.0),
        features.get("right_elbow_min", 180.0),
        features.get("trunk_angle_max", 0.0)
    ))
    conn.commit()
    conn.close()

def get_history():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT timestamp, video_name, movement_type, risk_label FROM history ORDER BY id DESC", conn)
    conn.close()
    return df

def clear_history():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM history")
    conn.commit()
    conn.close()

# Initialize database on startup
init_db()

# Streamlit App Setup
st.set_page_config(
    page_title="Badminton Pose AI & Injury-Risk",
    page_icon="🏸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar UI
st.sidebar.title("🏸 Badminton AI Controller")

# Check model availability
model_exists = os.path.exists(MODEL_PATH)
if model_exists:
    st.sidebar.success("✅ Real Model Loaded")
    mode = st.sidebar.selectbox("Inference Mode", ["Real Model", "Demo Mode (Integration Test Model)"])
else:
    st.sidebar.warning("⚠️ Real Model Not Installed")
    st.sidebar.info(
        "To install the real model, train a model using `train.py` and place "
        "the exported pickle in:\n`models/injury_risk_model.pkl`"
    )
    mode = st.sidebar.selectbox("Inference Mode", ["Demo Mode (Integration Test Model)"])

# Load history in sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("📜 Previous Analyses")
history_df = get_history()
if not history_df.empty:
    st.sidebar.dataframe(history_df, use_container_width=True, hide_index=True)
    if st.sidebar.button("Clear History"):
        clear_history()
        st.rerun()
else:
    st.sidebar.caption("No previous analyses found.")

# Main Page Header
st.title("🏸 AI-Based Badminton Posture Detection & Injury-Risk Assessment")
st.markdown(
    "Upload a video of a badminton movement sequence (e.g., Smash Landing, Lunge, Split Step), "
    "configure the start/end timing, and execute biomechanical pose analysis."
)

# Upload Section
col_upload, col_params = st.columns([2, 1])

with col_upload:
    st.subheader("📤 Step 1: Upload Movement Video")
    uploaded_file = st.file_uploader(
        "Choose video file...", 
        type=["mp4", "avi", "mov"],
        help="Make sure the player's full body is visible and shot from a side angle."
    )

with col_params:
    st.subheader("⚙️ Step 2: Configuration")
    movement_type = st.selectbox(
        "Select Movement Type",
        ["LUNGE", "SMASH_LANDING", "SCISSOR_JUMP", "CHANGE_OF_DIRECTION", "FOOTWORK_SPLIT_STEP"]
    )
    
    # Clip timing options
    start_time = st.number_input("Clip Start Time (sec)", min_value=0.0, value=0.0, step=0.5)
    end_time = st.number_input("Clip End Time (sec)", min_value=0.5, value=5.0, step=0.5)
    sample_fps = st.slider("Sampling Rate (FPS)", min_value=1.0, max_value=25.0, value=10.0, step=1.0)

# Run Analysis Trigger
if uploaded_file is not None:
    # Save uploaded file to temp path
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_file.read())
    tfile.close()

    if st.button("🚀 Analyze Video", use_container_width=True):
        with st.spinner("Initializing MediaPipe & extracting joint features..."):
            # Determine path to use for model
            m_path = MODEL_PATH if mode == "Real Model" else None
            
            # Execute inference pipeline
            result = analyze_video(
                video_path=tfile.name,
                start_time=start_time,
                end_time=end_time,
                sample_fps=sample_fps,
                model_path=m_path
            )

        # Cleanup upload temp file
        try:
            os.unlink(tfile.name)
        except Exception:
            pass

        if not result["success"]:
            st.error(f"❌ Analysis Failed: {result['error']}")
        else:
            st.success("🎉 Video processed successfully!")
            
            # Save results to local history DB
            save_to_history(
                video_name=uploaded_file.name,
                movement_type=movement_type,
                risk_label=result["risk_label"],
                features=result["features"]
            )

            # Display Results Layout
            st.markdown("---")
            col_video, col_results = st.columns([3, 2])

            with col_video:
                st.subheader("🎥 Skeletal Tracking Video")
                if result["annotated_video_path"] and os.path.exists(result["annotated_video_path"]):
                    st.video(result["annotated_video_path"])
                else:
                    st.warning("Processed overlay video file could not be generated.")

            with col_results:
                st.subheader("📊 Posture Analysis Result")
                
                # Visual Risk Card
                risk = result["risk_label"]
                if risk == "LOW":
                    st.markdown(
                        "<div style='background-color: #d4edda; border-left: 6px solid #28a745; padding: 15px; border-radius: 4px;'>"
                        "<h2 style='color: #155724; margin:0;'>💚 Risk Assessment: LOW</h2>"
                        "<p style='color: #155724; margin: 5px 0 0 0;'>Posture angles fall within normal anatomical limits.</p>"
                        "</div>", 
                        unsafe_allow_allow_html=True
                    )
                elif risk == "MODERATE":
                    st.markdown(
                        "<div style='background-color: #fff3cd; border-left: 6px solid #ffc107; padding: 15px; border-radius: 4px;'>"
                        "<h2 style='color: #856404; margin:0;'>💛 Risk Assessment: MODERATE</h2>"
                        "<p style='color: #856404; margin: 5px 0 0 0;'>Some joint extension/flexion limits exceeded. Monitor technique.</p>"
                        "</div>", 
                        unsafe_allow_html=True
                    )
                else:  # HIGH
                    st.markdown(
                        "<div style='background-color: #f8d7da; border-left: 6px solid #dc3545; padding: 15px; border-radius: 4px;'>"
                        "<h2 style='color: #721c24; margin:0;'>❤️ Risk Assessment: HIGH</h2>"
                        "<p style='color: #721c24; margin: 5px 0 0 0;'>Critical joint angles exceeded safety boundaries. High risk of joint loading/injury.</p>"
                        "</div>", 
                        unsafe_allow_html=True
                    )

                # Show Probabilities
                probs = result["risk_probabilities"]
                if probs:
                    st.markdown("##### Prediction Probabilities")
                    for k, v in probs.items():
                        st.progress(v, text=f"{k}: {v*100:.1f}%")

                st.markdown("---")
                st.markdown("##### 📐 Biomechanical Statistics (Min / Max)")
                feats = result["features"]
                
                # Display metrics grid
                col_knee, col_elbow, col_trunk = st.columns(3)
                with col_knee:
                    st.metric("Knee Min Angle (L/R)", f"{min(feats['left_knee_min'], feats['right_knee_min']):.1f}°")
                    st.caption(f"L: {feats['left_knee_min']}° | R: {feats['right_knee_min']}°")
                with col_elbow:
                    st.metric("Elbow Min Angle (L/R)", f"{min(feats['left_elbow_min'], feats['right_elbow_min']):.1f}°")
                    st.caption(f"L: {feats['left_elbow_min']}° | R: {feats['right_elbow_min']}°")
                with col_trunk:
                    st.metric("Max Torso Lean", f"{feats['trunk_angle_max']:.1f}°")
                    st.caption(f"Min: {feats['trunk_angle_min']}°")

            # Explanation Section
            st.markdown("---")
            st.subheader("💡 Why This Assessment? (Biomechanical Interpretation)")
            
            # Logic generating dynamic biomechanical comments based on metrics
            explanations = []
            
            k_min = min(feats['left_knee_min'], feats['right_knee_min'])
            if k_min < 80.0:
                explanations.append(
                    f"🔴 **Extreme Knee Flexion ({k_min:.1f}°)**: The knee joint angle went below 80° during movement. "
                    "This creates excessive shear forces on the patellar tendon and ACL."
                )
            elif k_min < 100.0:
                explanations.append(
                    f"🟡 **Deep Knee Flexion ({k_min:.1f}°)**: Moderate knee load observed. "
                    "Ensure quad strength is sufficient to absorb landing impact."
                )
            else:
                explanations.append(
                    f"🟢 **Safe Knee Flexion ({k_min:.1f}°)**: Knee flexion remained within standard biomechanical safety margins."
                )

            t_max = feats['trunk_angle_max']
            if t_max > 38.0:
                explanations.append(
                    f"🔴 **Severe Torso Lean ({t_max:.1f}°)**: The player tilted their torso excessively relative to their hips. "
                    "This increases rotational torque on the lumbar spine and raises lower back injury risks."
                )
            elif t_max > 25.0:
                explanations.append(
                    f"🟡 **Moderate Torso Lean ({t_max:.1f}°)**: Torso tilt is slightly elevated. "
                    "Focus on core stabilization to keep the trunk upright."
                )
            else:
                explanations.append(
                    f"🟢 **Good Torso Alignment ({t_max:.1f}°)**: Torso lean is nominal. Spine loading remains low."
                )

            e_min = min(feats['left_elbow_min'], feats['right_elbow_min'])
            if e_min < 90.0:
                explanations.append(
                    f"🟡 **Tight Elbow Angle ({e_min:.1f}°)**: The swing extension is restricted. "
                    "May lead to repetitive strain on the elbow joint (tennis elbow) to generate power."
                )
            else:
                explanations.append(
                    f"🟢 **Correct Elbow Extension ({e_min:.1f}°)**: Arms show natural extension characteristics."
                )

            # Left/Right Asymmetry
            asym = abs(feats['left_knee_min'] - feats['right_knee_min'])
            if asym > 15.0:
                explanations.append(
                    f"⚠️ **Knee Flexion Asymmetry ({asym:.1f}°)**: A difference of more than 15° between left and right knee flexion "
                    "suggests uneven shock absorption during landing or lunging, overloading one leg."
                )

            for exp in explanations:
                st.markdown(f"- {exp}")
                
            # Sidebar refresh to show the new run in history
            st.sidebar.empty()
            st.rerun()

else:
    st.info("💡 Please upload an MP4/AVI/MOV file above to start the posture tracking process.")
