import os
import time
import subprocess
import tempfile
import cv2
import numpy as np
import mediapipe as mp

from pose_analysis import MODEL_PATH, ensure_model_file, extract_pose_features
from src.model.interface import InjuryRiskModel
from src.model.integration_test_model import IntegrationTestModel

def draw_skeleton_overlay(frame, landmarks, pose_landmark):
    """
    Draws pose landmarks and connections on the image frame.
    """
    height, width = frame.shape[:2]
    points = {
        "left_shoulder": (int(landmarks[pose_landmark.LEFT_SHOULDER].x * width), int(landmarks[pose_landmark.LEFT_SHOULDER].y * height)),
        "left_elbow": (int(landmarks[pose_landmark.LEFT_ELBOW].x * width), int(landmarks[pose_landmark.LEFT_ELBOW].y * height)),
        "left_wrist": (int(landmarks[pose_landmark.LEFT_WRIST].x * width), int(landmarks[pose_landmark.LEFT_WRIST].y * height)),
        "left_hip": (int(landmarks[pose_landmark.LEFT_HIP].x * width), int(landmarks[pose_landmark.LEFT_HIP].y * height)),
        "left_knee": (int(landmarks[pose_landmark.LEFT_KNEE].x * width), int(landmarks[pose_landmark.LEFT_KNEE].y * height)),
        "left_ankle": (int(landmarks[pose_landmark.LEFT_ANKLE].x * width), int(landmarks[pose_landmark.LEFT_ANKLE].y * height)),
        "right_shoulder": (int(landmarks[pose_landmark.RIGHT_SHOULDER].x * width), int(landmarks[pose_landmark.RIGHT_SHOULDER].y * height)),
        "right_elbow": (int(landmarks[pose_landmark.RIGHT_ELBOW].x * width), int(landmarks[pose_landmark.RIGHT_ELBOW].y * height)),
        "right_wrist": (int(landmarks[pose_landmark.RIGHT_WRIST].x * width), int(landmarks[pose_landmark.RIGHT_WRIST].y * height)),
        "right_hip": (int(landmarks[pose_landmark.RIGHT_HIP].x * width), int(landmarks[pose_landmark.RIGHT_HIP].y * height)),
        "right_knee": (int(landmarks[pose_landmark.RIGHT_KNEE].x * width), int(landmarks[pose_landmark.RIGHT_KNEE].y * height)),
        "right_ankle": (int(landmarks[pose_landmark.RIGHT_ANKLE].x * width), int(landmarks[pose_landmark.RIGHT_ANKLE].y * height)),
    }

    # Draw joint points
    for pt in points.values():
        cv2.circle(frame, pt, 5, (0, 255, 0), -1)

    # Draw bone connections
    line_color = (255, 0, 0)
    # Left arm/leg
    cv2.line(frame, points["left_shoulder"], points["left_elbow"], line_color, 2)
    cv2.line(frame, points["left_elbow"], points["left_wrist"], line_color, 2)
    cv2.line(frame, points["left_hip"], points["left_knee"], line_color, 2)
    cv2.line(frame, points["left_knee"], points["left_ankle"], line_color, 2)
    # Right arm/leg
    cv2.line(frame, points["right_shoulder"], points["right_elbow"], line_color, 2)
    cv2.line(frame, points["right_elbow"], points["right_wrist"], line_color, 2)
    cv2.line(frame, points["right_hip"], points["right_knee"], line_color, 2)
    cv2.line(frame, points["right_knee"], points["right_ankle"], line_color, 2)
    # Torso connections
    cv2.line(frame, points["left_shoulder"], points["right_shoulder"], line_color, 2)
    cv2.line(frame, points["left_hip"], points["right_hip"], line_color, 2)
    cv2.line(frame, points["left_shoulder"], points["left_hip"], line_color, 2)
    cv2.line(frame, points["right_shoulder"], points["right_hip"], line_color, 2)

def convert_to_h264(input_path: str, output_path: str) -> bool:
    """
    Converts a video to H.264 format using ffmpeg for web playback compatibility.
    """
    try:
        cmd = [
            "ffmpeg", "-y", "-i", input_path,
            "-vcodec", "libx264", "-pix_fmt", "yuv420p",
            "-profile:v", "baseline", "-level", "3.0",
            output_path
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return True
    except Exception as e:
        print(f"ffmpeg conversion failed: {e}")
        return False

def analyze_video(
    video_path: str,
    start_time: float = None,
    end_time: float = None,
    sample_fps: float = 10.0,
    model_path: str = None
) -> dict:
    """
    Processes a video sequence, extracts biomechanical features, executes inference, 
    and outputs an annotated H.264 video.
    """
    ensure_model_file()
    
    result_dict = {
        "success": False,
        "error": None,
        "sequence_id": None,
        "video_name": os.path.basename(video_path),
        "duration": 0.0,
        "features": {},
        "risk_label": None,
        "risk_probabilities": {},
        "annotated_video_path": None,
        "pose_detected": False
    }

    # Verify input exists
    if not os.path.exists(video_path):
        result_dict["error"] = f"Video file not found at: {video_path}"
        return result_dict

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        result_dict["error"] = "Unable to open video file."
        return result_dict

    source_fps = cap.get(cv2.CAP_PROP_FPS)
    if source_fps <= 0:
        source_fps = 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = total_frames / source_fps

    # Align timing bounds
    start_sec = max(0.0, start_time) if start_time is not None else 0.0
    end_sec = min(video_duration, end_time) if end_time is not None else video_duration
    if end_sec <= start_sec:
        cap.release()
        result_dict["error"] = "Stop time must be greater than start time."
        return result_dict

    duration = end_sec - start_sec
    result_dict["duration"] = round(duration, 2)

    # Sequence ID generation
    timestamp_str = str(int(time.time()))
    clean_name = os.path.splitext(os.path.basename(video_path))[0].replace(" ", "_")
    seq_id = f"seq_{clean_name}_{start_sec:.1f}to{end_sec:.1f}_{timestamp_str}"
    result_dict["sequence_id"] = seq_id

    # MediaPipe landmarker initialization
    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode
    PoseLandmark = mp.tasks.vision.PoseLandmark

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=VisionRunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Set up directories
    clips_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "clips")
    os.makedirs(clips_dir, exist_ok=True)
    temp_video_path = os.path.join(clips_dir, f"temp_{seq_id}.mp4")
    annotated_output_path = os.path.join(clips_dir, f"{seq_id}_annotated.mp4")

    # Define video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_writer = cv2.VideoWriter(temp_video_path, fourcc, source_fps, (frame_width, frame_height))

    pose_features_list = []
    raw_landmarks_all_frames = []
    
    sample_interval = 1.0 / sample_fps
    next_sample_time = start_sec
    
    frame_idx = 0
    pose_detected = False

    try:
        with PoseLandmarker.create_from_options(options) as landmarker:
            while True:
                success, frame = cap.read()
                if not success:
                    break

                timestamp_sec = frame_idx / source_fps
                
                # Check if frame is within the selected timing window
                if timestamp_sec < start_sec:
                    frame_idx += 1
                    continue
                if timestamp_sec > end_sec:
                    break

                # Determine if we should process and sample features on this frame
                should_sample = False
                if timestamp_sec >= next_sample_time - 1e-6:
                    should_sample = True
                    next_sample_time += sample_interval

                # Convert to RGB for MediaPipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                timestamp_ms = int(timestamp_sec * 1000)
                
                results = landmarker.detect_for_video(mp_image, timestamp_ms)

                if results.pose_landmarks:
                    pose_detected = True
                    landmarks = results.pose_landmarks[0]
                    
                    # Extract 2D features
                    features = extract_pose_features(landmarks, PoseLandmark)
                    
                    # Draw overlay
                    draw_skeleton_overlay(frame, landmarks, PoseLandmark)
                    
                    if should_sample:
                        pose_features_list.append(features)
                        
                        # Store raw landmarks for JSON export later if needed
                        frame_landmarks = []
                        for lm in landmarks:
                            frame_landmarks.append({
                                "x": float(lm.x),
                                "y": float(lm.y),
                                "z": float(lm.z),
                                "visibility": float(lm.visibility) if hasattr(lm, "visibility") else 0.0,
                                "presence": float(lm.presence) if hasattr(lm, "presence") else 0.0
                            })
                        raw_landmarks_all_frames.append({
                            "frame_index": frame_idx,
                            "timestamp_sec": round(timestamp_sec, 3),
                            "landmarks": frame_landmarks
                        })

                # Write annotated frame
                out_writer.write(frame)
                frame_idx += 1

    finally:
        cap.release()
        out_writer.release()

    # If no pose features were successfully extracted, cleanup and exit
    if not pose_features_list:
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
        result_dict["error"] = "No pose detected in the specified video segment."
        return result_dict

    result_dict["pose_detected"] = pose_detected

    # Aggregate features into sequence metrics
    left_knees = [f.left_knee_angle for f in pose_features_list if f.left_knee_angle is not None]
    right_knees = [f.right_knee_angle for f in pose_features_list if f.right_knee_angle is not None]
    left_elbows = [f.left_elbow_angle for f in pose_features_list if f.left_elbow_angle is not None]
    right_elbows = [f.right_elbow_angle for f in pose_features_list if f.right_elbow_angle is not None]
    trunk_leans = [f.torso_lean for f in pose_features_list if f.torso_lean is not None]

    features_summary = {
        "left_knee_min": round(float(np.min(left_knees)), 2) if left_knees else 180.0,
        "left_knee_max": round(float(np.max(left_knees)), 2) if left_knees else 180.0,
        "right_knee_min": round(float(np.min(right_knees)), 2) if right_knees else 180.0,
        "right_knee_max": round(float(np.max(right_knees)), 2) if right_knees else 180.0,
        "left_elbow_min": round(float(np.min(left_elbows)), 2) if left_elbows else 180.0,
        "left_elbow_max": round(float(np.max(left_elbows)), 2) if left_elbows else 180.0,
        "right_elbow_min": round(float(np.min(right_elbows)), 2) if right_elbows else 180.0,
        "right_elbow_max": round(float(np.max(right_elbows)), 2) if right_elbows else 180.0,
        "trunk_angle_min": round(float(np.min(trunk_leans)), 2) if trunk_leans else 0.0,
        "trunk_angle_max": round(float(np.max(trunk_leans)), 2) if trunk_leans else 0.0
    }
    result_dict["features"] = features_summary

    # Run Model Inference
    injury_model = None
    if model_path is not None and os.path.exists(model_path):
        try:
            injury_model = InjuryRiskModel()
            injury_model.load(model_path)
        except Exception as e:
            print(f"Failed to load model from path, falling back to Demo Mode: {e}")
            injury_model = None

    # Fallback to Demo Mode (Integration Test Model)
    if injury_model is None:
        injury_model = IntegrationTestModel()

    result_dict["risk_label"] = injury_model.predict(features_summary)
    result_dict["risk_probabilities"] = injury_model.predict_proba(features_summary)

    # Save landmarks to raw JSON file under data/landmarks/
    landmarks_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "landmarks")
    os.makedirs(landmarks_dir, exist_ok=True)
    import json
    landmarks_file = os.path.join(landmarks_dir, f"{seq_id}.json")
    with open(landmarks_file, "w") as f:
        json.dump({
            "sequence_id": seq_id,
            "video_name": os.path.basename(video_path),
            "start_time": start_sec,
            "end_time": end_sec,
            "frames": raw_landmarks_all_frames
        }, f, indent=2)

    # Convert processed video to H.264 for Streamlit playback
    h264_success = convert_to_h264(temp_video_path, annotated_output_path)
    if h264_success:
        result_dict["annotated_video_path"] = annotated_output_path
        # Clean up temp raw video
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
    else:
        # Fallback to unconverted temp video if ffmpeg is broken
        result_dict["annotated_video_path"] = temp_video_path

    result_dict["success"] = True
    return result_dict
