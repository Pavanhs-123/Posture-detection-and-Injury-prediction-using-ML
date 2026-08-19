import argparse
import csv
import json
import os
import time
from pathlib import Path
import cv2
import mediapipe as mp
import numpy as np

from pose_analysis import MODEL_PATH, ensure_model_file, extract_pose_features

# Output CSV columns mapping to sequence statistics
FIELDNAMES = [
    "sequence_id",
    "video_id",
    "movement_type",
    "frame_start",
    "frame_end",
    "duration",
    "left_knee_min",
    "left_knee_max",
    "right_knee_min",
    "right_knee_max",
    "left_elbow_min",
    "left_elbow_max",
    "right_elbow_min",
    "right_elbow_max",
    "trunk_angle_min",
    "trunk_angle_max",
    "risk_label"
]

MOVEMENT_TYPES = {
    "1": "LUNGE",
    "2": "SMASH_LANDING",
    "3": "SCISSOR_JUMP",
    "4": "CHANGE_OF_DIRECTION",
    "5": "FOOTWORK_SPLIT_STEP"
}

RISK_LABELS = {
    "1": "LOW",
    "2": "MODERATE",
    "3": "HIGH"
}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract a sequence of pose features for badminton movements")
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument("--output-csv", default="data/dataset.csv", help="CSV path for dataset")
    parser.add_argument("--start-time", type=float, help="Sequence start time in seconds")
    parser.add_argument("--end-time", type=float, help="Sequence end time in seconds")
    parser.add_argument("--sample-fps", type=float, default=10.0, help="Frames per second to extract landmarks")
    parser.add_argument("--append", action="store_true", default=True, help="Append to existing dataset CSV")
    return parser.parse_args()

def prompt_float(prompt: str, minimum: float | None = None) -> float:
    while True:
        raw = input(prompt).strip()
        try:
            val = float(raw)
            if minimum is not None and val < minimum:
                print(f"Value must be >= {minimum}.")
                continue
            return val
        except ValueError:
            print("Please enter a valid decimal number.")

def prompt_choice(prompt_text: str, options: dict) -> str:
    print(prompt_text)
    for k, v in options.items():
        print(f"  [{k}] {v}")
    print("  [6] Enter custom string")
    
    while True:
        choice = input("Select: ").strip()
        if choice in options:
            return options[choice]
        elif choice == "6":
            custom = input("Enter custom string: ").strip().upper()
            if custom:
                return custom
        print("Invalid choice. Please select one of the numbers.")

def draw_skeleton_overlay(frame, landmarks, pose_landmark):
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

    # Draw bone lines
    line_color = (255, 0, 0)
    cv2.line(frame, points["left_shoulder"], points["left_elbow"], line_color, 2)
    cv2.line(frame, points["left_elbow"], points["left_wrist"], line_color, 2)
    cv2.line(frame, points["left_hip"], points["left_knee"], line_color, 2)
    cv2.line(frame, points["left_knee"], points["left_ankle"], line_color, 2)
    cv2.line(frame, points["right_shoulder"], points["right_elbow"], line_color, 2)
    cv2.line(frame, points["right_elbow"], points["right_wrist"], line_color, 2)
    cv2.line(frame, points["right_hip"], points["right_knee"], line_color, 2)
    cv2.line(frame, points["right_knee"], points["right_ankle"], line_color, 2)
    cv2.line(frame, points["left_shoulder"], points["right_shoulder"], line_color, 2)
    cv2.line(frame, points["left_hip"], points["right_hip"], line_color, 2)

def main() -> None:
    args = parse_args()
    video_path = Path(args.video).expanduser().resolve()
    output_csv = Path(args.output_csv).expanduser().resolve()
    
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found at: {video_path}")

    # Prompt for start/end/fps if not specified in CLI
    start_time = args.start_time if args.start_time is not None else prompt_float("Enter start time in seconds: ", 0.0)
    end_time = args.end_time if args.end_time is not None else prompt_float("Enter end time in seconds: ", start_time)
    sample_fps = args.sample_fps if args.sample_fps is not None else prompt_float("Enter sample FPS: ", 0.1)

    if end_time <= start_time:
        raise ValueError("End time must be greater than start time.")

    ensure_model_file()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("Unable to open video file.")

    source_fps = cap.get(cv2.CAP_PROP_FPS)
    if source_fps <= 0:
        source_fps = 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = total_frames / source_fps

    start_sec = max(0.0, min(start_time, video_duration))
    end_sec = max(start_sec, min(end_time, video_duration))
    frame_start = int(round(start_sec * source_fps))
    frame_end = int(round(end_sec * source_fps))
    duration = end_sec - start_sec

    # Setup directories
    data_dir = output_csv.parent
    landmarks_dir = data_dir / "landmarks"
    clips_dir = data_dir / "clips"
    landmarks_dir.mkdir(parents=True, exist_ok=True)
    clips_dir.mkdir(parents=True, exist_ok=True)

    # Sequence ID setup
    timestamp_str = str(int(time.time()))
    clean_name = video_path.stem.replace(" ", "_")
    seq_id = f"seq_{clean_name}_{start_sec:.1f}to{end_sec:.1f}_{timestamp_str}"

    print(f"\nProcessing {video_path.name} from {start_sec:.2f}s to {end_sec:.2f}s...")
    
    # MediaPipe setup
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
    
    # Temporary video file to write annotated clip
    temp_clip_path = clips_dir / f"temp_{seq_id}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_writer = cv2.VideoWriter(str(temp_clip_path), fourcc, source_fps, (frame_width, frame_height))

    pose_features_list = []
    raw_landmarks_all_frames = []
    playback_frames = []

    sample_interval = 1.0 / sample_fps
    next_sample_time = start_sec
    frame_idx = 0

    with PoseLandmarker.create_from_options(options) as landmarker:
        while True:
            success, frame = cap.read()
            if not success:
                break

            timestamp_sec = frame_idx / source_fps
            if timestamp_sec < start_sec:
                frame_idx += 1
                continue
            if timestamp_sec > end_sec:
                break

            should_sample = False
            if timestamp_sec >= next_sample_time - 1e-6:
                should_sample = True
                next_sample_time += sample_interval

            # Run detection
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            timestamp_ms = int(timestamp_sec * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            annotated_frame = frame.copy()
            if result.pose_landmarks:
                landmarks = result.pose_landmarks[0]
                features = extract_pose_features(landmarks, PoseLandmark)
                draw_skeleton_overlay(annotated_frame, landmarks, PoseLandmark)
                
                if should_sample:
                    pose_features_list.append(features)
                    # Store raw landmarks for JSON
                    frame_lms = []
                    for lm in landmarks:
                        frame_lms.append({
                            "x": float(lm.x),
                            "y": float(lm.y),
                            "z": float(lm.z),
                            "visibility": float(lm.visibility) if hasattr(lm, "visibility") else 0.0,
                            "presence": float(lm.presence) if hasattr(lm, "presence") else 0.0
                        })
                    raw_landmarks_all_frames.append({
                        "frame_index": frame_idx,
                        "timestamp_sec": round(timestamp_sec, 3),
                        "landmarks": frame_lms
                    })
            
            playback_frames.append(annotated_frame)
            out_writer.write(annotated_frame)
            frame_idx += 1

    cap.release()
    out_writer.release()

    if not pose_features_list:
        if temp_clip_path.exists():
            temp_clip_path.unlink()
        print("No pose landmarks detected in this range. Row skipped.")
        return

    # Aggregate features into sequence bounds
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

    # Verify if cv2 window can open (GUI availability check)
    gui_available = True
    try:
        cv2.imshow("Testing GUI Window", np.zeros((100, 100, 3), dtype=np.uint8))
        cv2.destroyWindow("Testing GUI Window")
    except Exception:
        gui_available = False

    # Loop preview playback
    if gui_available and playback_frames:
        print("\nPlaying loop preview in window... Press any key to stop playback and start labeling.")
        loop_preview = True
        while loop_preview:
            for frame in playback_frames:
                cv2.imshow("Sequence Preview - Press any key to Label", frame)
                # Play at ~25 FPS
                if cv2.waitKey(40) & 0xFF != 255:
                    loop_preview = False
                    break
        cv2.destroyWindow("Sequence Preview - Press any key to Label")

    # Interactive prompts
    print("\n" + "="*40)
    print(f"Sequence Summary:")
    print(f"  Knee range: L:[{features_summary['left_knee_min']} - {features_summary['left_knee_max']}] | R:[{features_summary['right_knee_min']} - {features_summary['right_knee_max']}]")
    print(f"  Elbow range: L:[{features_summary['left_elbow_min']} - {features_summary['left_elbow_max']}] | R:[{features_summary['right_elbow_min']} - {features_summary['right_elbow_max']}]")
    print(f"  Trunk lean tilt: [{features_summary['trunk_angle_min']} - {features_summary['trunk_angle_max']}]")
    print("="*40)

    movement_type = prompt_choice("Select Movement Type:", MOVEMENT_TYPES)
    risk_label = prompt_choice("Select Injury Risk Label:", RISK_LABELS)

    # Save raw landmarks to JSON
    landmarks_file = landmarks_dir / f"{seq_id}.json"
    with open(landmarks_file, "w") as f:
        json.dump({
            "sequence_id": seq_id,
            "video_name": video_path.name,
            "start_time": start_sec,
            "end_time": end_sec,
            "frames": raw_landmarks_all_frames
        }, f, indent=2)

    # Move temp video to permanent location (clips/seq_id.mp4)
    permanent_clip_path = clips_dir / f"{seq_id}.mp4"
    if temp_clip_path.exists():
        temp_clip_path.rename(permanent_clip_path)

    # Prepare dict to write
    row_data = {
        "sequence_id": seq_id,
        "video_id": video_path.name,
        "movement_type": movement_type,
        "frame_start": frame_start,
        "frame_end": frame_end,
        "duration": round(duration, 3),
        "left_knee_min": features_summary["left_knee_min"],
        "left_knee_max": features_summary["left_knee_max"],
        "right_knee_min": features_summary["right_knee_min"],
        "right_knee_max": features_summary["right_knee_max"],
        "left_elbow_min": features_summary["left_elbow_min"],
        "left_elbow_max": features_summary["left_elbow_max"],
        "right_elbow_min": features_summary["right_elbow_min"],
        "right_elbow_max": features_summary["right_elbow_max"],
        "trunk_angle_min": features_summary["trunk_angle_min"],
        "trunk_angle_max": features_summary["trunk_angle_max"],
        "risk_label": risk_label
    }

    # Write to CSV
    file_exists = output_csv.exists()
    mode = "a" if args.append and file_exists else "w"
    
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, mode, newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=FIELDNAMES)
        if mode == "w" or output_csv.stat().st_size == 0:
            writer.writeheader()
        writer.writerow(row_data)

    print(f"\nSuccessfully saved sequence '{seq_id}' to {output_csv}")
    print(f"Raw landmarks saved to {landmarks_file}")
    print(f"Video clip saved to {permanent_clip_path}")

if __name__ == "__main__":
    main()
