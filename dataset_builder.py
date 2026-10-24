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
    "player_id",
    "movement_type",
    "camera_viewpoint",
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
    "3": "CHANGE_OF_DIRECTION",
}

VALID_TARGET_CLASSES = {"LUNGE", "SMASH_LANDING", "CHANGE_OF_DIRECTION"}
RETIRED_CLASSES = {"SCISSOR_JUMP", "FOOTWORK_SPLIT_STEP"}

CAMERA_VIEWPOINTS = {
    "1": "side_90deg",
    "2": "diagonal_45deg",
    "3": "front_view",
    "4": "rear_view",
    "5": "broadcast_free",
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
    parser.add_argument(
        "--camera-viewpoint",
        type=str,
        default="side_90deg",
        choices=["side_90deg", "diagonal_45deg", "front_view", "rear_view", "broadcast_free"],
        help="Camera viewpoint metadata (default: side_90deg - preferred: fixed side-view at ~90 degrees with full body and feet visible)",
    )
    parser.add_argument("--candidates-csv", type=str, help="Optional candidates CSV from detect_candidates.py to select pre-detected movement intervals")
    parser.add_argument("--player-id", type=str, default="", help="Optional unique player identity to support player-grouped evaluation")
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

def prompt_choice(prompt_text: str, options: dict, default_choice: str | None = None) -> str:
    print(prompt_text)
    for k, v in options.items():
        tag = " (Suggested)" if v == default_choice else ""
        print(f"  [{k}] {v}{tag}")
    
    while True:
        prompt_label = f"Select (Enter for {default_choice}): " if default_choice else "Select: "
        choice = input(prompt_label).strip()
        if not choice and default_choice:
            return default_choice
        if choice in options:
            return options[choice]
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

    suggested_movement = None
    start_time = args.start_time
    end_time = args.end_time

    # Load candidates if CSV provided
    if args.candidates_csv and Path(args.candidates_csv).exists() and (start_time is None or end_time is None):
        cand_path = Path(args.candidates_csv)
        with open(cand_path, "r") as f:
            reader = list(csv.DictReader(f))

        # Filter matching video name if applicable
        video_cands = [r for r in reader if r.get("video_id") == video_path.name]
        cands_to_show = video_cands if video_cands else reader

        if cands_to_show:
            print("\n" + "=" * 70)
            print("🏸 Movement Candidates (Human Review Verification Required):")
            print("=" * 70)
            for idx, c in enumerate(cands_to_show, start=1):
                c_id = c.get("candidate_id", f"cand_{idx}")
                p_mov = c.get("proposed_movement") or c.get("movement_type", "UNCERTAIN")
                score = c.get("heuristic_score") or c.get("confidence_score", "N/A")
                decision = (c.get("human_decision") or "PENDING").upper()
                r_mov = c.get("reviewed_movement") or "-"
                status_icon = "✅ ACCEPTED" if decision == "ACCEPT" else ("❌ REJECTED" if decision == "REJECT" else ("❓ UNCERTAIN" if decision == "UNCERTAIN" else "⏳ PENDING"))
                print(f"  [{idx}] {c_id:<10} | Proposed: {p_mov:<20} | {c['start_time']}s - {c['end_time']}s | Score: {score} | Review: {status_icon} (Confirmed: {r_mov})")
            print("  [0] Enter manual timing bounds")
            print("=" * 70)

            while True:
                sel = input("Select candidate number to label (default: 1, 0 for manual): ").strip()
                sel_idx = int(sel) if sel.isdigit() else 1
                if sel_idx == 0:
                    break
                if 1 <= sel_idx <= len(cands_to_show):
                    chosen = cands_to_show[sel_idx - 1]
                    c_decision = (chosen.get("human_decision") or "PENDING").upper()
                    c_id = chosen.get("candidate_id", f"cand_{sel_idx}")
                    prop_class = chosen.get("proposed_movement") or chosen.get("movement_type", "LUNGE")
                    rev_class = chosen.get("reviewed_movement", "")

                    # Contamination Prevention Check 1: Reject explicit REJECT and UNCERTAIN
                    if c_decision in ("REJECT", "UNCERTAIN"):
                        print(f"\n[CONTAMINATION PREVENTED] Candidate '{c_id}' has review status '{c_decision}'.")
                        print("Rejected or uncertain candidates cannot enter the training dataset.")
                        continue

                    # Contamination Prevention Check 2: Block retired classes from entering dataset
                    if rev_class in RETIRED_CLASSES or (c_decision == "ACCEPT" and prop_class in RETIRED_CLASSES and rev_class not in VALID_TARGET_CLASSES):
                        print(f"\n[RETIRED CLASS BLOCKED] Candidate '{c_id}' is associated with retired class '{rev_class or prop_class}'.")
                        print("SCISSOR_JUMP and FOOTWORK_SPLIT_STEP are removed and cannot enter the dataset.")
                        print("Please reclassify to LUNGE, SMASH_LANDING, or CHANGE_OF_DIRECTION, or mark REJECT / UNCERTAIN.")
                        # Force reclassification
                        c_decision = "PENDING"

                    # Contamination Prevention Check 3: Require review for PENDING candidates
                    if c_decision == "PENDING":
                        print(f"\n[REVIEW REQUIRED] Candidate '{c_id}' is unreviewed (PENDING).")
                        print("Proposed classes must never automatically become ground-truth training samples.")
                        confirm_review = input(f"Do you want to human-review and verify this candidate now? [y/N]: ").strip().lower()
                        if confirm_review != "y":
                            print(f"Candidate '{c_id}' skipped. Please run 'python review_candidates.py' first.")
                            continue

                        # Prompt reviewer to confirm or correct movement class
                        print(f"\nConfirm ground-truth movement class for '{c_id}' (Proposed was: {prop_class}):")
                        for m_idx, m_val in MOVEMENT_TYPES.items():
                            print(f"  [{m_idx}] {m_val}")
                        print("  [R] REJECT candidate (not a valid target movement)")
                        print("  [U] Mark UNCERTAIN (cannot verify target movement)")

                        if prop_class in RETIRED_CLASSES:
                            print(f"  Note: Proposed '{prop_class}' is a retired class and cannot be selected as default.")
                            prompt_hint = "Select movement number [1-3], [R]eject, or [U]ncertain: "
                        else:
                            prompt_hint = f"Select movement number (or Enter for proposed '{prop_class}'): "

                        rev_choice = input(prompt_hint).strip()

                        if rev_choice.upper() == "R":
                            chosen["human_decision"] = "REJECT"
                            chosen["reviewed_movement"] = "NO_EVENT"
                            chosen["status"] = "REJECTED"
                            print(f"Candidate '{c_id}' marked as REJECTED. Cannot enter dataset.")
                            try:
                                with open(cand_path, "w", newline="") as fw:
                                    w = csv.DictWriter(fw, fieldnames=list(reader[0].keys()))
                                    w.writeheader()
                                    w.writerows(reader)
                            except Exception:
                                pass
                            continue
                        elif rev_choice.upper() == "U":
                            chosen["human_decision"] = "UNCERTAIN"
                            chosen["reviewed_movement"] = "UNCERTAIN"
                            chosen["status"] = "UNCERTAIN"
                            print(f"Candidate '{c_id}' marked as UNCERTAIN. Cannot enter dataset.")
                            try:
                                with open(cand_path, "w", newline="") as fw:
                                    w = csv.DictWriter(fw, fieldnames=list(reader[0].keys()))
                                    w.writeheader()
                                    w.writerows(reader)
                            except Exception:
                                pass
                            continue

                        if rev_choice in MOVEMENT_TYPES:
                            confirmed_class = MOVEMENT_TYPES[rev_choice]
                        elif not rev_choice and prop_class in VALID_TARGET_CLASSES:
                            confirmed_class = prop_class
                        else:
                            print(f"Invalid selection or retired class. Cannot proceed without a valid target class ({', '.join(VALID_TARGET_CLASSES)}).")
                            continue

                        chosen["human_decision"] = "ACCEPT"
                        chosen["reviewed_movement"] = confirmed_class
                        chosen["status"] = "ACCEPTED"
                        # Persist confirmed decision back to candidates CSV
                        try:
                            with open(cand_path, "w", newline="") as fw:
                                w = csv.DictWriter(fw, fieldnames=list(reader[0].keys()))
                                w.writeheader()
                                w.writerows(reader)
                            print(f"✅ Verified '{c_id}' as '{confirmed_class}' and saved to {cand_path.name}.")
                        except Exception:
                            pass

                    start_time = float(chosen["start_time"])
                    end_time = float(chosen["end_time"])
                    cand_mov = chosen.get("reviewed_movement") or chosen.get("proposed_movement") or chosen.get("movement_type")
                    if cand_mov in VALID_TARGET_CLASSES:
                        suggested_movement = cand_mov
                    else:
                        suggested_movement = None
                    print(f"\nLoaded Candidate '{c_id}': {suggested_movement or 'UNCERTAIN'} ({start_time}s to {end_time}s)")
                    break
                else:
                    print("Invalid selection. Try again.")

    # Prompt for start/end/fps if not specified in CLI or candidates
    start_time = start_time if start_time is not None else prompt_float("Enter start time in seconds: ", 0.0)
    end_time = end_time if end_time is not None else prompt_float("Enter end time in seconds: ", start_time)
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
                world_landmarks = result.pose_world_landmarks[0] if result.pose_world_landmarks else None
                features = extract_pose_features(
                    landmarks, PoseLandmark, image_dimensions=(frame_width, frame_height)
                )
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
                    frame_world_lms = []
                    if world_landmarks:
                        for wlm in world_landmarks:
                            frame_world_lms.append({
                                "x": float(wlm.x),
                                "y": float(wlm.y),
                                "z": float(wlm.z),
                                "visibility": float(wlm.visibility) if hasattr(wlm, "visibility") else 0.0,
                                "presence": float(wlm.presence) if hasattr(wlm, "presence") else 0.0
                            })
                    raw_landmarks_all_frames.append({
                        "frame_index": frame_idx,
                        "timestamp_sec": round(timestamp_sec, 3),
                        "landmarks": frame_lms,
                        "world_landmarks": frame_world_lms
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

    movement_type = prompt_choice("Select Movement Type:", MOVEMENT_TYPES, default_choice=suggested_movement)
    if movement_type not in VALID_TARGET_CLASSES:
        print(f"[ERROR] '{movement_type}' is not one of the valid target classes: {', '.join(VALID_TARGET_CLASSES)}.")
        return

    risk_label = prompt_choice("Select Injury Risk Label:", RISK_LABELS)
    camera_viewpoint = args.camera_viewpoint

    # Save raw landmarks to JSON
    landmarks_file = landmarks_dir / f"{seq_id}.json"
    with open(landmarks_file, "w") as f:
        json.dump({
            "sequence_id": seq_id,
            "video_name": video_path.name,
            "camera_viewpoint": camera_viewpoint,
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
        "player_id": getattr(args, "player_id", "") or "",
        "movement_type": movement_type,
        "camera_viewpoint": camera_viewpoint,
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

    # Prevent dataset contamination: Check for duplicate / heavily overlapping sequences
    file_exists = output_csv.exists()
    if file_exists and output_csv.stat().st_size > 0:
        with open(output_csv, "r", newline="") as existing_file:
            existing_rows = list(csv.DictReader(existing_file))
            for erow in existing_rows:
                if erow.get("video_id") == video_path.name:
                    try:
                        ef_start = int(erow.get("frame_start", 0))
                        ef_end = int(erow.get("frame_end", 0))
                        overlap_frames = min(frame_end, ef_end) - max(frame_start, ef_start)
                        if overlap_frames > int(source_fps * 0.4):
                            overlap_sec = overlap_frames / source_fps
                            print(f"\n⚠️  [DUPLICATE / OVERLAP WARNING] Sequence overlaps by {overlap_sec:.2f}s with existing sample:")
                            print(f"   Existing: '{erow['sequence_id']}' ({erow.get('movement_type')}, frames {ef_start}-{ef_end})")
                            confirm_dup = input("Add overlapping sample to dataset anyway? [y/N]: ").strip().lower()
                            if confirm_dup != "y":
                                print("[CONTAMINATION PREVENTED] Cancelled adding overlapping sequence to dataset.")
                                if temp_clip_path.exists():
                                    temp_clip_path.unlink()
                                return
                    except (ValueError, TypeError):
                        pass

    # Write to CSV
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
