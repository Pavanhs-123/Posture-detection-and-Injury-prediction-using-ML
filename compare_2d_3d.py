#!/usr/bin/env python3
"""
2D vs 3D Pose Feature Comparison Script

Compares normalized 2D landmarks against MediaPipe's 3D pose_world_landmarks
for knee angles, elbow angles, and torso lean (trunk inclination).
Exports a detailed comparison CSV and prints summary statistics.
"""

import argparse
import csv
import os
import sys
from pathlib import Path
import cv2
import sys
sys.modules.setdefault("sounddevice", None)
import mediapipe as mp
import numpy as np

# Ensure project root is in python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pose_analysis import (
    MODEL_PATH,
    ensure_model_file,
    get_knee_angle,
    get_knee_angle_3d,
    get_elbow_angle,
    get_elbow_angle_3d,
    get_torso_lean,
    get_torso_lean_3d,
    calculate_landmark_quality,
)

COMPARISON_FIELDNAMES = [
    "video_id",
    "sequence_id",
    "frame_index",
    "timestamp_sec",
    "pose_detected",
    "left_knee_2d",
    "left_knee_3d",
    "left_knee_diff",
    "right_knee_2d",
    "right_knee_3d",
    "right_knee_diff",
    "left_elbow_2d",
    "left_elbow_3d",
    "left_elbow_diff",
    "right_elbow_2d",
    "right_elbow_3d",
    "right_elbow_diff",
    "trunk_angle_2d",
    "trunk_angle_3d",
    "trunk_angle_diff",
    "left_knee_conf",
    "right_knee_conf",
    "left_elbow_conf",
    "right_elbow_conf",
    "torso_conf",
    "overall_conf",
    "rejected_2d",
    "rejected_3d",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare 2D vs 3D pose features from MediaPipe Pose Landmarker"
    )
    parser.add_argument(
        "--video",
        type=str,
        help="Path to input badminton video file. If not specified, checks data/videos/",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="data/comparison_2d_3d.csv",
        help="Path to export comparison CSV (default: data/comparison_2d_3d.csv)",
    )
    parser.add_argument(
        "--start-time",
        type=float,
        default=0.0,
        help="Start time in seconds (default: 0.0)",
    )
    parser.add_argument(
        "--end-time",
        type=float,
        default=None,
        help="End time in seconds (default: end of video)",
    )
    parser.add_argument(
        "--sample-fps",
        type=float,
        default=10.0,
        help="Sampling rate in frames per second (default: 10.0)",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.5,
        help="Landmark visibility/presence confidence rejection threshold (default: 0.5)",
    )
    return parser.parse_args()


def compute_angle_difference(val_2d: float | None, val_3d: float | None) -> float | None:
    if val_2d is None or val_3d is None:
        return None
    return round(float(val_3d - val_2d), 2)


def process_video_comparison(
    video_path: str,
    output_csv_path: str,
    start_time: float = 0.0,
    end_time: float | None = None,
    sample_fps: float = 10.0,
    min_confidence: float = 0.5,
) -> list[dict]:
    ensure_model_file()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    source_fps = cap.get(cv2.CAP_PROP_FPS)
    if source_fps <= 0:
        source_fps = 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = total_frames / source_fps if total_frames > 0 else 0.0

    start_sec = max(0.0, start_time)
    end_sec = min(video_duration, end_time) if end_time is not None else video_duration
    if end_sec <= start_sec and video_duration > 0:
        end_sec = video_duration

    video_name = Path(video_path).name
    seq_id = f"{Path(video_path).stem}_{start_sec:.1f}to{end_sec:.1f}"

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

    rows: list[dict] = []
    sample_interval = 1.0 / max(sample_fps, 0.1)
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
            if end_sec > start_sec and timestamp_sec > end_sec:
                break

            should_sample = False
            if timestamp_sec >= next_sample_time - 1e-6:
                should_sample = True
                next_sample_time += sample_interval

            if should_sample:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                timestamp_ms = int(timestamp_sec * 1000)
                result = landmarker.detect_for_video(mp_image, timestamp_ms)

                pose_detected = False
                left_k_2d = right_k_2d = left_e_2d = right_e_2d = trunk_2d = None
                left_k_3d = right_k_3d = left_e_3d = right_e_3d = trunk_3d = None
                rejected_2d: list[str] = []
                rejected_3d: list[str] = []
                quality = None

                if result.pose_landmarks and len(result.pose_landmarks) > 0:
                    pose_detected = True
                    lms_2d = result.pose_landmarks[0]
                    quality = calculate_landmark_quality(lms_2d, PoseLandmark)

                    # 2D Calculations with confidence checks
                    left_k_2d, right_k_2d, _ = get_knee_angle(lms_2d, PoseLandmark, min_confidence=min_confidence)
                    left_e_2d, right_e_2d, _ = get_elbow_angle(lms_2d, PoseLandmark, min_confidence=min_confidence)
                    trunk_2d = get_torso_lean(lms_2d, PoseLandmark, min_confidence=min_confidence)

                    if left_k_2d is None:
                        rejected_2d.append("left_knee")
                    if right_k_2d is None:
                        rejected_2d.append("right_knee")
                    if left_e_2d is None:
                        rejected_2d.append("left_elbow")
                    if right_e_2d is None:
                        rejected_2d.append("right_elbow")
                    if trunk_2d is None:
                        rejected_2d.append("trunk_angle")

                    # 3D Calculations using pose_world_landmarks
                    if result.pose_world_landmarks and len(result.pose_world_landmarks) > 0:
                        lms_3d = result.pose_world_landmarks[0]
                        left_k_3d, right_k_3d, _ = get_knee_angle_3d(lms_3d, PoseLandmark, min_confidence=min_confidence)
                        left_e_3d, right_e_3d, _ = get_elbow_angle_3d(lms_3d, PoseLandmark, min_confidence=min_confidence)
                        trunk_3d = get_torso_lean_3d(lms_3d, PoseLandmark, min_confidence=min_confidence)

                        if left_k_3d is None:
                            rejected_3d.append("left_knee")
                        if right_k_3d is None:
                            rejected_3d.append("right_knee")
                        if left_e_3d is None:
                            rejected_3d.append("left_elbow")
                        if right_e_3d is None:
                            rejected_3d.append("right_elbow")
                        if trunk_3d is None:
                            rejected_3d.append("trunk_angle")
                    else:
                        rejected_3d.extend(["left_knee", "right_knee", "left_elbow", "right_elbow", "trunk_angle"])

                row = {
                    "video_id": video_name,
                    "sequence_id": seq_id,
                    "frame_index": frame_idx,
                    "timestamp_sec": round(timestamp_sec, 3),
                    "pose_detected": int(pose_detected),
                    "left_knee_2d": round(left_k_2d, 2) if left_k_2d is not None else None,
                    "left_knee_3d": round(left_k_3d, 2) if left_k_3d is not None else None,
                    "left_knee_diff": compute_angle_difference(left_k_2d, left_k_3d),
                    "right_knee_2d": round(right_k_2d, 2) if right_k_2d is not None else None,
                    "right_knee_3d": round(right_k_3d, 2) if right_k_3d is not None else None,
                    "right_knee_diff": compute_angle_difference(right_k_2d, right_k_3d),
                    "left_elbow_2d": round(left_e_2d, 2) if left_e_2d is not None else None,
                    "left_elbow_3d": round(left_e_3d, 2) if left_e_3d is not None else None,
                    "left_elbow_diff": compute_angle_difference(left_e_2d, left_e_3d),
                    "right_elbow_2d": round(right_e_2d, 2) if right_e_2d is not None else None,
                    "right_elbow_3d": round(right_e_3d, 2) if right_e_3d is not None else None,
                    "right_elbow_diff": compute_angle_difference(right_e_2d, right_e_3d),
                    "trunk_angle_2d": round(trunk_2d, 2) if trunk_2d is not None else None,
                    "trunk_angle_3d": round(trunk_3d, 2) if trunk_3d is not None else None,
                    "trunk_angle_diff": compute_angle_difference(trunk_2d, trunk_3d),
                    "left_knee_conf": quality.left_knee_conf if quality else None,
                    "right_knee_conf": quality.right_knee_conf if quality else None,
                    "left_elbow_conf": quality.left_elbow_conf if quality else None,
                    "right_elbow_conf": quality.right_elbow_conf if quality else None,
                    "torso_conf": quality.torso_conf if quality else None,
                    "overall_conf": quality.overall_conf if quality else None,
                    "rejected_2d": ";".join(rejected_2d) if rejected_2d else "none",
                    "rejected_3d": ";".join(rejected_3d) if rejected_3d else "none",
                }
                rows.append(row)

            frame_idx += 1

    cap.release()

    # Save to CSV
    output_path = Path(output_csv_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COMPARISON_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    return rows


def print_comparison_summary(rows: list[dict], output_csv_path: str):
    total = len(rows)
    detected = sum(1 for r in rows if r["pose_detected"] == 1)

    print("\n" + "=" * 70)
    print("BIOMECHANICAL 2D VS 3D POSE FEATURE COMPARISON SUMMARY")
    print("=" * 70)
    print(f"Total sampled frames: {total}")
    print(f"Frames with pose detected: {detected} ({detected / max(total, 1) * 100:.1f}%)")
    print(f"Saved full comparison log to: {output_csv_path}")

    if detected == 0:
        print("\nNote: No human pose was detected in the processed video segment.")
        print("Please supply a video showing a player's full body.")
        print("=" * 70)
        return

    joints = [
        ("Left Knee", "left_knee_diff", "left_knee_2d", "left_knee_3d"),
        ("Right Knee", "right_knee_diff", "right_knee_2d", "right_knee_3d"),
        ("Left Elbow", "left_elbow_diff", "left_elbow_2d", "left_elbow_3d"),
        ("Right Elbow", "right_elbow_diff", "right_elbow_2d", "right_elbow_3d"),
        ("Trunk Lean", "trunk_angle_diff", "trunk_angle_2d", "trunk_angle_3d"),
    ]

    print(f"\n{'Feature':<14} | {'Valid Pairs':<11} | {'Mean Diff':<10} | {'Median Diff':<11} | {'Std Dev':<8} | {'Max |Diff|':<10}")
    print("-" * 75)

    for name, diff_col, col_2d, col_3d in joints:
        diffs = [abs(r[diff_col]) for r in rows if r[diff_col] is not None]
        signed_diffs = [r[diff_col] for r in rows if r[diff_col] is not None]
        if diffs:
            mean_d = np.mean(signed_diffs)
            median_d = np.median(signed_diffs)
            std_d = np.std(signed_diffs)
            max_d = np.max(diffs)
            print(
                f"{name:<14} | {len(diffs):<11} | {mean_d:>+7.2f}°   | {median_d:>+8.2f}°   | {std_d:>6.2f}° | {max_d:>8.2f}°"
            )
        else:
            print(f"{name:<14} | {'0':<11} | {'N/A':<10} | {'N/A':<11} | {'N/A':<8} | {'N/A':<10}")

    print("=" * 70)
    print("\nBIOMECHANICAL INTERPRETATION & LIMITATIONS:")
    print("1. Perspective Foreshortening in 2D:")
    print("   2D normalized landmarks project joint angles onto the camera's image plane.")
    print("   When a limb moves non-coplanar to the camera (e.g. knee angled toward lens),")
    print("   2D angles foreshorten and deviate from 3D anatomical angles.")
    print("2. Nature of MediaPipe 3D World Landmarks:")
    print("   MediaPipe's pose_world_landmarks are single-camera (monocular) depth estimates.")
    print("   Depth (Z) estimation from 2D images is inherently an ill-posed inverse problem.")
    print("   While 3D coordinates provide metric hip-relative vectors, they are NOT ground-truth")
    print("   motion-capture measurements and can exhibit depth jitter under dynamic movements.")
    print("3. Camera-Angle Dependency:")
    print("   3D world landmarks do NOT automatically eliminate camera-angle problems.")
    print("   Heavy self-occlusion (e.g., trail leg behind lead leg) degrades both 2D and 3D accuracy.")
    print("=" * 70 + "\n")


def main():
    args = parse_args()
    video_path = args.video

    if not video_path:
        # Check standard video locations
        candidates = []
        for search_dir in ["data/videos", "data/clips", "data"]:
            p = Path(search_dir)
            if p.exists():
                candidates.extend(list(p.glob("*.mp4")) + list(p.glob("*.avi")) + list(p.glob("*.mov")))

        if candidates:
            video_path = str(candidates[0])
            print(f"No --video specified. Using discovered clip: {video_path}")
        else:
            print("[INFO] No input video specified and no video files found in data/videos/ or data/clips/.")
            print("\nUsage example:")
            print("  python compare_2d_3d.py --video path/to/badminton_clip.mp4 --output-csv data/comparison_2d_3d.csv")
            print("  python compare_2d_3d.py --video path/to/clip.mp4 --sample-fps 15 --min-confidence 0.6\n")
            return

    if not os.path.exists(video_path):
        print(f"[Error] Video file not found: {video_path}")
        return

    print(f"Running 2D vs 3D pose feature comparison on: {video_path}")
    print(f"Sampling rate: {args.sample_fps} FPS | Min confidence: {args.min_confidence}")

    rows = process_video_comparison(
        video_path=video_path,
        output_csv_path=args.output_csv,
        start_time=args.start_time,
        end_time=args.end_time,
        sample_fps=args.sample_fps,
        min_confidence=args.min_confidence,
    )

    print_comparison_summary(rows, args.output_csv)


if __name__ == "__main__":
    main()
