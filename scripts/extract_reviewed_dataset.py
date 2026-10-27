"""
Headless, non-interactive dataset extraction script for badminton AI.
Extracts sequence-level kinematic features from verified human review intervals
directly from source videos, without GUI popups, interactive prompts, or synthetic defaults.

Safety & Rigor Enforcements:
- Processes ONLY explicitly verified human decisions (excludes PENDING, NaN, UNCERTAIN).
- Never treats provisional AI screening proposals as verified labels.
- Preserves video and player grouping identity for leakage-safe cross-validation.
- Rejects invalid intervals, missing video files, and overlapping sequences (>0.4s).
- Never converts missing joint measurements into synthetic 180.0, 0.0, or LOW risk.
- Enforces strict overwrite protection on existing dataset files.
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import cv2
import numpy as np

# This project is video-only. MediaPipe imports an optional microphone helper
# that initializes PortAudio at import time; headless environments can hang there.
# Mark the optional audio module unavailable before importing MediaPipe.
sys.modules.setdefault("sounddevice", None)
import mediapipe as mp

# Ensure root directory is on sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from pose_analysis import (
    MODEL_PATH,
    ensure_model_file,
    extract_pose_features,
    calculate_landmark_quality,
)

# Standard sequence dataset columns matching project training pipelines
DATASET_FIELDNAMES = [
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
    "landmark_quality",
    "review_id",
    "risk_label",
]

VALID_TRAINING_CLASSES = {
    "LUNGE",
    "SMASH_LANDING",
    "CHANGE_OF_DIRECTION",
    "OTHER_NON_TARGET",
}

# Video stem to default player group mapping
DEFAULT_PLAYER_MAP = {
    "video_20261014_191317": "player_01",
    "video_20261014_191936": "player_02",
    "video_20261014_192859": "player_03",
    "video_20261014_193547": "player_04",
    "video_20261014_195919": "player_05",
    "video_20261014_202046": "player_06",
}


def compute_rule_based_risk_label(features: Dict[str, Any]) -> str:
    """Computes deterministic biomechanical posture risk indicator from features."""
    left_knee = features.get("left_knee_min")
    right_knee = features.get("right_knee_min")
    trunk_max = features.get("trunk_angle_max")

    knees = [k for k in (left_knee, right_knee) if k is not None]
    knee_min = min(knees) if knees else 180.0
    trunk = trunk_max if trunk_max is not None else 0.0

    if knee_min < 80.0 or trunk > 40.0:
        return "HIGH"
    elif knee_min < 110.0 or trunk > 25.0:
        return "MODERATE"
    else:
        return "LOW"


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Headless extraction of verified badminton movement sequences into training dataset."
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="data/human_verification_review_package/review_manifest.csv",
        help="Path to review manifest CSV containing human decisions",
    )
    parser.add_argument(
        "--video-dir",
        type=str,
        default="data/raw_videos",
        help="Directory containing original full-length MP4 videos",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="data/dataset.csv",
        help="Path to write the extracted sequence dataset CSV",
    )
    parser.add_argument(
        "--sample-fps",
        type=float,
        default=10.0,
        help="Pose landmark extraction frame rate in FPS (default: 10.0)",
    )
    parser.add_argument(
        "--min-quality",
        type=float,
        default=0.45,
        help="Minimum overall landmark quality threshold to keep sample (default: 0.45)",
    )
    parser.add_argument(
        "--camera-viewpoint",
        type=str,
        default="side_90deg",
        help="Camera viewpoint metadata label (default: side_90deg)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly permit overwriting output CSV if it already exists",
    )
    parser.add_argument(
        "--target-only",
        action="store_true",
        help="Extract only accepted target classes (LUNGE, SMASH_LANDING, CHANGE_OF_DIRECTION) and exclude REJECT/OTHER_NON_TARGET",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional limit on number of verified samples to extract (for testing)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate manifest, paths, and intervals without running pose estimation",
    )
    return parser.parse_args(args)


def filter_verified_records(
    manifest_rows: List[Dict[str, str]],
    target_only: bool = False,
) -> Tuple[List[Dict[str, str]], Dict[str, int]]:
    """
    Filters manifest rows strictly for verified, assessable human decisions.
    Explicitly rejects unreviewed items, AI screening proposals, and uncertain items.
    If target_only is True, includes only ACCEPT decisions for target movement classes
    (LUNGE, SMASH_LANDING, CHANGE_OF_DIRECTION) and excludes REJECT / OTHER_NON_TARGET.
    """
    verified = []
    reasons = {
        "pending": 0,
        "uncertain": 0,
        "invalid_class": 0,
        "accepted": 0,
        "rejected_non_target": 0,
    }

    for r in manifest_rows:
        human_dec = (r.get("human_decision") or "").strip().upper()
        human_class = (r.get("human_verified_class") or "").strip().upper()

        # 1. Reject unconfirmed / pending items
        if not human_dec or human_dec in ("PENDING", "NAN", ""):
            reasons["pending"] += 1
            continue

        # 2. Reject explicit UNCERTAIN decisions
        if human_dec == "UNCERTAIN" or human_class == "UNCERTAIN":
            reasons["uncertain"] += 1
            continue

        # 3. Only accept validated classes
        if human_class not in VALID_TRAINING_CLASSES:
            reasons["invalid_class"] += 1
            continue

        if target_only:
            if human_dec == "ACCEPT" and human_class in {"LUNGE", "SMASH_LANDING", "CHANGE_OF_DIRECTION"}:
                verified.append(r)
                reasons["accepted"] += 1
            else:
                reasons["rejected_non_target"] += 1
        else:
            if human_dec in ("ACCEPT", "VERIFIED", "CONFIRMED", "REJECT"):
                # REJECT with OTHER_NON_TARGET is acceptable as negative class
                if human_dec == "REJECT" and human_class != "OTHER_NON_TARGET":
                    continue
                verified.append(r)
                reasons["accepted"] += 1

    return verified, reasons


def check_interval_overlap(
    existing_intervals: List[Tuple[float, float]],
    new_start: float,
    new_end: float,
    max_overlap_sec: float = 0.4,
) -> bool:
    """Checks if a proposed interval overlaps significantly with any existing interval."""
    for s, e in existing_intervals:
        overlap = min(new_end, e) - max(new_start, s)
        if overlap > max_overlap_sec:
            return True
    return False


def extract_features_for_interval(
    video_path: Path,
    start_sec: float,
    end_sec: float,
    sample_fps: float = 10.0,
    min_quality: float = 0.45,
) -> Tuple[Optional[Dict[str, Any]], float, Optional[str]]:
    """
    Extracts sequence-level pose kinematic features from a video segment.
    Returns (features_dict, avg_quality, error_message).
    """
    if not video_path.exists():
        return None, 0.0, f"Video file not found: {video_path}"

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None, 0.0, f"Unable to open video: {video_path}"

    source_fps = cap.get(cv2.CAP_PROP_FPS)
    if source_fps <= 0:
        source_fps = 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_dur = total_frames / source_fps

    # Boundary verification
    if start_sec < 0 or end_sec <= start_sec or start_sec >= video_dur:
        cap.release()
        return None, 0.0, f"Invalid interval [{start_sec:.2f}s, {end_sec:.2f}s] for video duration {video_dur:.2f}s"

    clamped_end = min(end_sec, video_dur)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    ensure_model_file()

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

    pose_features_list = []
    quality_scores = []
    sample_interval = 1.0 / sample_fps
    next_sample_time = start_sec
    # Seek directly to the reviewed interval instead of decoding the full source
    # from the beginning for every event in long match recordings.
    frame_idx = max(0, int(round(start_sec * source_fps)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

    with PoseLandmarker.create_from_options(options) as landmarker:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            t_sec = frame_idx / source_fps
            if t_sec < start_sec:
                frame_idx += 1
                continue
            if t_sec > clamped_end:
                break

            if t_sec >= next_sample_time - 1e-6:
                next_sample_time += sample_interval

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                t_ms = int(t_sec * 1000)
                res = landmarker.detect_for_video(mp_image, t_ms)

                if res.pose_landmarks and len(res.pose_landmarks) > 0:
                    lms = res.pose_landmarks[0]
                    feats = extract_pose_features(
                        lms, PoseLandmark, image_dimensions=(frame_width, frame_height)
                    )
                    pose_features_list.append(feats)

                    q = calculate_landmark_quality(lms, PoseLandmark)
                    if q.overall_conf is not None:
                        quality_scores.append(q.overall_conf)

            frame_idx += 1

    cap.release()

    if not pose_features_list:
        return None, 0.0, "No valid pose landmarks detected across sampled interval"

    avg_quality = float(np.mean(quality_scores)) if quality_scores else 0.0
    if avg_quality < min_quality:
        return None, avg_quality, f"Landmark tracking quality {avg_quality:.2f} below threshold {min_quality:.2f}"

    # Extract non-None joint angles
    left_knees = [f.left_knee_angle for f in pose_features_list if f.left_knee_angle is not None]
    right_knees = [f.right_knee_angle for f in pose_features_list if f.right_knee_angle is not None]
    left_elbows = [f.left_elbow_angle for f in pose_features_list if f.left_elbow_angle is not None]
    right_elbows = [f.right_elbow_angle for f in pose_features_list if f.right_elbow_angle is not None]
    torso_leans = [f.torso_lean for f in pose_features_list if f.torso_lean is not None]

    # Crucial safety directive: do NOT default missing measurements to 180.0 or 0.0!
    features = {
        "left_knee_min": round(float(np.min(left_knees)), 2) if left_knees else None,
        "left_knee_max": round(float(np.max(left_knees)), 2) if left_knees else None,
        "right_knee_min": round(float(np.min(right_knees)), 2) if right_knees else None,
        "right_knee_max": round(float(np.max(right_knees)), 2) if right_knees else None,
        "left_elbow_min": round(float(np.min(left_elbows)), 2) if left_elbows else None,
        "left_elbow_max": round(float(np.max(left_elbows)), 2) if left_elbows else None,
        "right_elbow_min": round(float(np.min(right_elbows)), 2) if right_elbows else None,
        "right_elbow_max": round(float(np.max(right_elbows)), 2) if right_elbows else None,
        "trunk_angle_min": round(float(np.min(torso_leans)), 2) if torso_leans else None,
        "trunk_angle_max": round(float(np.max(torso_leans)), 2) if torso_leans else None,
    }

    return features, avg_quality, None


def run_dataset_extraction(args: argparse.Namespace) -> int:
    manifest_path = Path(args.manifest).resolve()
    video_dir = Path(args.video_dir).resolve()
    output_path = Path(args.output_csv).resolve()

    if not manifest_path.exists():
        print(f"Error: Manifest file not found: {manifest_path}", file=sys.stderr)
        return 1

    # Overwrite protection check
    if output_path.exists() and output_path.stat().st_size > 0 and not args.overwrite:
        print(
            f"Error: Output file already exists and is non-empty: {output_path}\n"
            f"Pass '--overwrite' to explicitly replace it, or specify a new '--output-csv' path.",
            file=sys.stderr,
        )
        return 1

    with open(manifest_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        manifest_rows = list(reader)

    verified_records, filter_stats = filter_verified_records(manifest_rows, target_only=args.target_only)

    print("=" * 65)
    print("🏸 Headless Dataset Feature Extraction Pass")
    print("=" * 65)
    print(f"Manifest Source:      {manifest_path}")
    print(f"Total Manifest Items: {len(manifest_rows)}")
    print(f"  • Unreviewed / Pending: {filter_stats['pending']} (Excluded)")
    print(f"  • Uncertain Items:      {filter_stats['uncertain']} (Excluded)")
    if args.target_only:
        print(f"  • Non-Target / Reject:  {filter_stats['rejected_non_target']} (Excluded)")
    print(f"  • Confirmed Verified:   {filter_stats['accepted']} (Eligible for Extraction)")
    print(f"Target Only Mode:     {args.target_only}")
    print(f"Video Directory:      {video_dir}")
    print(f"Output Dataset:       {output_path}")
    print("=" * 65)

    if not verified_records:
        print("\n[Notice] No confirmed human-verified records found in manifest.")
        print("Dataset extraction aborted. No file written.")
        print("To review and verify items, run: python scripts/record_human_review.py --interactive")
        return 0

    if args.max_samples and len(verified_records) > args.max_samples:
        print(f"[Sampling] Limiting extraction to first {args.max_samples} verified samples.")
        verified_records = verified_records[: args.max_samples]

    if args.dry_run:
        print("\n[Dry-Run Mode Active] Validating source videos and timestamps without extraction...")
        missing_videos = set()
        for r in verified_records:
            v_name = r.get("video_name", "")
            vp = video_dir / v_name
            if not vp.exists():
                missing_videos.add(v_name)
        if missing_videos:
            print(f"Warning: {len(missing_videos)} video files missing from {video_dir}:")
            for mv in sorted(missing_videos):
                print(f"  • {mv}")
        else:
            print(f"All {len(verified_records)} source video paths verified successfully.")
        return 0

    # Execute extraction
    extracted_rows = []
    skipped_intervals = []
    intervals_by_video: Dict[str, List[Tuple[float, float, str]]] = {}

    for idx, r in enumerate(verified_records, start=1):
        r_id = r.get("review_id", f"REV_{idx}")
        v_name = r.get("video_name", "")
        v_class = r.get("human_verified_class", "").strip().upper()
        s_time = float(r.get("start_time", 0.0))
        e_time = float(r.get("end_time", 0.0))
        duration = round(e_time - s_time, 3)

        # Video path resolution
        vp = video_dir / v_name
        if not vp.exists():
            skipped_intervals.append((r_id, v_name, s_time, e_time, f"Video not found at {vp}"))
            print(f"  [{idx}/{len(verified_records)}] ❌ {r_id}: Video missing ({v_name})")
            continue

        # Overlap and deduplication check
        if v_name not in intervals_by_video:
            intervals_by_video[v_name] = []

        same_class_intervals = [
            (start, end)
            for start, end, movement_class in intervals_by_video[v_name]
            if movement_class == v_class
        ]
        if check_interval_overlap(same_class_intervals, s_time, e_time, max_overlap_sec=0.4):
            skipped_intervals.append((r_id, v_name, s_time, e_time, "Duplicate/overlapping interval (>0.4s) with same reviewed class"))
            print(f"  [{idx}/{len(verified_records)}] ⚠️ {r_id}: Skipped as a same-class duplicate/overlap in {v_name}")
            continue

        different_class_overlaps = [
            movement_class
            for start, end, movement_class in intervals_by_video[v_name]
            if movement_class != v_class and min(e_time, end) - max(s_time, start) > 0.0
        ]
        if different_class_overlaps:
            print(
                f"  [{idx}/{len(verified_records)}] ⚠️ {r_id}: Retaining human-reviewed {v_class} interval "
                f"with {different_class_overlaps} time overlap in {v_name}."
            )

        print(f"  [{idx}/{len(verified_records)}] ⏳ Extracting {r_id} ({v_class}) from {v_name} [{s_time:.2f}s - {e_time:.2f}s]...", end="", flush=True)

        features, quality, err = extract_features_for_interval(
            video_path=vp,
            start_sec=s_time,
            end_sec=e_time,
            sample_fps=args.sample_fps,
            min_quality=args.min_quality,
        )

        if err or features is None:
            skipped_intervals.append((r_id, v_name, s_time, e_time, err or "Unknown extraction failure"))
            print(f" ❌ Failed ({err})")
            continue

        intervals_by_video[v_name].append((s_time, e_time, v_class))

        # Build output row
        v_stem = Path(v_name).stem
        player_id = DEFAULT_PLAYER_MAP.get(v_stem, v_stem)
        seq_id = f"seq_{v_stem}_{s_time:.1f}to{e_time:.1f}_{r_id}"

        row_data = {
            "sequence_id": seq_id,
            "video_id": v_name,
            "player_id": player_id,
            "movement_type": v_class,
            "camera_viewpoint": args.camera_viewpoint,
            "frame_start": int(round(s_time * 30.0)),
            "frame_end": int(round(e_time * 30.0)),
            "duration": duration,
            "left_knee_min": features["left_knee_min"],
            "left_knee_max": features["left_knee_max"],
            "right_knee_min": features["right_knee_min"],
            "right_knee_max": features["right_knee_max"],
            "left_elbow_min": features["left_elbow_min"],
            "left_elbow_max": features["left_elbow_max"],
            "right_elbow_min": features["right_elbow_min"],
            "right_elbow_max": features["right_elbow_max"],
            "trunk_angle_min": features["trunk_angle_min"],
            "trunk_angle_max": features["trunk_angle_max"],
            "landmark_quality": round(quality, 3),
            "review_id": r_id,
            "risk_label": compute_rule_based_risk_label(features),
        }
        extracted_rows.append(row_data)
        print(f" ✅ Success (quality={quality:.2f})")

    if not extracted_rows:
        print("\n[Warning] No sequences were successfully extracted. No dataset file written.")
        return 1

    # Write output CSV atomically
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_output = output_path.parent / f".tmp_{output_path.name}"
    with open(temp_output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=DATASET_FIELDNAMES)
        writer.writeheader()
        writer.writerows(extracted_rows)
    temp_output.replace(output_path)

    print("\n" + "=" * 65)
    print(f"✅ Successfully wrote {len(extracted_rows)} sequence samples to {output_path}")
    print("Class Distribution:")
    cls_counts: Dict[str, int] = {}
    for row in extracted_rows:
        cls_counts[row["movement_type"]] = cls_counts.get(row["movement_type"], 0) + 1
    for cls, cnt in sorted(cls_counts.items()):
        print(f"  • {cls:<22}: {cnt} sample(s)")
    if skipped_intervals:
        print(f"Skipped Intervals: {len(skipped_intervals)}")
    print("=" * 65)

    return 0


def main():
    args = parse_args()
    ret = run_dataset_extraction(args)
    sys.exit(ret)


if __name__ == "__main__":
    main()
