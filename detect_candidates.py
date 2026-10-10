#!/usr/bin/env python3
"""
CLI Tool for Automatic Badminton Movement Candidate Detection

Scans a badminton video and proposes movement candidate segments for:
- LUNGE
- SMASH_LANDING
- SCISSOR_JUMP
- CHANGE_OF_DIRECTION
- FOOTWORK_SPLIT_STEP

Saves proposed candidates to CSV for human review and optionally exports preview clips.
"""

import argparse
import os
import sys
from pathlib import Path

# Add project root to python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.detection.candidate_detector import (
    MovementCandidateDetector,
    DetectorConfig,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect candidate badminton movement segments in video"
    )
    parser.add_argument(
        "--video",
        type=str,
        help="Path to input badminton video file. If not specified, searches data/videos/",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="data/candidates.csv",
        help="Path to export candidate suggestions CSV (default: data/candidates.csv)",
    )
    parser.add_argument(
        "--export-clips",
        action="store_true",
        help="Slice and export MP4 preview clips for each detected candidate into data/clips/",
    )
    parser.add_argument(
        "--clips-dir",
        type=str,
        default="data/clips",
        help="Directory to save exported candidate clips (default: data/clips)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=15.0,
        help="Sampling rate in frames per second for temporal signal scanning (default: 15.0)",
    )
    parser.add_argument(
        "--lunge-knee-thresh",
        type=float,
        default=120.0,
        help="Maximum knee angle in degrees to trigger lunge candidate (default: 120.0)",
    )
    parser.add_argument(
        "--jump-elev-thresh",
        type=float,
        default=0.045,
        help="Minimum vertical hip elevation ratio to trigger jump candidates (default: 0.045)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    video_path = args.video

    if not video_path:
        # Search data/videos, data/clips, data/
        candidates = []
        for search_dir in ["data/videos", "data/clips", "data"]:
            p = Path(search_dir)
            if p.exists():
                candidates.extend(list(p.glob("*.mp4")) + list(p.glob("*.avi")) + list(p.glob("*.mov")))

        if candidates:
            video_path = str(candidates[0])
            print(f"No --video specified. Using discovered video: {video_path}")
        else:
            print("[INFO] No input video specified and no video files found in data/videos/ or data/clips/.")
            print("\nUsage example:")
            print("  python detect_candidates.py --video path/to/badminton_game.mp4 --output-csv data/candidates.csv")
            print("  python detect_candidates.py --video path/to/video.mp4 --export-clips --fps 15\n")
            return

    if not os.path.exists(video_path):
        print(f"[Error] Video file not found: {video_path}")
        return

    print("=" * 70)
    print("🏸 AUTOMATIC BADMINTON MOVEMENT CANDIDATE DETECTOR")
    print("=" * 70)
    print(f"Video: {video_path}")
    print(f"Sampling Rate: {args.fps} FPS")
    print(f"Output CSV: {args.output_csv}")
    print(f"Export Clips: {args.export_clips}")
    print("\nScanning video for movement candidates (MediaPipe Pose Tracking)...")

    config = DetectorConfig(
        sample_fps=args.fps,
        lunge_knee_threshold=args.lunge_knee_thresh,
        jump_elevation_min=args.jump_elev_thresh,
    )

    detector = MovementCandidateDetector(config=config)
    candidates = detector.detect_video_candidates(
        video_path=video_path,
        output_csv=args.output_csv,
        export_clips=args.export_clips,
        clips_dir=args.clips_dir,
    )

    print("\n" + "=" * 78)
    print(f"DETECTION COMPLETE: Found {len(candidates)} movement candidate proposal(s)")
    print("=" * 78)

    if not candidates:
        print("No movement candidates exceeded detection thresholds in this video.")
        print("Note: Ensure player full body is visible and movements correspond to supported categories.")
    else:
        print(f"{'ID':<10} | {'Proposed Class':<22} | {'Time (s)':<15} | {'Duration':<9} | {'Score*':<8} | {'Status':<10}")
        print("-" * 88)
        for c in candidates:
            time_range = f"{c.start_time:.2f}s - {c.end_time:.2f}s"
            disp_class = c.proposed_movement or c.movement_type
            disp_score = c.heuristic_score if c.heuristic_score > 0 else c.confidence_score
            disp_status = c.status if c.status != "PROPOSED" else "PENDING"
            print(f"{c.candidate_id:<10} | {disp_class:<22} | {time_range:<15} | {c.duration:.2f}s    | {disp_score:<8.2f} | {disp_status:<10}")

        print("\n* NOTE: Score is an uncalibrated heuristic rule-match score (0.0-1.0), NOT a probability.")
        print("Proposed movements are candidate suggestions and MUST undergo human review before entering dataset.")
        print("\n" + "=" * 78)
        print(f"✅ Proposed candidates exported to: {args.output_csv}")
        if args.export_clips:
            print(f"✅ Preview clips exported to: {args.clips_dir}")
        print("\nNext step: Run the Human Review Workflow to accept, reject, or correct candidates:")
        print(f"  python review_candidates.py --candidates-csv {args.output_csv}")
        print("Then use dataset_builder.py to assign human injury-risk labels:")
        print(f"  python dataset_builder.py --video {video_path} --candidates-csv {args.output_csv}")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    main()
