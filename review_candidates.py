#!/usr/bin/env python3
"""
Human Review Tool for Badminton Movement Candidates

Enables sports analysts and researchers to review, accept, reject, or reclassify
automatically detected candidate movement proposals before they enter the training dataset.

The tool strictly prevents unreviewed, rejected, or uncertain candidates from entering
the dataset without explicit human confirmation.
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.detection.candidate_detector import (
    MovementCandidate,
    CANDIDATE_CSV_FIELDS,
)

TARGET_MOVEMENT_CLASSES = [
    "LUNGE",
    "SMASH_LANDING",
    "CHANGE_OF_DIRECTION",
]

VALID_MOVEMENT_CLASSES = [
    "LUNGE",
    "SMASH_LANDING",
    "CHANGE_OF_DIRECTION",
    "NO_EVENT",
    "UNCERTAIN",
]

RETIRED_CLASSES = [
    "SCISSOR_JUMP",
    "FOOTWORK_SPLIT_STEP",
]

DECISIONS = ["PENDING", "ACCEPT", "REJECT", "UNCERTAIN"]


def load_candidates_from_csv(csv_path: str) -> List[Dict[str, str]]:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Candidate CSV not found: {csv_path}")

    with open(path, "r", newline="") as f:
        reader = list(csv.DictReader(f))
    return reader


def save_candidates_to_csv(candidates: List[Dict[str, str]], output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CANDIDATE_CSV_FIELDS)
        writer.writeheader()
        for cand in candidates:
            # Ensure all required fields exist
            row = {}
            for field in CANDIDATE_CSV_FIELDS:
                row[field] = cand.get(field, "")
            writer.writerow(row)


def calculate_review_summary(candidates: List[Dict[str, str]]) -> Dict[str, int]:
    summary = {
        "total": len(candidates),
        "pending": 0,
        "accepted": 0,
        "accepted_target": 0,
        "accepted_retired": 0,
        "rejected": 0,
        "uncertain": 0,
    }
    for c in candidates:
        dec = (c.get("human_decision") or "PENDING").upper()
        rev_mov = (c.get("reviewed_movement") or "").strip().upper()
        if dec == "ACCEPT":
            summary["accepted"] += 1
            if rev_mov in RETIRED_CLASSES:
                summary["accepted_retired"] += 1
            elif rev_mov in TARGET_MOVEMENT_CLASSES:
                summary["accepted_target"] += 1
        elif dec == "REJECT":
            summary["rejected"] += 1
        elif dec == "UNCERTAIN":
            summary["uncertain"] += 1
        else:
            summary["pending"] += 1
    return summary


def find_overlapping_candidates(candidates: List[Dict[str, str]], target_idx: int) -> List[tuple[Dict[str, str], float]]:
    """Identifies any other candidates in the same video that temporally overlap with the target candidate."""
    target = candidates[target_idx]
    v_target = target.get("video_id", "")
    try:
        t_s = float(target.get("start_time", 0.0))
        t_e = float(target.get("end_time", 0.0))
    except (ValueError, TypeError):
        return []

    overlaps = []
    for i, other in enumerate(candidates):
        if i == target_idx:
            continue
        if other.get("video_id") != v_target:
            continue
        try:
            o_s = float(other.get("start_time", 0.0))
            o_e = float(other.get("end_time", 0.0))
            overlap_dur = min(t_e, o_e) - max(t_s, o_s)
            if overlap_dur > 0.05:  # Overlap greater than 50ms
                overlaps.append((other, round(overlap_dur, 2)))
        except (ValueError, TypeError):
            continue
    return overlaps


def print_summary_table(summary: Dict[str, int], csv_path: str) -> None:
    total = max(summary["total"], 1)
    print("\n" + "=" * 65)
    print(f"📊 CANDIDATE HUMAN-REVIEW DIAGNOSTICS: {Path(csv_path).name}")
    print("=" * 65)
    print(f"Total Candidates:   {summary['total']}")
    print(f"  ✅ Accepted:      {summary['accepted']:<4} ({summary['accepted']/total*100:.1f}%) -> Total approved")
    print(f"     • Valid Target Classes ({', '.join(TARGET_MOVEMENT_CLASSES)}): {summary.get('accepted_target', 0)}")
    if summary.get("accepted_retired", 0) > 0:
        print(f"     ⚠️  Retired Classes (SCISSOR_JUMP/SPLIT_STEP): {summary['accepted_retired']} (Blocked from dataset!)")
    print(f"  ❌ Rejected:      {summary['rejected']:<4} ({summary['rejected']/total*100:.1f}%) -> Blocked from dataset")
    print(f"  ❓ Uncertain:     {summary['uncertain']:<4} ({summary['uncertain']/total*100:.1f}%) -> Blocked from dataset")
    print(f"  ⏳ Awaiting Review:{summary['pending']:<4} ({summary['pending']/total*100:.1f}%) -> Blocked from dataset")
    print("=" * 65 + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Review, verify, and validate badminton movement candidate proposals"
    )
    parser.add_argument(
        "--candidates-csv",
        type=str,
        default="data/candidates.csv",
        help="Path to candidates CSV (default: data/candidates.csv)",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        help="Path to save updated review CSV (default: overwrites candidates-csv)",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print diagnostic summary of candidate review statuses and exit",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        default=True,
        help="Run interactive human review terminal workflow (default: True)",
    )
    parser.add_argument(
        "--filter-status",
        type=str,
        choices=["ALL", "PENDING", "ACCEPTED", "REJECTED", "UNCERTAIN"],
        default="PENDING",
        help="Which candidates to review in interactive mode (default: PENDING)",
    )
    return parser.parse_args()


def interactive_review(candidates: List[Dict[str, str]], filter_status: str, output_path: str) -> None:
    print("\n" + "=" * 70)
    print("🏸 BADMINTON CANDIDATE HUMAN REVIEW WORKFLOW")
    print("=" * 70)
    print("Controls for each candidate:")
    print("  [A] ACCEPT candidate (confirms proposed class as ground truth)")
    print("  [C] CORRECT class (reclassify to another movement type)")
    print("  [R] REJECT candidate (false positive, non-badminton movement, or discarded)")
    print("  [U] Mark UNCERTAIN (cannot determine from video/posture)")
    print("  [S] SKIP to next candidate without making a decision")
    print("  [Q] QUIT and save all progress")
    print("-" * 70)

    to_review = []
    for idx, c in enumerate(candidates):
        dec = (c.get("human_decision") or "PENDING").upper()
        if filter_status == "ALL":
            to_review.append(idx)
        elif filter_status == "PENDING" and dec == "PENDING":
            to_review.append(idx)
        elif filter_status == "ACCEPTED" and dec == "ACCEPT":
            to_review.append(idx)
        elif filter_status == "REJECTED" and dec == "REJECT":
            to_review.append(idx)
        elif filter_status == "UNCERTAIN" and dec == "UNCERTAIN":
            to_review.append(idx)

    if not to_review:
        print(f"No candidates found matching filter: {filter_status}")
        return

    print(f"Found {len(to_review)} candidate(s) to review under filter '{filter_status}'.\n")

    for step_num, cand_idx in enumerate(to_review, start=1):
        c = candidates[cand_idx]
        cand_id = c.get("candidate_id", f"cand_{cand_idx+1}")
        video_id = c.get("video_id", "Unknown")
        t_start = c.get("start_time", "0.0")
        t_end = c.get("end_time", "0.0")
        dur = c.get("duration", "0.0")
        prop_class = c.get("proposed_movement") or c.get("movement_type", "UNCERTAIN")
        score = c.get("heuristic_score") or c.get("confidence_score", "0.0")
        curr_decision = c.get("human_decision") or "PENDING"
        curr_reviewed_class = c.get("reviewed_movement", "")

        metrics_str = c.get("key_metrics", "{}")
        try:
            metrics_dict = json.loads(metrics_str)
            metrics_formatted = ", ".join(f"{k}: {v}" for k, v in metrics_dict.items())
        except Exception:
            metrics_formatted = metrics_str

        print("-" * 70)
        print(f"[{step_num}/{len(to_review)}] CANDIDATE: {cand_id} | Video: {video_id}")
        print(f"  Time Interval:   {t_start}s to {t_end}s (Duration: {dur}s)")
        print(f"  Proposed Class:  {prop_class}")
        print(f"  Heuristic Score: {score}  (Heuristic rule match score, NOT a probability)")
        print(f"  Key Metrics:     {metrics_formatted}")
        print(f"  Current Status:  {curr_decision} (Reviewed class: {curr_reviewed_class or 'None'})")

        # Check for temporal overlaps with other candidates from the same video
        overlaps = find_overlapping_candidates(candidates, cand_idx)
        if overlaps:
            print("  ⚠️  [OVERLAP ALERT] Overlaps with existing candidate(s) in same video:")
            for other_cand, ov_dur in overlaps:
                o_id = other_cand.get("candidate_id", "Unknown")
                o_class = other_cand.get("reviewed_movement") or other_cand.get("proposed_movement", "N/A")
                o_dec = other_cand.get("human_decision", "PENDING")
                print(f"      • Overlaps by {ov_dur:.2f}s with {o_id} (Class: {o_class}, Decision: {o_dec})")
        print("-" * 70)

        while True:
            choice = input("Decision [A]ccept, [C]orrect, [R]eject, [U]ncertain, [S]kip, [Q]uit: ").strip().upper()
            if choice == "A":
                if prop_class not in TARGET_MOVEMENT_CLASSES:
                    print(f"  ❌ Cannot accept '{prop_class}'! This class is retired or invalid.")
                    print(f"  Valid target classes are: {', '.join(TARGET_MOVEMENT_CLASSES)}.")
                    print("  Please [C]orrect to a target class, mark [U]ncertain, or [R]eject.")
                    continue
                c["human_decision"] = "ACCEPT"
                c["reviewed_movement"] = prop_class
                c["status"] = "ACCEPTED"
                notes = input("Reviewer notes (optional, press Enter to skip): ").strip()
                if notes:
                    c["reviewer_notes"] = notes
                print(f"  -> Accepted as '{prop_class}'")
                break
            elif choice == "C":
                print("\nSelect corrected movement class:")
                for i, mc in enumerate(VALID_MOVEMENT_CLASSES, start=1):
                    print(f"  [{i}] {mc}")
                sel = input("Choice (number): ").strip()
                if sel.isdigit() and 1 <= int(sel) <= len(VALID_MOVEMENT_CLASSES):
                    chosen_class = VALID_MOVEMENT_CLASSES[int(sel) - 1]
                    if chosen_class == "NO_EVENT":
                        c["human_decision"] = "REJECT"
                        c["reviewed_movement"] = "NO_EVENT"
                        c["status"] = "REJECTED"
                        print("  -> Rejected as NO_EVENT")
                    elif chosen_class == "UNCERTAIN":
                        c["human_decision"] = "UNCERTAIN"
                        c["reviewed_movement"] = "UNCERTAIN"
                        c["status"] = "UNCERTAIN"
                        print("  -> Marked as UNCERTAIN")
                    else:
                        c["human_decision"] = "ACCEPT"
                        c["reviewed_movement"] = chosen_class
                        c["status"] = "ACCEPTED"
                        print(f"  -> Accepted with corrected class: '{chosen_class}'")
                    notes = input("Reviewer notes (optional, press Enter to skip): ").strip()
                    if notes:
                        c["reviewer_notes"] = notes
                    break
                else:
                    print("Invalid choice. Try again.")
            elif choice == "R":
                c["human_decision"] = "REJECT"
                c["reviewed_movement"] = "NO_EVENT"
                c["status"] = "REJECTED"
                notes = input("Reason for rejection (optional): ").strip()
                if notes:
                    c["reviewer_notes"] = notes
                print("  -> Marked as REJECTED")
                break
            elif choice == "U":
                c["human_decision"] = "UNCERTAIN"
                c["reviewed_movement"] = "UNCERTAIN"
                c["status"] = "UNCERTAIN"
                notes = input("Notes on uncertainty (optional): ").strip()
                if notes:
                    c["reviewer_notes"] = notes
                print("  -> Marked as UNCERTAIN")
                break
            elif choice == "S":
                print("  -> Skipped")
                break
            elif choice == "Q":
                print("\nSaving progress and exiting...")
                save_candidates_to_csv(candidates, output_path)
                print(f"Saved review updates to: {output_path}")
                return
            else:
                print("Invalid input. Please choose A, C, R, U, S, or Q.")

        # Auto-save after each candidate
        save_candidates_to_csv(candidates, output_path)

    print("\nReview cycle completed!")
    save_candidates_to_csv(candidates, output_path)
    summary = calculate_review_summary(candidates)
    print_summary_table(summary, output_path)


def main():
    args = parse_args()
    csv_path = args.candidates_csv
    out_path = args.output_csv or csv_path

    try:
        candidates = load_candidates_from_csv(csv_path)
    except FileNotFoundError as e:
        print(f"[Error] {e}")
        return

    summary = calculate_review_summary(candidates)

    if args.summary:
        print_summary_table(summary, csv_path)
        return

    print_summary_table(summary, csv_path)
    interactive_review(candidates, args.filter_status, out_path)


if __name__ == "__main__":
    main()
