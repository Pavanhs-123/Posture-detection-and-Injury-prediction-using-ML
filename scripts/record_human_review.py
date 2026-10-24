"""
CLI and programmatic tool to safely record explicit human review decisions
into the review manifest without manufacturing ground truth or bulk-approving
provisional AI judgments.

Enforces:
- Allowed movement classes: LUNGE, SMASH_LANDING, CHANGE_OF_DIRECTION, OTHER_NON_TARGET, UNCERTAIN.
- Unreviewed items remain PENDING / empty by default.
- Backs up the manifest before modifying.
- Does not modify any source videos or candidate CSVs.
"""

import argparse
import csv
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

VALID_HUMAN_CLASSES = {
    "LUNGE",
    "SMASH_LANDING",
    "CHANGE_OF_DIRECTION",
    "OTHER_NON_TARGET",
    "UNCERTAIN",
}

VALID_HUMAN_DECISIONS = {
    "ACCEPT",
    "REJECT",
    "UNCERTAIN",
}


def load_manifest(manifest_path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    """Loads review manifest CSV preserving exact column headers and row order."""
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

    with open(manifest_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    return fieldnames, rows


def create_manifest_backup(manifest_path: Path) -> Path:
    """Creates a timestamped backup of the manifest prior to any modifications."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = manifest_path.parent / f"{manifest_path.stem}.backup_{timestamp}{manifest_path.suffix}"
    shutil.copy2(manifest_path, backup_path)
    return backup_path


def save_manifest(manifest_path: Path, fieldnames: List[str], rows: List[Dict[str, str]]) -> None:
    """Saves updated manifest CSV safely with atomic rename."""
    temp_path = manifest_path.parent / f".tmp_{manifest_path.name}"
    with open(temp_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temp_path.replace(manifest_path)


def get_manifest_status(rows: List[Dict[str, str]]) -> Dict[str, int]:
    """Computes review progress statistics across manifest rows."""
    stats = {
        "total": len(rows),
        "pending": 0,
        "accepted": 0,
        "rejected": 0,
        "uncertain": 0,
        "classes": {cls: 0 for cls in VALID_HUMAN_CLASSES},
    }

    for r in rows:
        dec = (r.get("human_decision") or "").strip().upper()
        cls = (r.get("human_verified_class") or "").strip().upper()

        if not dec or dec in ("PENDING", "NAN", ""):
            stats["pending"] += 1
        elif dec == "ACCEPT":
            stats["accepted"] += 1
            if cls in stats["classes"]:
                stats["classes"][cls] += 1
        elif dec == "REJECT":
            stats["rejected"] += 1
            if cls in stats["classes"]:
                stats["classes"][cls] += 1
        elif dec == "UNCERTAIN":
            stats["uncertain"] += 1
            stats["classes"]["UNCERTAIN"] += 1
        else:
            stats["pending"] += 1

    return stats


def print_status_report(stats: Dict[str, int]) -> None:
    """Prints a formatted manifest status summary."""
    print("=" * 60)
    print("🏸 Human Review Verification Status:")
    print("=" * 60)
    print(f"Total Review Items:  {stats['total']}")
    print(f"  • Confirmed Verified: {stats['accepted']}")
    print(f"  • Non-Target / Reject: {stats['rejected']}")
    print(f"  • Marked Uncertain:   {stats['uncertain']}")
    print(f"  • Unreviewed / Pending: {stats['pending']}")
    print("-" * 60)
    print("Verified Class Breakdown:")
    for cls, count in stats["classes"].items():
        print(f"  • {cls:<22}: {count}")
    print("=" * 60)


def record_single_decision(
    manifest_path: Path,
    review_id: str,
    verified_class: str,
    decision: str = "ACCEPT",
    reviewer: str = "human",
    notes: str = "",
) -> bool:
    """
    Records an explicit human verification decision for a single review item.
    """
    v_class = (verified_class or "").strip().upper()
    v_dec = (decision or "").strip().upper()

    if v_class not in VALID_HUMAN_CLASSES:
        raise ValueError(
            f"Invalid verified class '{v_class}'. Must be one of: {', '.join(sorted(VALID_HUMAN_CLASSES))}"
        )

    if v_dec not in VALID_HUMAN_DECISIONS:
        raise ValueError(
            f"Invalid human decision '{v_dec}'. Must be one of: {', '.join(sorted(VALID_HUMAN_DECISIONS))}"
        )

    fieldnames, rows = load_manifest(manifest_path)

    # Locate target row
    target_idx = None
    for idx, r in enumerate(rows):
        if r.get("review_id") == review_id:
            target_idx = idx
            break

    if target_idx is None:
        raise KeyError(f"Review ID '{review_id}' not found in manifest: {manifest_path}")

    # Ensure required columns exist in fieldnames
    for col in ["human_verified_class", "human_decision", "human_notes"]:
        if col not in fieldnames:
            fieldnames.append(col)

    # Create backup before updating
    backup = create_manifest_backup(manifest_path)

    row = rows[target_idx]
    prev_dec = row.get("human_decision", "PENDING")
    prev_class = row.get("human_verified_class", "NONE")

    row["human_verified_class"] = v_class
    row["human_decision"] = v_dec
    full_notes = f"Reviewer: {reviewer} | Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    if notes:
        full_notes += f" | Notes: {notes}"
    row["human_notes"] = full_notes

    save_manifest(manifest_path, fieldnames, rows)
    print(f"✅ Successfully updated {review_id}:")
    print(f"   Previous: decision={prev_dec}, class={prev_class}")
    print(f"   Updated:  decision={v_dec}, class={v_class}")
    print(f"   Backup saved to: {backup.name}")
    return True


def add_review_item(
    manifest_path: Path,
    item_dict: Dict[str, Any],
) -> Path:
    """
    Safely appends a new review item to the review manifest CSV.
    Enforces uniqueness of review_id and creates a timestamped backup before writing.
    """
    review_id = str(item_dict.get("review_id", "")).strip()
    if not review_id:
        raise ValueError("review_id must be provided in item_dict")

    fieldnames, rows = load_manifest(manifest_path)

    for r in rows:
        if r.get("review_id") == review_id:
            raise ValueError(f"Review item '{review_id}' already exists in manifest: {manifest_path}")

    # Ensure required columns exist in fieldnames
    for col in ["human_verified_class", "human_decision", "human_notes"]:
        if col not in fieldnames:
            fieldnames.append(col)

    backup = create_manifest_backup(manifest_path)

    new_row = {}
    for col in fieldnames:
        val = item_dict.get(col, "")
        new_row[col] = "" if val is None else str(val)

    rows.append(new_row)
    save_manifest(manifest_path, fieldnames, rows)
    print(f"✅ Successfully added new item {review_id} to manifest.")
    print(f"   Backup saved to: {backup.name}")
    return backup


def interactive_review_session(manifest_path: Path, reviewer: str = "human") -> None:
    """Interactive loop to review and confirm items one by one."""
    fieldnames, rows = load_manifest(manifest_path)

    print("\nStarting Interactive Review Session...")
    print("Press Ctrl+C at any time to exit.\n")

    updated_count = 0

    for idx, r in enumerate(rows, start=1):
        dec = (r.get("human_decision") or "").strip().upper()
        if dec and dec not in ("PENDING", "NAN", ""):
            continue  # Skip already reviewed items

        r_id = r.get("review_id", f"REV_{idx}")
        v_name = r.get("video_name", "unknown")
        start = r.get("start_time", "0.0")
        end = r.get("end_time", "0.0")
        prop_class = r.get("proposed_class", "UNKNOWN")
        mach_assess = r.get("machine_assessment", "UNKNOWN")
        mach_reason = r.get("machine_reason", "")
        contact_sheet = r.get("contact_sheet_path", "")

        print("-" * 65)
        print(f"Review Item [{idx}/{len(rows)}]: {r_id}")
        print(f"  Video:        {v_name} ({start}s - {end}s)")
        print(f"  Proposed:     {prop_class}")
        print(f"  AI Screening: {mach_assess}")
        if mach_reason:
            print(f"  AI Reason:    {mach_reason}")
        if contact_sheet:
            print(f"  Contact Sheet: data/human_verification_review_package/{contact_sheet}")
        print("-" * 65)

        print("Select Human Verification Class:")
        print("  [1] LUNGE")
        print("  [2] SMASH_LANDING")
        print("  [3] CHANGE_OF_DIRECTION")
        print("  [4] OTHER_NON_TARGET (Standing clear, walking, pickup, etc.)")
        print("  [5] UNCERTAIN")
        print("  [S] Skip (Leave pending)")
        print("  [Q] Quit session")

        choice = input("\nEnter choice [1-5, S, Q]: ").strip().upper()

        if choice == "Q":
            print("\nExiting interactive review session.")
            break
        elif choice == "S" or not choice:
            print(f"Skipped {r_id} (remains pending).")
            continue

        choice_map = {
            "1": ("LUNGE", "ACCEPT"),
            "2": ("SMASH_LANDING", "ACCEPT"),
            "3": ("CHANGE_OF_DIRECTION", "ACCEPT"),
            "4": ("OTHER_NON_TARGET", "REJECT"),
            "5": ("UNCERTAIN", "UNCERTAIN"),
        }

        if choice in choice_map:
            v_class, v_dec = choice_map[choice]
            notes = input("Optional observation notes (Enter to skip): ").strip()
            record_single_decision(
                manifest_path=manifest_path,
                review_id=r_id,
                verified_class=v_class,
                decision=v_dec,
                reviewer=reviewer,
                notes=notes,
            )
            updated_count += 1
        else:
            print("Invalid selection. Item skipped.")

    print(f"\nReview session finished. Total items confirmed in this session: {updated_count}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Record explicit human review decisions into review manifest.")
    parser.add_argument(
        "--manifest",
        type=str,
        default="data/human_verification_review_package/review_manifest.csv",
        help="Path to review manifest CSV",
    )
    parser.add_argument("--review-id", type=str, help="Specific Review ID to update (e.g. REV_001)")
    parser.add_argument(
        "--verified-class",
        type=str,
        choices=list(VALID_HUMAN_CLASSES),
        help="Verified human movement class",
    )
    parser.add_argument(
        "--decision",
        type=str,
        default="ACCEPT",
        choices=list(VALID_HUMAN_DECISIONS),
        help="Decision status (ACCEPT, REJECT, UNCERTAIN)",
    )
    parser.add_argument("--reviewer", type=str, default="human", help="Reviewer identity / name")
    parser.add_argument("--notes", type=str, default="", help="Optional notes on visual observations")
    parser.add_argument("--status", action="store_true", help="Print manifest review progress and exit")
    parser.add_argument("--interactive", action="store_true", help="Launch interactive step-by-step review")

    args = parser.parse_args()
    manifest_path = Path(args.manifest).resolve()

    if not manifest_path.exists():
        print(f"Error: Manifest not found at {manifest_path}", file=sys.stderr)
        sys.exit(1)

    if args.status:
        _, rows = load_manifest(manifest_path)
        stats = get_manifest_status(rows)
        print_status_report(stats)
        sys.exit(0)

    if args.interactive:
        interactive_review_session(manifest_path, reviewer=args.reviewer)
        sys.exit(0)

    if args.review_id:
        if not args.verified_class:
            print("Error: --verified-class is required when specifying --review-id", file=sys.stderr)
            sys.exit(1)
        record_single_decision(
            manifest_path=manifest_path,
            review_id=args.review_id,
            verified_class=args.verified_class,
            decision=args.decision,
            reviewer=args.reviewer,
            notes=args.notes,
        )
        sys.exit(0)

    # If no action specified, show status
    _, rows = load_manifest(manifest_path)
    stats = get_manifest_status(rows)
    print_status_report(stats)
    print("\nTip: Use '--review-id REV_XXX --verified-class <CLASS>' to record a decision, or '--interactive' to review.")


if __name__ == "__main__":
    main()
