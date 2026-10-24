"""
Unit tests for safe human-labelled dataset preparation pipeline.
Validates:
1. Pending decisions are excluded.
2. Provisional AI assessments are not treated as verified labels.
3. Approved human decisions are included.
4. Invalid timestamps and missing source videos are safely handled.
5. Missing pose measurements are not defaulted to normal posture/zero/LOW risk.
6. Duplicate and overlapping intervals (>0.4s) are rejected.
7. Output preservation and overwrite protection.
"""

import csv
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

import cv2
import numpy as np

from scripts.record_human_review import (
    load_manifest,
    save_manifest,
    record_single_decision,
    get_manifest_status,
    VALID_HUMAN_CLASSES,
)
from scripts.extract_reviewed_dataset import (
    filter_verified_records,
    check_interval_overlap,
    extract_features_for_interval,
    run_dataset_extraction,
    parse_args,
    DATASET_FIELDNAMES,
)


class TestDatasetExtraction(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

        # Create sample manifest
        self.manifest_path = self.temp_path / "test_manifest.csv"
        self.fieldnames = [
            "review_id",
            "review_category",
            "video_name",
            "source_item_id",
            "proposed_class",
            "start_time",
            "end_time",
            "duration",
            "machine_assessment",
            "machine_reason",
            "surrounding_context_notes",
            "source_media_path",
            "contact_sheet_path",
            "human_verified_class",
            "human_decision",
            "human_notes",
        ]

        self.sample_rows = [
            {
                "review_id": "REV_001",
                "review_category": "COD_POSSIBLE_WRONG_CLASS",
                "video_name": "test_video1.mp4",
                "source_item_id": "cand_001",
                "proposed_class": "CHANGE_OF_DIRECTION",
                "start_time": "10.0",
                "end_time": "12.0",
                "duration": "2.0",
                "machine_assessment": "POSSIBLE_WRONG_CLASS",
                "machine_reason": "Jump elevation detected",
                "surrounding_context_notes": "",
                "source_media_path": "",
                "contact_sheet_path": "",
                "human_verified_class": "",
                "human_decision": "",
                "human_notes": "",
            },
            {
                "review_id": "REV_002",
                "review_category": "COD_LIKELY_FALSE_POSITIVE",
                "video_name": "test_video1.mp4",
                "source_item_id": "cand_002",
                "proposed_class": "CHANGE_OF_DIRECTION",
                "start_time": "25.0",
                "end_time": "27.0",
                "duration": "2.0",
                "machine_assessment": "LIKELY_FALSE_POSITIVE",
                "machine_reason": "No lateral velocity reversal",
                "surrounding_context_notes": "",
                "source_media_path": "",
                "contact_sheet_path": "",
                "human_verified_class": "LUNGE",
                "human_decision": "ACCEPT",
                "human_notes": "Confirmed forward lunge",
            },
            {
                "review_id": "REV_003",
                "review_category": "MISSED_LUNGE_SUGGESTION",
                "video_name": "test_video2.mp4",
                "source_item_id": "sugg_lunge_01",
                "proposed_class": "LUNGE",
                "start_time": "30.0",
                "end_time": "32.0",
                "duration": "2.0",
                "machine_assessment": "LIKELY_VALID",
                "machine_reason": "Knee flexion <= 122",
                "surrounding_context_notes": "",
                "source_media_path": "",
                "contact_sheet_path": "",
                "human_verified_class": "UNCERTAIN",
                "human_decision": "UNCERTAIN",
                "human_notes": "Occluded by post",
            },
            {
                "review_id": "REV_004",
                "review_category": "COD_POSSIBLE_WRONG_CLASS",
                "video_name": "test_video2.mp4",
                "source_item_id": "cand_003",
                "proposed_class": "CHANGE_OF_DIRECTION",
                "start_time": "50.0",
                "end_time": "52.0",
                "duration": "2.0",
                "machine_assessment": "POSSIBLE_WRONG_CLASS",
                "machine_reason": "Standing clear",
                "surrounding_context_notes": "",
                "source_media_path": "",
                "contact_sheet_path": "",
                "human_verified_class": "OTHER_NON_TARGET",
                "human_decision": "REJECT",
                "human_notes": "Standing overhead clear",
            },
        ]

        save_manifest(self.manifest_path, self.fieldnames, self.sample_rows)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_pending_decisions_excluded(self):
        """Unreviewed / pending manifest rows must never enter training data."""
        fieldnames, rows = load_manifest(self.manifest_path)
        verified, stats = filter_verified_records(rows)

        # REV_001 is pending (empty decision)
        verified_ids = [r["review_id"] for r in verified]
        self.assertNotIn("REV_001", verified_ids)
        self.assertEqual(stats["pending"], 1)

    def test_provisional_ai_assessments_not_treated_as_verified(self):
        """
        Even if machine_assessment indicates LIKELY_VALID or POSSIBLE_WRONG_CLASS,
        it must not be treated as a verified label without human confirmation.
        """
        # Create row where AI suggests SMASH_LANDING but human has not reviewed
        unconfirmed_row = {
            "review_id": "REV_999",
            "proposed_class": "CHANGE_OF_DIRECTION",
            "machine_assessment": "POSSIBLE_WRONG_CLASS",
            "machine_reason": "Detected jump elevation and overhead arm; possible SMASH_LANDING",
            "human_verified_class": "",
            "human_decision": "PENDING",
        }
        verified, stats = filter_verified_records([unconfirmed_row])
        self.assertEqual(len(verified), 0)
        self.assertEqual(stats["pending"], 1)

    def test_approved_decisions_included(self):
        """Explicitly approved human decisions must be included."""
        fieldnames, rows = load_manifest(self.manifest_path)
        verified, stats = filter_verified_records(rows)

        verified_ids = [r["review_id"] for r in verified]
        # REV_002 is ACCEPT with LUNGE
        self.assertIn("REV_002", verified_ids)
        # REV_004 is REJECT with OTHER_NON_TARGET (negative class)
        self.assertIn("REV_004", verified_ids)
        # REV_003 is UNCERTAIN (must be excluded)
        self.assertNotIn("REV_003", verified_ids)

        self.assertEqual(len(verified), 2)
        self.assertEqual(stats["accepted"], 2)
        self.assertEqual(stats["uncertain"], 1)

    def test_record_single_decision_updates_safely(self):
        """record_single_decision records label, updates status, and backs up manifest."""
        status_before = get_manifest_status(self.sample_rows)
        self.assertEqual(status_before["pending"], 1)

        # Confirm REV_001 as SMASH_LANDING
        record_single_decision(
            manifest_path=self.manifest_path,
            review_id="REV_001",
            verified_class="SMASH_LANDING",
            decision="ACCEPT",
            reviewer="test_reviewer",
            notes="Jump smash confirmed",
        )

        _, updated_rows = load_manifest(self.manifest_path)
        status_after = get_manifest_status(updated_rows)

        self.assertEqual(status_after["pending"], 0)
        self.assertEqual(status_after["classes"]["SMASH_LANDING"], 1)

        # Verify backup was created
        backups = list(self.temp_path.glob("test_manifest.backup_*.csv"))
        self.assertGreaterEqual(len(backups), 1)

    def test_record_single_decision_invalid_class_raises(self):
        """Invalid class raises ValueError."""
        with self.assertRaises(ValueError):
            record_single_decision(
                manifest_path=self.manifest_path,
                review_id="REV_001",
                verified_class="INVALID_DANCE_MOVE",
            )

    def test_duplicate_and_overlapping_intervals(self):
        """Intervals overlapping by >0.4s on the same video are detected as duplicates."""
        existing = [(10.0, 12.0), (30.0, 32.5)]

        # Heavily overlapping: 10.5 to 12.5 (1.5s overlap)
        self.assertTrue(check_interval_overlap(existing, 10.5, 12.5, max_overlap_sec=0.4))

        # Barely overlapping: 11.9 to 13.0 (0.1s overlap <= 0.4s)
        self.assertFalse(check_interval_overlap(existing, 11.9, 13.0, max_overlap_sec=0.4))

        # Completely disjoint: 15.0 to 18.0
        self.assertFalse(check_interval_overlap(existing, 15.0, 18.0, max_overlap_sec=0.4))

    def test_invalid_timestamps_and_missing_source_videos(self):
        """Missing videos or inverted timestamps are handled without crashing."""
        missing_vid = self.temp_path / "non_existent_video.mp4"

        # Missing video
        feats, qual, err = extract_features_for_interval(missing_vid, 0.0, 2.0)
        self.assertIsNone(feats)
        self.assertIn("not found", err.lower())

        # Create a valid 2-second dummy video
        dummy_vid = self.temp_path / "dummy.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(dummy_vid), fourcc, 10.0, (160, 120))
        for _ in range(20):
            writer.write(np.zeros((120, 160, 3), dtype=np.uint8))
        writer.release()

        # Inverted interval (start >= end)
        feats, qual, err = extract_features_for_interval(dummy_vid, 3.0, 1.0)
        self.assertIsNone(feats)
        self.assertIn("invalid interval", err.lower())

        # Out-of-bounds interval (start >= duration)
        feats, qual, err = extract_features_for_interval(dummy_vid, 5.0, 7.0)
        self.assertIsNone(feats)
        self.assertIn("invalid interval", err.lower())

    def test_missing_pose_measurements_not_defaulted_to_normal_posture(self):
        """
        When all frames in an interval have no detected person/landmarks,
        the extraction returns None and an explicit error; it must NEVER fabricate
        180.0 knee angle, 0.0 trunk lean, or LOW risk.
        """
        # Blank black video with no people
        dummy_vid = self.temp_path / "blank.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(dummy_vid), fourcc, 10.0, (160, 120))
        for _ in range(15):
            writer.write(np.zeros((120, 160, 3), dtype=np.uint8))
        writer.release()

        feats, qual, err = extract_features_for_interval(dummy_vid, 0.0, 1.0)
        self.assertIsNone(feats)
        self.assertEqual(qual, 0.0)
        self.assertIn("no valid pose landmarks", err.lower())

    def test_output_preservation_and_overwrite_protection(self):
        """Existing output file is not overwritten without explicit permission."""
        output_file = self.temp_path / "existing_dataset.csv"
        output_file.write_text("sequence_id,video_id\ntest,test.mp4\n")

        args = parse_args([
            "--manifest", str(self.manifest_path),
            "--output-csv", str(output_file),
            "--video-dir", str(self.temp_path),
        ])

        # Without --overwrite, must abort with non-zero exit code
        ret = run_dataset_extraction(args)
        self.assertEqual(ret, 1)

        # File content must remain unchanged
        content = output_file.read_text()
        self.assertEqual(content, "sequence_id,video_id\ntest,test.mp4\n")


if __name__ == "__main__":
    unittest.main()
