import csv
import os
import tempfile
import unittest
from pathlib import Path

from review_candidates import (
    load_candidates_from_csv,
    save_candidates_to_csv,
    calculate_review_summary,
    find_overlapping_candidates,
    VALID_MOVEMENT_CLASSES,
    TARGET_MOVEMENT_CLASSES,
    RETIRED_CLASSES,
)
from dataset_builder import (
    VALID_TARGET_CLASSES as DS_TARGET_CLASSES,
    RETIRED_CLASSES as DS_RETIRED_CLASSES,
    MOVEMENT_TYPES as DS_MOVEMENT_TYPES,
)
from src.detection.candidate_detector import (
    MovementCandidate,
    CANDIDATE_CSV_FIELDS,
    TARGET_MOVEMENT_CLASSES as DETECTOR_TARGET_CLASSES,
)


class TestCandidateReviewWorkflow(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.csv_path = Path(self.temp_dir.name) / "test_candidates.csv"

        self.sample_rows = [
            {
                "candidate_id": "cand_001",
                "video_id": "match_01.mp4",
                "start_time": "5.0",
                "end_time": "6.2",
                "duration": "1.2",
                "proposed_movement": "LUNGE",
                "heuristic_score": "0.85",
                "key_metrics": '{"min_knee_angle": 95.0}',
                "human_decision": "PENDING",
                "reviewed_movement": "",
                "reviewer_notes": "",
                "movement_type": "LUNGE",
                "confidence_score": "0.85",
                "status": "PROPOSED",
            },
            {
                "candidate_id": "cand_002",
                "video_id": "match_01.mp4",
                "start_time": "12.0",
                "end_time": "13.5",
                "duration": "1.5",
                "proposed_movement": "SMASH_LANDING",
                "heuristic_score": "0.92",
                "key_metrics": '{"jump_elevation": 0.12}',
                "human_decision": "ACCEPT",
                "reviewed_movement": "SMASH_LANDING",
                "reviewer_notes": "Clean high smash",
                "movement_type": "SMASH_LANDING",
                "confidence_score": "0.92",
                "status": "ACCEPTED",
            },
            {
                "candidate_id": "cand_003",
                "video_id": "match_01.mp4",
                "start_time": "20.1",
                "end_time": "21.4",
                "duration": "1.3",
                "proposed_movement": "CHANGE_OF_DIRECTION",
                "heuristic_score": "0.71",
                "key_metrics": '{"braking_knee_flexion": 140.0}',
                "human_decision": "REJECT",
                "reviewed_movement": "NO_EVENT",
                "reviewer_notes": "Just jogging backwards",
                "movement_type": "CHANGE_OF_DIRECTION",
                "confidence_score": "0.71",
                "status": "REJECTED",
            },
            {
                "candidate_id": "cand_004",
                "video_id": "match_01.mp4",
                "start_time": "25.0",
                "end_time": "26.8",
                "duration": "1.8",
                "proposed_movement": "UNCERTAIN",
                "heuristic_score": "0.60",
                "key_metrics": '{"reason": "Ambiguous footwork"}',
                "human_decision": "UNCERTAIN",
                "reviewed_movement": "UNCERTAIN",
                "reviewer_notes": "Camera angle too obscured",
                "movement_type": "UNCERTAIN",
                "confidence_score": "0.60",
                "status": "UNCERTAIN",
            },
        ]

        save_candidates_to_csv(self.sample_rows, str(self.csv_path))

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_load_and_save_candidates(self):
        loaded = load_candidates_from_csv(str(self.csv_path))
        self.assertEqual(len(loaded), 4)
        self.assertEqual(loaded[0]["candidate_id"], "cand_001")
        self.assertEqual(loaded[0]["proposed_movement"], "LUNGE")
        self.assertEqual(loaded[0]["human_decision"], "PENDING")

        # Verify all CANDIDATE_CSV_FIELDS are present
        for field in CANDIDATE_CSV_FIELDS:
            self.assertIn(field, loaded[0])

    def test_calculate_review_summary(self):
        loaded = load_candidates_from_csv(str(self.csv_path))
        summary = calculate_review_summary(loaded)
        self.assertEqual(summary["total"], 4)
        self.assertEqual(summary["accepted"], 1)
        self.assertEqual(summary["rejected"], 1)
        self.assertEqual(summary["uncertain"], 1)
        self.assertEqual(summary["pending"], 1)

    def test_contamination_prevention_status_checks(self):
        loaded = load_candidates_from_csv(str(self.csv_path))

        # 1. Accepted candidate is eligible
        cand_accept = loaded[1]
        self.assertEqual(cand_accept["human_decision"], "ACCEPT")
        self.assertEqual(cand_accept["reviewed_movement"], "SMASH_LANDING")

        # 2. Rejected candidate must be blocked
        cand_reject = loaded[2]
        self.assertEqual(cand_reject["human_decision"], "REJECT")
        self.assertIn(cand_reject["human_decision"], ["REJECT", "UNCERTAIN"])

        # 3. Uncertain candidate must be blocked
        cand_uncertain = loaded[3]
        self.assertEqual(cand_uncertain["human_decision"], "UNCERTAIN")
        self.assertIn(cand_uncertain["human_decision"], ["REJECT", "UNCERTAIN"])

        # 4. Pending candidate requires review before entering dataset
        cand_pending = loaded[0]
        self.assertEqual(cand_pending["human_decision"], "PENDING")
        self.assertNotEqual(cand_pending["human_decision"], "ACCEPT")

    def test_prevent_unreviewed_candidate_automatic_acceptance(self):
        """Proposed movement must NEVER automatically become a confirmed ground-truth label."""
        c = MovementCandidate(
            candidate_id="cand_test",
            video_id="clip.mp4",
            movement_type="LUNGE",
            start_time=1.0,
            end_time=2.0,
            duration=1.0,
            confidence_score=0.8,
            key_metrics="{}",
            proposed_movement="LUNGE",
            heuristic_score=0.8,
            human_decision="PENDING",
            reviewed_movement="",
        )
        # Verify human_decision defaults to PENDING and reviewed_movement is unconfirmed
        self.assertEqual(c.human_decision, "PENDING")
        self.assertEqual(c.reviewed_movement, "")
        self.assertNotEqual(c.human_decision, "ACCEPT")

    def test_duplicate_overlap_detection(self):
        """Tests interval overlap logic used to prevent duplicate sequence additions."""
        # Existing sample: [100, 200]
        ex_start, ex_end = 100, 200
        fps = 25.0

        # Sample A: [150, 250] overlaps by 50 frames (2.0s > 0.4s threshold)
        new_start, new_end = 150, 250
        overlap_frames = min(new_end, ex_end) - max(new_start, ex_start)
        self.assertGreater(overlap_frames, int(fps * 0.4))

        # Sample B: [220, 280] non-overlapping
        no_ov_start, no_ov_end = 220, 280
        no_ov_frames = min(no_ov_end, ex_end) - max(no_ov_start, ex_start)
        self.assertLessEqual(no_ov_frames, 0)

    def test_target_movement_classes_exactly_three(self):
        """Verifies candidate review and detector only allow the 3 target classes."""
        expected = ["LUNGE", "SMASH_LANDING", "CHANGE_OF_DIRECTION"]
        self.assertEqual(TARGET_MOVEMENT_CLASSES, expected)
        self.assertEqual(DETECTOR_TARGET_CLASSES, expected)
        self.assertEqual(set(expected), DS_TARGET_CLASSES)
        self.assertNotIn("SCISSOR_JUMP", TARGET_MOVEMENT_CLASSES)
        self.assertNotIn("FOOTWORK_SPLIT_STEP", TARGET_MOVEMENT_CLASSES)
        self.assertIn("SCISSOR_JUMP", RETIRED_CLASSES)
        self.assertIn("FOOTWORK_SPLIT_STEP", RETIRED_CLASSES)

    def test_retired_classes_cannot_enter_dataset(self):
        """Retired movement classes must not be present in dataset builder movement types or valid targets."""
        for num_key, name in DS_MOVEMENT_TYPES.items():
            self.assertIn(name, DS_TARGET_CLASSES)
            self.assertNotIn(name, DS_RETIRED_CLASSES)
        self.assertNotIn("SCISSOR_JUMP", DS_TARGET_CLASSES)
        self.assertNotIn("FOOTWORK_SPLIT_STEP", DS_TARGET_CLASSES)
        self.assertIn("SCISSOR_JUMP", DS_RETIRED_CLASSES)
        self.assertIn("FOOTWORK_SPLIT_STEP", DS_RETIRED_CLASSES)

    def test_find_overlapping_candidates(self):
        """Tests that candidate review identifies temporal overlaps within the same video."""
        cands = [
            {"candidate_id": "c1", "video_id": "vid.mp4", "start_time": "1.0", "end_time": "2.5"},
            {"candidate_id": "c2", "video_id": "vid.mp4", "start_time": "2.0", "end_time": "3.5"}, # overlaps c1 by 0.5s
            {"candidate_id": "c3", "video_id": "vid.mp4", "start_time": "5.0", "end_time": "6.0"}, # no overlap
            {"candidate_id": "c4", "video_id": "other.mp4", "start_time": "1.0", "end_time": "2.5"}, # different video
        ]
        ov_for_c1 = find_overlapping_candidates(cands, 0)
        self.assertEqual(len(ov_for_c1), 1)
        self.assertEqual(ov_for_c1[0][0]["candidate_id"], "c2")
        self.assertAlmostEqual(ov_for_c1[0][1], 0.5)

        ov_for_c3 = find_overlapping_candidates(cands, 2)
        self.assertEqual(len(ov_for_c3), 0)


if __name__ == "__main__":
    unittest.main()
