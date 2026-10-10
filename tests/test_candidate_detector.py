import os
import unittest
import numpy as np
import cv2

from src.detection.candidate_detector import (
    MovementCandidateDetector,
    DetectorConfig,
    MovementCandidate,
)


class TestCandidateDetector(unittest.TestCase):
    def setUp(self):
        self.config = DetectorConfig(sample_fps=10.0)
        self.detector = MovementCandidateDetector(self.config)

    def _create_base_signals(self, num_frames=30, fps=10.0):
        signals = []
        for i in range(num_frames):
            t = i / fps
            signals.append({
                "frame_idx": i,
                "timestamp": t,
                "has_pose": True,
                "hip_x": 0.5,
                "hip_y": 0.5,
                "hip_z": 0.0,
                "ankle_l_x": 0.45,
                "ankle_l_y": 0.9,
                "ankle_r_x": 0.55,
                "ankle_r_y": 0.9,
                "ankle_dist": 0.1,
                "wrist_l_y": 0.6,
                "wrist_r_y": 0.6,
                "shoulder_y": 0.3,
                "knee_l": 165.0,
                "knee_r": 165.0,
                "min_knee": 165.0,
                "elbow_l": 160.0,
                "elbow_r": 160.0,
                "max_elbow": 160.0,
                "torso_lean": 5.0,
                "confidence": 0.9,
                "hip_vx": 0.0,
                "hip_vy": 0.0,
            })
        return signals

    def test_lunge_detection(self):
        signals = self._create_base_signals(num_frames=30)
        # Introduce a deep lunge between frame 8 (0.8s) and frame 20 (2.0s)
        for i in range(8, 20):
            # Hip drops from 0.5 to 0.62 (larger y = lower height)
            signals[i]["hip_y"] = 0.62
            # Knee flexes to 95 degrees
            signals[i]["min_knee"] = 95.0
            # Ankle distance expands
            signals[i]["ankle_dist"] = 0.35

        candidates = self.detector.detect_candidates_from_signals(signals, "test_video.mp4")
        lunges = [c for c in candidates if c.movement_type == "LUNGE"]
        self.assertGreaterEqual(len(lunges), 1)
        lunge = lunges[0]
        self.assertAlmostEqual(lunge.start_time, 0.5, delta=0.4)
        self.assertAlmostEqual(lunge.end_time, 2.2, delta=0.4)
        self.assertGreater(lunge.confidence_score, 0.5)

    def test_smash_landing_detection(self):
        signals = self._create_base_signals(num_frames=35)
        # Apex at frames 10-14: Hip elevates (hip_y drops to 0.42 from 0.50)
        for i in range(10, 15):
            signals[i]["hip_y"] = 0.42
            signals[i]["wrist_r_y"] = 0.20  # Overhead wrist above shoulder (0.30)
            signals[i]["max_elbow"] = 165.0

        # Landing impact at frames 16-19: Hip contacts ground, knee flexes to 110 deg
        for i in range(16, 20):
            signals[i]["hip_y"] = 0.51
            signals[i]["min_knee"] = 110.0

        candidates = self.detector.detect_candidates_from_signals(signals, "test_smash.mp4")
        smashes = [c for c in candidates if c.movement_type == "SMASH_LANDING"]
        self.assertGreaterEqual(len(smashes), 1)
        smash = smashes[0]
        self.assertGreater(smash.confidence_score, 0.5)

    def test_scissor_movement_not_detected_as_target_class(self):
        """Aerial scissor leg swap must NOT generate SCISSOR_JUMP and must NOT silently map to LUNGE or SMASH_LANDING."""
        signals = self._create_base_signals(num_frames=35)
        # Pre-jump (frames 0-9): Left ankle in front of right ankle
        for i in range(0, 10):
            signals[i]["ankle_l_x"] = 0.60
            signals[i]["ankle_r_x"] = 0.40  # diff = +0.20

        # Jump phase (frames 10-16): Hip elevates
        for i in range(10, 17):
            signals[i]["hip_y"] = 0.43

        # Post-jump / landing (frames 17-28): Right ankle in front of left ankle
        for i in range(17, 28):
            signals[i]["ankle_l_x"] = 0.40
            signals[i]["ankle_r_x"] = 0.60  # diff = -0.20 (swapped!)

        candidates = self.detector.detect_candidates_from_signals(signals, "test_scissor.mp4")
        # Ensure SCISSOR_JUMP is never generated
        scissors = [c for c in candidates if c.movement_type == "SCISSOR_JUMP" or c.proposed_movement == "SCISSOR_JUMP"]
        self.assertEqual(len(scissors), 0, "SCISSOR_JUMP must not be generated as a target class")
        # Ensure not silently mapped to LUNGE or SMASH_LANDING
        lunges = [c for c in candidates if c.proposed_movement == "LUNGE"]
        smashes = [c for c in candidates if c.proposed_movement == "SMASH_LANDING"]
        self.assertEqual(len(lunges), 0, "Scissor movement must not silently map to LUNGE")
        self.assertEqual(len(smashes), 0, "Scissor movement without smash mechanics must not map to SMASH_LANDING")

    def test_change_of_direction_detection(self):
        signals = self._create_base_signals(num_frames=30)
        # Velocity moving right (+0.25) from frames 5-10
        for i in range(5, 11):
            signals[i]["hip_vx"] = 0.25

        # Reversal crossing zero at frame 12
        signals[12]["hip_vx"] = 0.0
        signals[12]["min_knee"] = 125.0  # Braking plant

        # Velocity moving left (-0.25) from frames 14-19
        for i in range(14, 20):
            signals[i]["hip_vx"] = -0.25

        candidates = self.detector.detect_candidates_from_signals(signals, "test_cod.mp4")
        cods = [c for c in candidates if c.movement_type == "CHANGE_OF_DIRECTION"]
        self.assertGreaterEqual(len(cods), 1)
        self.assertGreater(cods[0].confidence_score, 0.4)

    def test_split_step_movement_not_detected_as_target_class(self):
        """Footwork split step must NOT generate FOOTWORK_SPLIT_STEP and must NOT silently map to LUNGE or SMASH_LANDING."""
        signals = self._create_base_signals(num_frames=30)
        # Quick, small hop at frames 8-12 (elevation = 0.50 - 0.47 = 0.03)
        for i in range(8, 12):
            signals[i]["hip_y"] = 0.47
            signals[i]["ankle_dist"] = 0.12

        # Touchdown at frame 13 with stance expansion and athletic ready bend
        for i in range(13, 17):
            signals[i]["hip_y"] = 0.50
            signals[i]["ankle_dist"] = 0.18  # Widening
            signals[i]["min_knee"] = 135.0   # Ready bend (not a deep lunge)

        candidates = self.detector.detect_candidates_from_signals(signals, "test_split.mp4")
        # Ensure FOOTWORK_SPLIT_STEP is never generated
        splits = [c for c in candidates if c.movement_type == "FOOTWORK_SPLIT_STEP" or c.proposed_movement == "FOOTWORK_SPLIT_STEP"]
        self.assertEqual(len(splits), 0, "FOOTWORK_SPLIT_STEP must not be generated as a target class")
        # Ensure not silently mapped to LUNGE or SMASH_LANDING
        lunges = [c for c in candidates if c.proposed_movement == "LUNGE"]
        self.assertEqual(len(lunges), 0, "Split step hop must not silently map to LUNGE")

    def test_nms_merging(self):
        # Two overlapping candidates of same type
        c1 = MovementCandidate(
            candidate_id="1",
            video_id="v",
            movement_type="LUNGE",
            start_time=1.0,
            end_time=2.0,
            duration=1.0,
            confidence_score=0.7,
            key_metrics="{}",
        )
        c2 = MovementCandidate(
            candidate_id="2",
            video_id="v",
            movement_type="LUNGE",
            start_time=1.4,
            end_time=2.4,
            duration=1.0,
            confidence_score=0.85,
            key_metrics="{}",
        )
        merged = self.detector._merge_candidates([c1, c2])
        self.assertEqual(len(merged), 1)
        self.assertAlmostEqual(merged[0].start_time, 1.0)
        self.assertAlmostEqual(merged[0].end_time, 2.4)
        self.assertEqual(merged[0].confidence_score, 0.85)

    def test_pipeline_on_dummy_video(self):
        temp_dir = os.path.dirname(os.path.abspath(__file__))
        dummy_video = os.path.join(temp_dir, "temp_cand_test.mp4")
        output_csv = os.path.join(temp_dir, "temp_cand_out.csv")

        width, height = 320, 240
        fps = 10
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(dummy_video, fourcc, fps, (width, height))
        for _ in range(10):
            writer.write(np.zeros((height, width, 3), dtype=np.uint8))
        writer.release()

        try:
            candidates = self.detector.detect_video_candidates(dummy_video, output_csv=output_csv)
            self.assertEqual(len(candidates), 0)
            self.assertTrue(os.path.exists(output_csv))
        finally:
            if os.path.exists(dummy_video):
                os.remove(dummy_video)
            if os.path.exists(output_csv):
                os.remove(output_csv)

    def test_bilateral_squat_not_confident_lunge(self):
        """Bilateral squat with symmetric knees and narrow stance must NOT be declared a confident LUNGE."""
        signals = self._create_base_signals(num_frames=30)
        for i in range(8, 20):
            signals[i]["hip_y"] = 0.62
            signals[i]["min_knee"] = 90.0
            signals[i]["knee_l"] = 90.0
            signals[i]["knee_r"] = 90.0  # Symmetric bilateral squat!
            signals[i]["ankle_dist"] = 0.10  # Narrow stance

        candidates = self.detector.detect_candidates_from_signals(signals, "test_squat.mp4")
        confident_lunges = [c for c in candidates if c.proposed_movement == "LUNGE"]
        self.assertEqual(len(confident_lunges), 0, "Bilateral squat should not trigger confident LUNGE")

    def test_jump_without_overhead_arm_is_uncertain(self):
        """Jump landing without overhead arm must NOT be classified as SMASH_LANDING; it must be UNCERTAIN."""
        signals = self._create_base_signals(num_frames=35)
        # Apex at frames 10-14: Hip elevates
        for i in range(10, 15):
            signals[i]["hip_y"] = 0.42
            signals[i]["wrist_r_y"] = 0.60  # Arm is DOWN (below shoulder at 0.30)
            signals[i]["wrist_l_y"] = 0.60

        # Landing impact at frames 16-19: Hip contacts ground, knee flexes
        for i in range(16, 20):
            signals[i]["hip_y"] = 0.51
            signals[i]["min_knee"] = 110.0

        candidates = self.detector.detect_candidates_from_signals(signals, "test_jump_arms_down.mp4")
        smash_landings = [c for c in candidates if c.proposed_movement == "SMASH_LANDING"]
        self.assertEqual(len(smash_landings), 0, "Jump without overhead arm must not be classified as SMASH_LANDING")

        uncertains = [c for c in candidates if c.proposed_movement == "UNCERTAIN"]
        self.assertGreaterEqual(len(uncertains), 1, "Jump without arm should be flagged as UNCERTAIN")

    def test_grounded_leg_switch_not_scissor_jump(self):
        """Swapping lead/trail ankles while standing on floor must NOT trigger SCISSOR_JUMP."""
        signals = self._create_base_signals(num_frames=35)
        # Left foot in front, right foot in back
        for i in range(0, 12):
            signals[i]["ankle_l_x"] = 0.65
            signals[i]["ankle_r_x"] = 0.35
            signals[i]["hip_y"] = 0.50  # Grounded!

        # Swap feet on the floor (e.g. stepping or walking)
        for i in range(12, 28):
            signals[i]["ankle_l_x"] = 0.35
            signals[i]["ankle_r_x"] = 0.65
            signals[i]["hip_y"] = 0.50  # Still grounded!

        candidates = self.detector.detect_candidates_from_signals(signals, "test_ground_step.mp4")
        scissors = [c for c in candidates if c.proposed_movement == "SCISSOR_JUMP"]
        self.assertEqual(len(scissors), 0, "Grounded foot stepping must not trigger SCISSOR_JUMP")

    def test_split_step_locked_knees_rejection(self):
        """Hop where knees stay locked straight (> 160 deg) must NOT trigger FOOTWORK_SPLIT_STEP."""
        signals = self._create_base_signals(num_frames=30)
        for i in range(8, 12):
            signals[i]["hip_y"] = 0.47
            signals[i]["ankle_dist"] = 0.12

        for i in range(13, 17):
            signals[i]["hip_y"] = 0.50
            signals[i]["ankle_dist"] = 0.18
            signals[i]["min_knee"] = 175.0  # Stiff straight legs, not an athletic ready bend!

        candidates = self.detector.detect_candidates_from_signals(signals, "test_locked_knees.mp4")
        splits = [c for c in candidates if c.proposed_movement == "FOOTWORK_SPLIT_STEP"]
        self.assertEqual(len(splits), 0, "Locked knees must not trigger FOOTWORK_SPLIT_STEP")

    def test_conflicting_detections_resolve_to_uncertain(self):
        """Overlapping detections of different types with close scores should resolve to UNCERTAIN."""
        c1 = MovementCandidate(
            candidate_id="1",
            video_id="v",
            movement_type="LUNGE",
            start_time=1.0,
            end_time=2.2,
            duration=1.2,
            confidence_score=0.72,
            key_metrics="{}",
            proposed_movement="LUNGE",
            heuristic_score=0.72,
        )
        c2 = MovementCandidate(
            candidate_id="2",
            video_id="v",
            movement_type="CHANGE_OF_DIRECTION",
            start_time=1.1,
            end_time=2.3,
            duration=1.2,
            confidence_score=0.74,
            key_metrics="{}",
            proposed_movement="CHANGE_OF_DIRECTION",
            heuristic_score=0.74,
        )
        merged = self.detector._merge_candidates([c1, c2])
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].proposed_movement, "UNCERTAIN")
        self.assertEqual(merged[0].status, "REVIEW_REQUIRED")

    def test_removed_classes_never_generated(self):
        """Verifies candidate detector never proposes retired classes under any conditions."""
        signals = self._create_base_signals(num_frames=40)
        # Combine elevated jump with swap and widen
        for i in range(10, 18):
            signals[i]["hip_y"] = 0.40
            signals[i]["ankle_l_x"] = 0.35
            signals[i]["ankle_r_x"] = 0.65
            signals[i]["ankle_dist"] = 0.30

        candidates = self.detector.detect_candidates_from_signals(signals, "test_mixed.mp4")
        allowed = {"LUNGE", "SMASH_LANDING", "CHANGE_OF_DIRECTION", "UNCERTAIN", "NO_EVENT"}
        for c in candidates:
            self.assertIn(c.proposed_movement, allowed)
            self.assertNotIn(c.proposed_movement, ["SCISSOR_JUMP", "FOOTWORK_SPLIT_STEP"])
            self.assertNotIn(c.movement_type, ["SCISSOR_JUMP", "FOOTWORK_SPLIT_STEP"])


if __name__ == "__main__":
    unittest.main()
