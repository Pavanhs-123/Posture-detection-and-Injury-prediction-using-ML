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

    def test_create_candidate_method(self):
        """Verifies _create_candidate helper constructs compliant MovementCandidate."""
        cand = self.detector._create_candidate(
            video_id="clip_01.mp4",
            category="UNCERTAIN",
            start_time=0.5,
            end_time=2.5,
            score=0.4,
            metrics={"trigger": "fallback"},
        )
        self.assertIsInstance(cand, MovementCandidate)
        self.assertEqual(cand.candidate_id, "cand_001")
        self.assertEqual(cand.video_id, "clip_01.mp4")
        self.assertEqual(cand.movement_type, "UNCERTAIN")
        self.assertEqual(cand.proposed_movement, "UNCERTAIN")
        self.assertEqual(cand.start_time, 0.5)
        self.assertEqual(cand.end_time, 2.5)
        self.assertEqual(cand.duration, 2.0)
        self.assertEqual(cand.heuristic_score, 0.4)
        self.assertEqual(cand.confidence_score, 0.4)
        self.assertEqual(cand.status, "REVIEW_REQUIRED")
        self.assertEqual(cand.human_decision, "PENDING")

    def test_two_consecutive_jump_smashes_separated_by_standing_recovery(self):
        """Verifies two consecutive jump smashes are detected as separate candidates."""
        # 70 frames @ 10 fps = 7.0s
        signals = self._create_base_signals(num_frames=70, fps=10.0)

        # Jump 1: Apex at frames 10-13 (1.0s - 1.3s), elevates from 0.50 to 0.42
        for i in range(10, 14):
            signals[i]["hip_y"] = 0.42
            signals[i]["wrist_r_y"] = 0.20  # Overhead arm
        # Jump 1 Landing: Touchdown at frame 14-17 (1.4s - 1.7s), stiff landing knee 157.6 deg
        for i in range(14, 18):
            signals[i]["hip_y"] = 0.51
            signals[i]["min_knee"] = 157.6

        # Recovery / upright standing at frames 18-29 (1.8s - 2.9s)
        for i in range(18, 30):
            signals[i]["hip_y"] = 0.50
            signals[i]["min_knee"] = 170.0

        # Jump 2: Pre-crouch at frame 31, apex at frames 34-38 (3.4s - 3.8s), elevates to 0.40
        signals[31]["hip_y"] = 0.54
        signals[31]["min_knee"] = 138.0
        for i in range(34, 39):
            signals[i]["hip_y"] = 0.40
            signals[i]["wrist_r_y"] = 0.18  # Overhead arm
        # Jump 2 Landing: Touchdown at frame 39-43 (3.9s - 4.3s), landing knee 146.7 deg
        for i in range(39, 44):
            signals[i]["hip_y"] = 0.51
            signals[i]["min_knee"] = 146.7

        candidates = self.detector.detect_candidates_from_signals(signals, "test_two_smashes.mp4")
        smashes = [c for c in candidates if c.proposed_movement == "SMASH_LANDING"]
        self.assertEqual(len(smashes), 2, f"Expected 2 smash landings, found {len(smashes)}")
        self.assertLess(smashes[0].end_time, smashes[1].start_time, "Jump 1 and Jump 2 candidates must not overlap")
        self.assertAlmostEqual(smashes[0].start_time, 0.5, delta=0.5)
        self.assertAlmostEqual(smashes[1].start_time, 2.9, delta=0.5)

    def test_stiff_landing_detection_around_157_6_deg(self):
        """Verifies stiff-legged landings (up to 160 deg) are captured rather than discarded."""
        signals = self._create_base_signals(num_frames=35, fps=10.0)
        # Apex at frames 10-14
        for i in range(10, 15):
            signals[i]["hip_y"] = 0.42
            signals[i]["wrist_r_y"] = 0.20
        # Stiff landing cushioning at 157.6 deg
        for i in range(16, 20):
            signals[i]["hip_y"] = 0.51
            signals[i]["min_knee"] = 157.6

        candidates = self.detector.detect_candidates_from_signals(signals, "test_stiff_smash.mp4")
        smashes = [c for c in candidates if c.proposed_movement == "SMASH_LANDING"]
        self.assertEqual(len(smashes), 1)
        self.assertAlmostEqual(smashes[0].start_time, 0.5, delta=0.4)
        self.assertAlmostEqual(smashes[0].end_time, 2.4, delta=0.4)

    def test_moderate_stiff_landing_around_146_7_deg(self):
        """Verifies landing around 146.7 deg is detected."""
        signals = self._create_base_signals(num_frames=35, fps=10.0)
        for i in range(10, 15):
            signals[i]["hip_y"] = 0.42
            signals[i]["wrist_r_y"] = 0.20
        for i in range(16, 20):
            signals[i]["hip_y"] = 0.51
            signals[i]["min_knee"] = 146.7

        candidates = self.detector.detect_candidates_from_signals(signals, "test_moderate_smash.mp4")
        smashes = [c for c in candidates if c.proposed_movement == "SMASH_LANDING"]
        self.assertEqual(len(smashes), 1)

    def test_jump_without_valid_landing_fails_detection(self):
        """A jump with no knee cushioning below 160 deg is rejected."""
        signals = self._create_base_signals(num_frames=35, fps=10.0)
        for i in range(10, 15):
            signals[i]["hip_y"] = 0.42
            signals[i]["wrist_r_y"] = 0.20
        # Remains completely locked straight (>170 deg) at touchdown
        for i in range(16, 25):
            signals[i]["hip_y"] = 0.51
            signals[i]["min_knee"] = 175.0

        candidates = self.detector.detect_candidates_from_signals(signals, "test_no_landing.mp4")
        smashes = [c for c in candidates if c.proposed_movement == "SMASH_LANDING"]
        self.assertEqual(len(smashes), 0, "Jump without landing cushioning must not be accepted as SMASH_LANDING")

    def test_upright_reset_prevents_bleeding_into_subsequent_action(self):
        """Verifies upright standing after touchdown terminates search and prevents absorbing future movements."""
        signals = self._create_base_signals(num_frames=50, fps=10.0)
        # Jump apex at 10-13
        for i in range(10, 14):
            signals[i]["hip_y"] = 0.42
            signals[i]["wrist_r_y"] = 0.20
        # Touchdown with no knee cushioning at 15-17 (knees 170 deg)
        for i in range(15, 18):
            signals[i]["hip_y"] = 0.51
            signals[i]["min_knee"] = 170.0
        # Returns to standing at 18-22
        for i in range(18, 23):
            signals[i]["hip_y"] = 0.50
            signals[i]["min_knee"] = 172.0
        # Unrelated deep knee bend at frame 30 (3.0s)
        for i in range(30, 35):
            signals[i]["hip_y"] = 0.58
            signals[i]["min_knee"] = 120.0

        # With upright reset, the jump search must terminate when standing is resumed
        # and NOT link frame 30's knee bend to frame 12's jump.
        raw_smashes = self.detector._detect_smash_landings(signals, "test_reset.mp4")
        self.assertEqual(len(raw_smashes), 0, "Upright reset should prevent linking late knee bend to earlier jump")


if __name__ == "__main__":
    unittest.main()
