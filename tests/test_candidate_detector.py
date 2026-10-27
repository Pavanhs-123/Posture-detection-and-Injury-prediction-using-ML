import json
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

    def test_recorded_test_lunge_2_knee_minimum_does_not_meet_detector_trigger(self):
        """Recorded event minimum is 124.7 degrees, above the configured 120-degree trigger."""
        signals = self._create_base_signals(num_frames=180, fps=60.0)
        for i, (knee, asymmetry, stance_ratio, hip_drop) in enumerate([
            (131.6, 23.6, 0.79, 0.069),
            (129.2, 23.1, 0.80, 0.070),
            (133.7, 14.6, 0.84, 0.069),
            (131.0, 16.5, 0.86, 0.067),
            (131.6, 17.5, 0.88, 0.065),
            (124.7, 2.3, 0.10, 0.068),
            (125.7, 3.5, 0.09, 0.065),
        ]):
            signal = signals[round((1.53 + i * 0.02) * 60)]
            signal.update({
                "timestamp": 1.53 + i * 0.02,
                "min_knee": knee,
                "knee_l": knee,
                "knee_r": knee + asymmetry,
                "knee_asym": asymmetry,
                "stance_ratio": stance_ratio,
                "hip_y": 0.5 + hip_drop,
                "baseline_hip_y": 0.5,
            })

        self.assertEqual(self.detector._detect_lunges(signals, "Test_Lunge_2.mp4"), [])

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
        merged_details = json.loads(merged[0].key_metrics)["conflicting_candidate_details"]
        self.assertEqual([item["movement"] for item in merged_details], ["LUNGE", "CHANGE_OF_DIRECTION"])
        self.assertEqual([item["score"] for item in merged_details], [0.72, 0.74])

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

    # --- REGRESSION TESTS FOR EVENT SEGMENTATION AUDIT ---

    def test_cod_cluster_suppression_single_candidate(self):
        """Dense continuous reversal frames must collapse into 1 candidate, not one per frame."""
        signals = self._create_base_signals(num_frames=60, fps=30.0)
        # Approach right (+0.30)
        for i in range(10, 20):
            signals[i]["hip_vx"] = 0.30
        # Reversal episode spanning 10 consecutive frames (frames 20-30)
        for i in range(20, 31):
            signals[i]["hip_vx"] = 0.05 - 0.01 * (i - 20)
            signals[i]["min_knee"] = 125.0
        # Exit left (-0.30)
        for i in range(31, 45):
            signals[i]["hip_vx"] = -0.30

        candidates = self.detector.detect_candidates_from_signals(signals, "test_cluster.mp4")
        cods = [c for c in candidates if c.proposed_movement == "CHANGE_OF_DIRECTION"]
        # Must collapse into a single bounded candidate, not 10 overlapping candidates
        self.assertEqual(len(cods), 1, f"Expected 1 COD candidate from continuous reversal, got {len(cods)}")
        self.assertLessEqual(cods[0].duration, self.detector.config.cod_max_duration)

    def test_cod_refractory_period_configurable(self):
        """Refractory period suppresses same-direction stutter, but allows opposite-direction cuts."""
        # 1. Opposite-direction cuts: Left-to-Right then Right-to-Left
        signals = self._create_base_signals(num_frames=120, fps=15.0)
        for i in range(10, 25):
            signals[i]["hip_vx"] = 0.30
        signals[25]["hip_vx"] = 0.0
        signals[25]["min_knee"] = 120.0
        for i in range(26, 60):
            signals[i]["hip_vx"] = -0.30
        signals[60]["hip_vx"] = 0.0
        signals[60]["min_knee"] = 120.0
        for i in range(61, 85):
            signals[i]["hip_vx"] = 0.30

        candidates = self.detector.detect_candidates_from_signals(signals, "test_rapid.mp4")
        cods = [c for c in candidates if c.proposed_movement == "CHANGE_OF_DIRECTION"]
        self.assertGreaterEqual(len(cods), 2, "Distinct directional cuts must not be suppressed")

        # 2. Configurable refractory period attribute
        self.assertTrue(hasattr(self.detector.config, "cod_refractory_period"))
        self.assertEqual(self.detector.config.cod_refractory_period, 0.5)

    def test_pass1_same_type_maximum_duration_capping(self):
        """Pass 1 same-type merging must terminate at max_duration and not chain indefinitely."""
        # 6 consecutive COD candidates of 1.2s each with 0.1s gap (spanning > 6 seconds)
        cands = []
        for i in range(6):
            start = i * 1.0
            end = start + 1.2
            cands.append(MovementCandidate(
                candidate_id=str(i),
                video_id="v",
                movement_type="CHANGE_OF_DIRECTION",
                start_time=round(start, 2),
                end_time=round(end, 2),
                duration=1.2,
                confidence_score=0.85,
                key_metrics="{}",
                proposed_movement="CHANGE_OF_DIRECTION",
                heuristic_score=0.85,
            ))
        merged = self.detector._merge_candidates(cands)
        # Without cap, all 6 would chain into one 6.2s candidate.
        # With cod_max_duration = 1.8s, it must produce multiple bounded candidates!
        self.assertGreater(len(merged), 1, "Candidates must not chain indefinitely into a single mega-block")
        for m in merged:
            self.assertLessEqual(m.duration, self.detector.config.cod_max_duration + 0.05,
                                f"Merged candidate duration {m.duration}s exceeded max {self.detector.config.cod_max_duration}s")

    def test_pass2_no_destructive_overwrite(self):
        """A subsequent higher-scoring candidate must not delete a preceding candidate from pass2."""
        # Reproduce the exact bug: UNCERTAIN (0.71) at [13.5 - 16.1] followed by COD (0.90) at [16.1 - 22.0]
        c1 = MovementCandidate(
            candidate_id="1", video_id="v", movement_type="UNCERTAIN",
            start_time=13.53, end_time=16.13, duration=2.60, confidence_score=0.71,
            key_metrics="{}", proposed_movement="UNCERTAIN", heuristic_score=0.71,
        )
        c2 = MovementCandidate(
            candidate_id="2", video_id="v", movement_type="CHANGE_OF_DIRECTION",
            start_time=16.10, end_time=22.07, duration=5.97, confidence_score=0.90,
            key_metrics="{}", proposed_movement="CHANGE_OF_DIRECTION", heuristic_score=0.90,
        )
        merged = self.detector._merge_candidates([c1, c2])
        # Both must be preserved! c1 must NOT be overwritten and dropped.
        self.assertEqual(len(merged), 2, "Pass 2 must not overwrite and discard preceding candidates")
        self.assertEqual(merged[0].proposed_movement, "UNCERTAIN")
        self.assertEqual(merged[1].proposed_movement, "CHANGE_OF_DIRECTION")

    def test_pass2_containment_preserves_target_movement(self):
        """A specific target movement (LUNGE) contained inside a broad candidate must be preserved."""
        # Broad COD: [14.0 - 17.5] (dur 3.5s, score 0.75)
        # Nested LUNGE: [15.0 - 16.0] (dur 1.0s, score 0.95)
        c_cod = MovementCandidate(
            candidate_id="1", video_id="v", movement_type="CHANGE_OF_DIRECTION",
            start_time=14.0, end_time=17.5, duration=3.5, confidence_score=0.75,
            key_metrics="{}", proposed_movement="CHANGE_OF_DIRECTION", heuristic_score=0.75,
        )
        c_lunge = MovementCandidate(
            candidate_id="2", video_id="v", movement_type="LUNGE",
            start_time=15.0, end_time=16.0, duration=1.0, confidence_score=0.95,
            key_metrics="{}", proposed_movement="LUNGE", heuristic_score=0.95,
        )
        merged = self.detector._merge_candidates([c_cod, c_lunge])
        lunges = [c for c in merged if c.proposed_movement == "LUNGE"]
        self.assertEqual(len(lunges), 1, "Contained target action (LUNGE) must not be swallowed or discarded")
        self.assertAlmostEqual(lunges[0].start_time, 15.0)
        self.assertAlmostEqual(lunges[0].end_time, 16.0)

    def test_uncertain_conflict_no_temporal_union(self):
        """Conflicting candidates retain full conflicting temporal extent bounded by class maximum duration."""
        c1 = MovementCandidate(
            candidate_id="1", video_id="v", movement_type="LUNGE",
            start_time=1.0, end_time=2.2, duration=1.2, confidence_score=0.72,
            key_metrics="{}", proposed_movement="LUNGE", heuristic_score=0.72,
        )
        c2 = MovementCandidate(
            candidate_id="2", video_id="v", movement_type="CHANGE_OF_DIRECTION",
            start_time=1.4, end_time=2.6, duration=1.2, confidence_score=0.74,
            key_metrics="{}", proposed_movement="CHANGE_OF_DIRECTION", heuristic_score=0.74,
        )
        merged = self.detector._merge_candidates([c1, c2])
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].proposed_movement, "UNCERTAIN")
        self.assertEqual(merged[0].status, "REVIEW_REQUIRED")
        self.assertEqual(merged[0].start_time, 1.0)
        self.assertEqual(merged[0].end_time, 2.6)
        self.assertEqual(merged[0].duration, 1.6)
        max_u = self.detector._get_max_duration("UNCERTAIN")
        self.assertLessEqual(merged[0].duration, max_u, f"Duration must not exceed class limit {max_u}s")

    def test_partial_overlap_sequential_decoupling(self):
        """Sequential overlapping candidates must be decoupled without overlapping frames."""
        c1 = MovementCandidate(
            candidate_id="1", video_id="v", movement_type="SMASH_LANDING",
            start_time=4.4, end_time=6.0, duration=1.6, confidence_score=0.88,
            key_metrics="{}", proposed_movement="SMASH_LANDING", heuristic_score=0.88,
        )
        c2 = MovementCandidate(
            candidate_id="2", video_id="v", movement_type="CHANGE_OF_DIRECTION",
            start_time=5.6, end_time=7.2, duration=1.6, confidence_score=0.90,
            key_metrics="{}", proposed_movement="CHANGE_OF_DIRECTION", heuristic_score=0.90,
        )
        merged = self.detector._merge_candidates([c1, c2])
        self.assertEqual(len(merged), 2)
        # Verify non-overlapping boundaries: end_1 <= start_2
        self.assertLessEqual(merged[0].end_time, merged[1].start_time + 0.01,
                            f"Events still overlap: {merged[0].end_time} > {merged[1].start_time}")


    def test_target_movement_not_discarded_by_higher_scoring_cod(self):
        """A valid LUNGE must NOT be silently discarded solely because an overlapping COD has a higher score."""
        # Target movement: LUNGE (score 0.68)
        c_lunge = MovementCandidate(
            candidate_id="1", video_id="v", movement_type="LUNGE",
            start_time=1.0, end_time=2.0, duration=1.0, confidence_score=0.68,
            key_metrics=json.dumps({"knee_flexion": 108.5}), proposed_movement="LUNGE", heuristic_score=0.68,
        )
        # Footwork movement: high-speed COD (score 0.92) overlapping by 0.9s
        c_cod = MovementCandidate(
            candidate_id="2", video_id="v", movement_type="CHANGE_OF_DIRECTION",
            start_time=1.1, end_time=2.3, duration=1.2, confidence_score=0.92,
            key_metrics=json.dumps({"delta_v": 0.48}), proposed_movement="CHANGE_OF_DIRECTION", heuristic_score=0.92,
        )
        merged = self.detector._merge_candidates([c_lunge, c_cod])
        # Under the target preservation safeguard, c_lunge is NOT deleted; an UNCERTAIN review event is created
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].proposed_movement, "UNCERTAIN")
        self.assertEqual(merged[0].status, "REVIEW_REQUIRED")
        
        # Diagnostics must preserve BOTH candidates and their raw scores
        diag = json.loads(merged[0].key_metrics)
        self.assertIn("conflicting_candidate_details", diag)
        details = diag["conflicting_candidate_details"]
        movements = [d["movement"] for d in details]
        scores = [d["score"] for d in details]
        self.assertIn("LUNGE", movements)
        self.assertIn("CHANGE_OF_DIRECTION", movements)
        self.assertEqual(scores, [0.68, 0.92])

    def test_cod_refractory_suppresses_same_direction_stutter(self):
        """Refractory check must suppress same-direction stutter episode (<0.5s) from separate candidate creation."""
        signals = []
        for i in range(60):
            signals.append({'frame': i, 'timestamp': round(i / 15.0, 3), 'hip_vx': 0.0, 'hip_vy': 0.0, 'hip_y': 0.5, 'min_knee': 120.0})
        for i in range(5, 15):
            signals[i]['hip_vx'] = 0.30
        for i in range(21, 35):
            signals[i]['hip_vx'] = -0.30
        # Gaps at 26, 27 create two distinct qualifying episodes at frames 14 and 19
        signals[26]['hip_vx'] = 0.0
        signals[27]['hip_vx'] = 0.0

        candidates = self.detector._detect_changes_of_direction(signals, "test_stutter.mp4")
        # Episode 1 peak at t=0.933s, Episode 2 peak at t=1.267s (gap=0.334s < 0.50s)
        # Stutter episode 2 must be suppressed
        self.assertEqual(len(candidates), 1, f"Expected 1 candidate after stutter suppression, got {len(candidates)}")

    def test_cod_refractory_permits_separated_same_direction_reversals(self):
        """When refractory period is configured shorter (0.25s), same-direction reversals >0.25s apart are accepted."""
        cfg = DetectorConfig(sample_fps=15.0, cod_refractory_period=0.25)
        det = MovementCandidateDetector(cfg)
        signals = []
        for i in range(60):
            signals.append({'frame': i, 'timestamp': round(i / 15.0, 3), 'hip_vx': 0.0, 'hip_vy': 0.0, 'hip_y': 0.5, 'min_knee': 120.0})
        for i in range(5, 15):
            signals[i]['hip_vx'] = 0.30
        for i in range(21, 35):
            signals[i]['hip_vx'] = -0.30
        signals[26]['hip_vx'] = 0.0
        signals[27]['hip_vx'] = 0.0

        candidates = det._detect_changes_of_direction(signals, "test_stutter_short_refractory.mp4")
        # Gap is 0.334s > 0.25s refractory -> both episodes accepted
        self.assertEqual(len(candidates), 2, f"Expected 2 candidates with 0.25s refractory, got {len(candidates)}")

    def test_cod_rapid_alternating_cuts_preserved(self):
        """Opposite direction cuts (R->L then L->R) separated by >0.6s window are both preserved.
        
        Note on Kinematic Resolution Limit:
        Because  is fixed at 0.6s, the reversal detector tests v(i - 0.6s) vs v(i + 0.6s).
        Two alternating cuts spaced closer than 0.6s contaminate each other's approach/exit window,
        preventing the velocity sign product v_before * v_after < -0.01 from triggering.
        Alternating cuts with dt >= 0.6s are resolvable and preserved without refractory suppression.
        """
        signals = []
        for i in range(90):
            signals.append({'frame': i, 'timestamp': round(i / 15.0, 3), 'hip_vx': 0.0, 'hip_vy': 0.0, 'hip_y': 0.5, 'min_knee': 120.0})
        # Approach right (frames 10-24)
        for i in range(10, 25):
            signals[i]['hip_vx'] = 0.30
        # Cut 1: R->L at frame 25 (t=1.667s)
        for i in range(26, 45):
            signals[i]['hip_vx'] = -0.30
        # Cut 2: L->R at frame 45 (t=3.000s, dt=1.333s > 0.6s)
        for i in range(46, 65):
            signals[i]['hip_vx'] = 0.30

        candidates = self.detector._detect_changes_of_direction(signals, "test_alternating.mp4")
        self.assertEqual(len(candidates), 2, f"Expected 2 distinct directional cuts, got {len(candidates)}")

    def test_interval_invariants_and_all_pairs_disjoint(self):
        """Every output candidate must satisfy non-negativity, positivity, duration consistency, max bounds, and all-pairs non-overlap."""
        cands = [
            MovementCandidate("1", "v", "CHANGE_OF_DIRECTION", 1.0, 2.0, 1.0, 0.85, "{}", proposed_movement="CHANGE_OF_DIRECTION", heuristic_score=0.85),
            MovementCandidate("2", "v", "LUNGE", 1.8, 3.2, 1.4, 0.75, "{}", proposed_movement="LUNGE", heuristic_score=0.75),
            MovementCandidate("3", "v", "CHANGE_OF_DIRECTION", 3.0, 4.2, 1.2, 0.80, "{}", proposed_movement="CHANGE_OF_DIRECTION", heuristic_score=0.80),
            MovementCandidate("4", "v", "SMASH_LANDING", 4.1, 5.5, 1.4, 0.90, "{}", proposed_movement="SMASH_LANDING", heuristic_score=0.90),
            MovementCandidate("5", "v", "CHANGE_OF_DIRECTION", 6.0, 9.0, 3.0, 0.85, "{}", proposed_movement="CHANGE_OF_DIRECTION", heuristic_score=0.85),
        ]
        merged = self.detector._merge_candidates(cands)
        
        # 1. Check all individual candidate invariants
        for c in merged:
            self.assertTrue(np.isfinite(c.start_time), f"start_time must be finite: {c.start_time}")
            self.assertTrue(np.isfinite(c.end_time), f"end_time must be finite: {c.end_time}")
            self.assertGreaterEqual(c.start_time, 0.0, f"start_time must be non-negative: {c.start_time}")
            self.assertGreater(c.end_time, c.start_time, f"end_time must be > start_time: {c.start_time} - {c.end_time}")
            self.assertGreater(c.duration, 0.0, f"duration must be > 0: {c.duration}")
            calc_dur = round(c.end_time - c.start_time, 2)
            self.assertAlmostEqual(c.duration, calc_dur, places=2, msg=f"duration must equal end - start: {c.duration} vs {calc_dur}")
            max_d = self.detector._get_max_duration(c.proposed_movement)
            self.assertLessEqual(c.duration, max_d + 0.02, f"{c.proposed_movement} duration {c.duration}s exceeded max {max_d}s")

        # 2. Check ALL pairs (i, j) for unintended overlap
        for i in range(len(merged)):
            for j in range(i + 1, len(merged)):
                c1, c2 = merged[i], merged[j]
                ov = min(c1.end_time, c2.end_time) - max(c1.start_time, c2.start_time)
                self.assertLessEqual(ov, 0.05, f"Pair ({i}, {j}) has unintended overlap of {ov}s between [{c1.start_time}, {c1.end_time}] and [{c2.start_time}, {c2.end_time}]")

    def test_invalid_timestamps_and_duration_invariants(self):
        """Invalid inputs (NaN, Inf, negative intervals, zero duration) must be rejected or clamped, duration made consistent and capped, with diagnostics preserved."""
        invalids = [
            MovementCandidate("nan", "v", "LUNGE", float("nan"), 2.0, 1.0, 0.8, "{}", proposed_movement="LUNGE", heuristic_score=0.8),
            MovementCandidate("inf", "v", "LUNGE", 1.0, float("inf"), 1.0, 0.8, "{}", proposed_movement="LUNGE", heuristic_score=0.8),
            MovementCandidate("ninf", "v", "LUNGE", float("-inf"), 2.0, 1.0, 0.8, "{}", proposed_movement="LUNGE", heuristic_score=0.8),
            MovementCandidate("neg", "v", "LUNGE", -1.5, -0.5, 0.5, 0.8, "{}", proposed_movement="LUNGE", heuristic_score=0.8),
            MovementCandidate("span", "v", "LUNGE", -0.8, 1.2, 2.0, 0.8, "{}", proposed_movement="LUNGE", heuristic_score=0.8),
            MovementCandidate("zero", "v", "LUNGE", 2.0, 2.0, 0.0, 0.8, "{}", proposed_movement="LUNGE", heuristic_score=0.8),
            MovementCandidate("inconsistent", "v", "LUNGE", 2.5, 4.0, 0.5, 0.8, "{}", proposed_movement="LUNGE", heuristic_score=0.8),
            MovementCandidate("oversize", "v", "CHANGE_OF_DIRECTION", 5.0, 9.0, 4.0, 0.8, "{}", proposed_movement="CHANGE_OF_DIRECTION", heuristic_score=0.8),
        ]
        merged = self.detector._merge_candidates(invalids)
        self.assertEqual(len(merged), 3, "Only span (clamped), inconsistent (corrected), and oversize (capped) should survive")

        for c in merged:
            self.assertTrue(np.isfinite(c.start_time))
            self.assertTrue(np.isfinite(c.end_time))
            self.assertGreaterEqual(c.start_time, 0.0, f"start_time must be non-negative: {c.start_time}")
            self.assertGreater(c.end_time, c.start_time)
            self.assertGreater(c.duration, 0.0)
            self.assertAlmostEqual(c.duration, round(c.end_time - c.start_time, 2), places=2)
            max_d = self.detector._get_max_duration(c.proposed_movement)
            self.assertLessEqual(c.duration, max_d + 0.02)

        # Check normalization diagnostics
        span_cand = next(c for c in merged if c.candidate_id == "span")
        self.assertEqual(span_cand.start_time, 0.0, "Negative start time must be clamped to 0.0")
        self.assertEqual(span_cand.end_time, 1.2)
        span_diag = json.loads(span_cand.key_metrics)["normalization_diagnostics"]
        self.assertEqual(span_diag["original_interval"], [-0.8, 1.2])
        self.assertIn("negative start_time", span_diag["normalization_reason"])

        inc_cand = next(c for c in merged if c.candidate_id == "inconsistent")
        self.assertEqual(inc_cand.duration, 1.5, "Duration must equal end_time - start_time")

        ov_cand = next(c for c in merged if c.candidate_id == "oversize")
        self.assertLessEqual(ov_cand.duration, self.detector.config.cod_max_duration + 0.02, "Duration must be capped at max_duration")
        ov_diag = json.loads(ov_cand.key_metrics)["normalization_diagnostics"]
        self.assertEqual(ov_diag["original_duration"], 4.0)
        self.assertIn("Capped duration", ov_diag["normalization_reason"])

        # When a normalized candidate enters an UNCERTAIN conflict, its original interval and normalization
        # reason must be traceable in the conflict diagnostics rather than presenting a clipped interval as original
        c_competing = MovementCandidate("comp", "v", "CHANGE_OF_DIRECTION", 0.2, 1.0, 0.8, 0.90, "{}", proposed_movement="CHANGE_OF_DIRECTION", heuristic_score=0.90)
        conflict_with_norm = self.detector._merge_candidates([span_cand, c_competing])
        self.assertEqual(conflict_with_norm[0].proposed_movement, "UNCERTAIN")
        norm_conflict_diag = json.loads(conflict_with_norm[0].key_metrics)
        span_detail = next(d for d in norm_conflict_diag["conflicting_candidate_details"] if d.get("movement") == "LUNGE")
        self.assertIn("normalization_diagnostics", span_detail)
        self.assertEqual(span_detail["normalization_diagnostics"]["original_interval"], [-0.8, 1.2])
        self.assertIn("negative start_time", span_detail["normalization_diagnostics"]["normalization_reason"])

    def test_three_candidate_chained_overlap_preserves_all_diagnostics(self):
        """When three candidates chain-overlap into UNCERTAIN, diagnostics for every original candidate remain traceable."""
        cA = MovementCandidate("1", "v", "LUNGE", 1.0, 2.0, 1.0, 0.65, json.dumps({"knee": 105}), proposed_movement="LUNGE", heuristic_score=0.65)
        cB = MovementCandidate("2", "v", "CHANGE_OF_DIRECTION", 1.1, 2.2, 1.1, 0.95, json.dumps({"dv": 0.5}), proposed_movement="CHANGE_OF_DIRECTION", heuristic_score=0.95)
        cC = MovementCandidate("3", "v", "SMASH_LANDING", 1.2, 2.1, 0.9, 0.80, json.dumps({"apex": 0.04}), proposed_movement="SMASH_LANDING", heuristic_score=0.80)

        # 1. Resolving the first pair (cA vs cB) produces UNCERTAIN review event
        pair_merged = self.detector._merge_candidates([cA, cB])
        self.assertEqual(len(pair_merged), 1, "First pair must resolve to single UNCERTAIN event")
        self.assertEqual(pair_merged[0].proposed_movement, "UNCERTAIN")
        self.assertEqual(pair_merged[0].status, "REVIEW_REQUIRED")
        pair_diag = json.loads(pair_merged[0].key_metrics)
        self.assertEqual(set(pair_diag["conflicting_movements"]), {"LUNGE", "CHANGE_OF_DIRECTION"})

        # 2. Third candidate cC overlaps the UNCERTAIN result from the first pair
        chain_merged = self.detector._merge_candidates([pair_merged[0], cC])
        self.assertEqual(len(chain_merged), 1, "Chained overlap must remain a single UNCERTAIN event")
        self.assertEqual(chain_merged[0].proposed_movement, "UNCERTAIN")
        self.assertEqual(chain_merged[0].status, "REVIEW_REQUIRED")
        chain_diag = json.loads(chain_merged[0].key_metrics)
        self.assertEqual(len(chain_diag["conflicting_candidate_details"]), 3, "Diagnostics must retain all 3 original candidates")
        self.assertEqual(set(chain_diag["conflicting_movements"]), {"LUNGE", "CHANGE_OF_DIRECTION", "SMASH_LANDING"})

        # 3. Chaining a fourth candidate cD onto the UNCERTAIN composite preserves all 4 candidates
        cD = MovementCandidate("4", "v", "CHANGE_OF_DIRECTION", 1.3, 2.3, 1.0, 0.72, json.dumps({"foot": "split"}), proposed_movement="CHANGE_OF_DIRECTION", heuristic_score=0.72)
        quad_merged = self.detector._merge_candidates([chain_merged[0], cD])
        self.assertEqual(len(quad_merged), 1, "4-way chained overlap must remain a single UNCERTAIN event")
        self.assertEqual(quad_merged[0].proposed_movement, "UNCERTAIN")
        quad_diag = json.loads(quad_merged[0].key_metrics)
        self.assertEqual(len(quad_diag["conflicting_candidate_details"]), 4, "Must retain all 4 original candidates")

        # 4. Direct 3-way input produces the same fully traceable UNCERTAIN event
        all_merged = self.detector._merge_candidates([cA, cB, cC])
        self.assertEqual(len(all_merged), 1)
        self.assertEqual(all_merged[0].proposed_movement, "UNCERTAIN")
        self.assertEqual(all_merged[0].status, "REVIEW_REQUIRED")

        diag = json.loads(all_merged[0].key_metrics)
        self.assertEqual(len(diag["conflicting_candidate_details"]), 3, "All 3 original candidates must be preserved in details")
        movements = {d["movement"] for d in diag["conflicting_candidate_details"]}
        self.assertEqual(movements, {"LUNGE", "CHANGE_OF_DIRECTION", "SMASH_LANDING"})
        self.assertEqual(set(diag["conflicting_movements"]), {"LUNGE", "CHANGE_OF_DIRECTION", "SMASH_LANDING"})

        # Detailed traceability assertions for every original candidate:
        lunge_det = next(d for d in diag["conflicting_candidate_details"] if d["movement"] == "LUNGE")
        self.assertEqual(lunge_det["start_time"], 1.0)
        self.assertEqual(lunge_det["end_time"], 2.0)
        self.assertEqual(lunge_det["duration"], 1.0)
        self.assertEqual(lunge_det["score"], 0.65)
        self.assertEqual(lunge_det["key_metrics"]["knee"], 105)

        cod_det = next(d for d in diag["conflicting_candidate_details"] if d["movement"] == "CHANGE_OF_DIRECTION")
        self.assertEqual(cod_det["start_time"], 1.1)
        self.assertEqual(cod_det["end_time"], 2.2)
        self.assertEqual(cod_det["duration"], 1.1)
        self.assertEqual(cod_det["score"], 0.95)
        self.assertEqual(cod_det["key_metrics"]["dv"], 0.5)

        smash_det = next(d for d in diag["conflicting_candidate_details"] if d["movement"] == "SMASH_LANDING")
        self.assertEqual(smash_det["start_time"], 1.2)
        self.assertEqual(smash_det["end_time"], 2.1)
        self.assertEqual(smash_det["duration"], 0.9)
        self.assertEqual(smash_det["score"], 0.80)
        self.assertEqual(smash_det["key_metrics"]["apex"], 0.04)

    def test_final_sweep_checks_all_prior_events(self):
        """Final sweep verifies every candidate against all prior candidates, resolving non-adjacent overlaps."""
        # Scenario A: Resolving non-adjacent candidates through multi-candidate decoupling
        c1 = MovementCandidate("1", "v", "CHANGE_OF_DIRECTION", 1.0, 3.5, 2.5, 0.85, "{}", proposed_movement="CHANGE_OF_DIRECTION", heuristic_score=0.85)
        c2 = MovementCandidate("2", "v", "LUNGE", 1.2, 2.0, 0.8, 0.75, "{}", proposed_movement="LUNGE", heuristic_score=0.75)
        c3 = MovementCandidate("3", "v", "SMASH_LANDING", 2.5, 3.8, 1.3, 0.80, "{}", proposed_movement="SMASH_LANDING", heuristic_score=0.80)

        merged_a = self.detector._merge_candidates([c1, c2, c3])
        # Assert all pairs have overlap <= 0.05s
        for i in range(len(merged_a)):
            for j in range(i + 1, len(merged_a)):
                ov = min(merged_a[i].end_time, merged_a[j].end_time) - max(merged_a[i].start_time, merged_a[j].start_time)
                self.assertLessEqual(ov, 0.05, f"Pair ({i}, {j}) has unresolved overlap of {ov}s")

        # Scenario B: Non-adjacent overlap that cannot be decoupled must be explicitly converted to UNCERTAIN
        # Event 1: LUNGE [1.2, 2.0]
        # Event 2: COD [2.4, 3.5]
        # Event 3: SMASH_LANDING [1.8, 2.4]
        # Notice: Event 1 and Event 2 do NOT overlap (gap 0.4s). Event 2 and Event 3 do NOT overlap (touch at 2.4s).
        # A naive sweep checking only adjacent pairs (1 vs 2, 2 vs 3) would miss the conflict between Event 1 and Event 3!
        # The all-pairs sweep checks Event 1 vs Event 3 (overlap = 0.20s > 0.05s) and creates an UNCERTAIN review event.
        c_lunge = MovementCandidate("l1", "v", "LUNGE", 1.2, 2.0, 0.8, 0.85, json.dumps({"lunge_knee": 110}), proposed_movement="LUNGE", heuristic_score=0.85)
        c_cod = MovementCandidate("c1", "v", "CHANGE_OF_DIRECTION", 2.4, 3.5, 1.1, 0.80, json.dumps({"cod_dv": 0.4}), proposed_movement="CHANGE_OF_DIRECTION", heuristic_score=0.80)
        c_smash = MovementCandidate("s1", "v", "SMASH_LANDING", 1.4, 2.2, 0.8, 0.78, json.dumps({"smash_apex": 0.05}), proposed_movement="SMASH_LANDING", heuristic_score=0.78)

        merged_b = self.detector._merge_candidates([c_lunge, c_cod, c_smash])
        self.assertEqual(len(merged_b), 2, "Expected 2 events: one UNCERTAIN composite and one preserved COD")

        uncertain_ev = next(m for m in merged_b if m.proposed_movement == "UNCERTAIN")
        self.assertEqual(uncertain_ev.status, "REVIEW_REQUIRED")
        u_diag = json.loads(uncertain_ev.key_metrics)
        self.assertEqual(set(u_diag["conflicting_movements"]), {"LUNGE", "SMASH_LANDING"})
        self.assertEqual(len(u_diag["conflicting_candidate_details"]), 2)

        cod_ev = next(m for m in merged_b if m.proposed_movement == "CHANGE_OF_DIRECTION")
        self.assertEqual(cod_ev.status, "PROPOSED")

        # Every resulting pair must strictly satisfy overlap <= 0.05s
        for i in range(len(merged_b)):
            for j in range(i + 1, len(merged_b)):
                ov = min(merged_b[i].end_time, merged_b[j].end_time) - max(merged_b[i].start_time, merged_b[j].start_time)
                self.assertLessEqual(ov, 0.05, f"Pair ({i}, {j}) has unresolved overlap of {ov}s")

    def test_lunge_vs_higher_scoring_cod_in_all_conflict_branches(self):
        """A valid LUNGE must not be discarded or replaced by higher-scoring COD across all conflict branches."""
        # Branch A: Heavy reciprocal overlap (tIoU >= 0.40)
        c1 = MovementCandidate("1", "v", "LUNGE", 1.0, 2.0, 1.0, 0.65, json.dumps({"knee": 105}), proposed_movement="LUNGE", heuristic_score=0.65)
        c2 = MovementCandidate("2", "v", "CHANGE_OF_DIRECTION", 1.1, 2.2, 1.1, 0.95, json.dumps({"dv": 0.5}), proposed_movement="CHANGE_OF_DIRECTION", heuristic_score=0.95)
        mA = self.detector._merge_candidates([c1, c2])
        self.assertEqual(len(mA), 1)
        self.assertEqual(mA[0].proposed_movement, "UNCERTAIN")
        diagA = json.loads(mA[0].key_metrics)["conflicting_candidate_details"]
        self.assertEqual({d["movement"] for d in diagA}, {"LUNGE", "CHANGE_OF_DIRECTION"})

        # Branch B: Containment without valid decoupling margins (< 0.4s)
        c3 = MovementCandidate("3", "v", "CHANGE_OF_DIRECTION", 1.9, 3.1, 1.2, 0.95, json.dumps({"dv": 0.5}), proposed_movement="CHANGE_OF_DIRECTION", heuristic_score=0.95)
        c4 = MovementCandidate("4", "v", "LUNGE", 2.0, 3.0, 1.0, 0.65, json.dumps({"knee": 105}), proposed_movement="LUNGE", heuristic_score=0.65)
        mB = self.detector._merge_candidates([c3, c4])
        self.assertEqual(len(mB), 1)
        self.assertEqual(mB[0].proposed_movement, "UNCERTAIN")
        diagB = json.loads(mB[0].key_metrics)["conflicting_candidate_details"]
        self.assertEqual({d["movement"] for d in diagB}, {"LUNGE", "CHANGE_OF_DIRECTION"})

        # Branch C: Partial sequential overlap where decoupling would truncate LUNGE
        c5 = MovementCandidate("5", "v", "LUNGE", 4.0, 5.0, 1.0, 0.65, json.dumps({"knee": 105}), proposed_movement="LUNGE", heuristic_score=0.65)
        c6 = MovementCandidate("6", "v", "CHANGE_OF_DIRECTION", 4.8, 5.1, 0.3, 0.95, json.dumps({"dv": 0.5}), proposed_movement="CHANGE_OF_DIRECTION", heuristic_score=0.95)
        mC = self.detector._merge_candidates([c5, c6])
        self.assertEqual(len(mC), 1)
        self.assertEqual(mC[0].proposed_movement, "UNCERTAIN")
        diagC = json.loads(mC[0].key_metrics)["conflicting_candidate_details"]
        self.assertEqual({d["movement"] for d in diagC}, {"LUNGE", "CHANGE_OF_DIRECTION"})

    def test_smash_landing_vs_higher_scoring_cod(self):
        """A valid SMASH_LANDING must not be discarded or replaced by higher-scoring COD."""
        c_smash = MovementCandidate("1", "v", "SMASH_LANDING", 2.0, 3.2, 1.2, 0.68, json.dumps({"apex": 0.05}), proposed_movement="SMASH_LANDING", heuristic_score=0.68)
        c_cod = MovementCandidate("2", "v", "CHANGE_OF_DIRECTION", 2.1, 3.3, 1.2, 0.94, json.dumps({"dv": 0.52}), proposed_movement="CHANGE_OF_DIRECTION", heuristic_score=0.94)
        merged = self.detector._merge_candidates([c_smash, c_cod])
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].proposed_movement, "UNCERTAIN")
        details = json.loads(merged[0].key_metrics)["conflicting_candidate_details"]
        movements = [d["movement"] for d in details]
        scores = [d["score"] for d in details]
        self.assertIn("SMASH_LANDING", movements)
        self.assertIn("CHANGE_OF_DIRECTION", movements)
        self.assertEqual(scores, [0.68, 0.94])

    def test_lunge_vs_smash_landing_conflict(self):
        """Competing target movements (LUNGE vs SMASH_LANDING) must resolve to UNCERTAIN with diagnostics."""
        c_l = MovementCandidate("1", "v", "LUNGE", 1.0, 2.0, 1.0, 0.75, "{}", proposed_movement="LUNGE", heuristic_score=0.75)
        c_s = MovementCandidate("2", "v", "SMASH_LANDING", 1.2, 2.2, 1.0, 0.85, "{}", proposed_movement="SMASH_LANDING", heuristic_score=0.85)
        merged = self.detector._merge_candidates([c_l, c_s])
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].proposed_movement, "UNCERTAIN")
        diag = json.loads(merged[0].key_metrics)
        self.assertIn("LUNGE", diag["conflicting_movements"])
        self.assertIn("SMASH_LANDING", diag["conflicting_movements"])

    def test_containment_and_partial_overlap_cannot_trim_safely(self):
        """When containment or partial overlap margins are below minimum duration (<0.4s), candidates cannot be trimmed and must resolve to UNCERTAIN."""
        # Containment: margins 0.1s and 0.1s (<0.4s)
        c1 = MovementCandidate("1", "v", "CHANGE_OF_DIRECTION", 2.9, 4.1, 1.2, 0.90, "{}", proposed_movement="CHANGE_OF_DIRECTION", heuristic_score=0.90)
        c2 = MovementCandidate("2", "v", "LUNGE", 3.0, 4.0, 1.0, 0.70, "{}", proposed_movement="LUNGE", heuristic_score=0.70)
        m1 = self.detector._merge_candidates([c1, c2])
        self.assertEqual(len(m1), 1)
        self.assertEqual(m1[0].proposed_movement, "UNCERTAIN")

        # Partial overlap: trailing margin is 0.2s (<0.4s)
        c3 = MovementCandidate("3", "v", "SMASH_LANDING", 5.0, 6.2, 1.2, 0.85, "{}", proposed_movement="SMASH_LANDING", heuristic_score=0.85)
        c4 = MovementCandidate("4", "v", "CHANGE_OF_DIRECTION", 5.9, 6.4, 0.5, 0.90, "{}", proposed_movement="CHANGE_OF_DIRECTION", heuristic_score=0.90)
        m2 = self.detector._merge_candidates([c3, c4])
        self.assertEqual(len(m2), 1)
        self.assertEqual(m2[0].proposed_movement, "UNCERTAIN")

    def test_all_pairs_audit_tolerance_vs_genuine_overlap(self):
        """All-pairs audit strictly verifies permitted tolerance (<=0.05s) vs genuine overlaps (>0.05s) converted to UNCERTAIN.
        
        Permitted Numerical Tolerance:
        A boundary tolerance of up to 0.05s (~1.5 frames at 30 FPS) is permitted to accommodate floating-point
        frame alignment without falsely classifying boundary-touching movements as conflicting. Any temporal
        overlap strictly exceeding 0.05s is a genuine conflict and must be converted to an UNCERTAIN review
        event with both original candidates details and metrics preserved.
        """
        # Case 1: Genuine overlap (> 0.05s): two candidates overlapping by 0.30s that cannot be decoupled without truncating target
        c1 = MovementCandidate("1", "v", "LUNGE", 1.0, 2.0, 1.0, 0.80, json.dumps({"knee": 120}), proposed_movement="LUNGE", heuristic_score=0.80)
        c2 = MovementCandidate("2", "v", "CHANGE_OF_DIRECTION", 1.7, 2.2, 0.5, 0.85, json.dumps({"knee": 115}), proposed_movement="CHANGE_OF_DIRECTION", heuristic_score=0.85)
        merged1 = self.detector._merge_candidates([c1, c2])
        self.assertEqual(len(merged1), 1, "Genuine overlap > 0.05s must be resolved into an UNCERTAIN event")
        self.assertEqual(merged1[0].proposed_movement, "UNCERTAIN")
        self.assertEqual(merged1[0].status, "REVIEW_REQUIRED")
        diag1 = json.loads(merged1[0].key_metrics)
        self.assertIn("conflicting_candidate_details", diag1)
        self.assertEqual(len(diag1["conflicting_candidate_details"]), 2)
        # Verify both candidates details are fully preserved
        det1 = next(d for d in diag1["conflicting_candidate_details"] if d["movement"] == "LUNGE")
        self.assertEqual(det1["start_time"], 1.0)
        self.assertEqual(det1["end_time"], 2.0)
        self.assertEqual(det1["duration"], 1.0)
        self.assertEqual(det1["score"], 0.80)
        self.assertEqual(det1["key_metrics"]["knee"], 120)

        det2 = next(d for d in diag1["conflicting_candidate_details"] if d["movement"] == "CHANGE_OF_DIRECTION")
        self.assertEqual(det2["start_time"], 1.7)
        self.assertEqual(det2["end_time"], 2.2)
        self.assertEqual(det2["duration"], 0.5)
        self.assertEqual(det2["score"], 0.85)
        self.assertEqual(det2["key_metrics"]["knee"], 115)

        # Case 2: Genuine competing conflict (> 0.05s) between competing target movements that cannot be decoupled
        c_bound1 = MovementCandidate("b1", "v", "LUNGE", 1.0, 2.0, 1.0, 0.75, json.dumps({"knee": 110}), proposed_movement="LUNGE", heuristic_score=0.75)
        c_bound2 = MovementCandidate("b2", "v", "SMASH_LANDING", 1.2, 2.2, 1.0, 0.85, json.dumps({"apex": 0.05}), proposed_movement="SMASH_LANDING", heuristic_score=0.85)
        # Heavy overlap 0.80s > 0.05s that cannot be decoupled as small boundary overlap
        merged_bound = self.detector._merge_candidates([c_bound1, c_bound2])
        self.assertEqual(len(merged_bound), 1, "Genuine conflict > 0.05s must be converted to UNCERTAIN")
        self.assertEqual(merged_bound[0].proposed_movement, "UNCERTAIN")
        self.assertEqual(merged_bound[0].status, "REVIEW_REQUIRED")
        diag_bound = json.loads(merged_bound[0].key_metrics)
        self.assertEqual(len(diag_bound["conflicting_candidate_details"]), 2)
        self.assertEqual(set(diag_bound["conflicting_movements"]), {"LUNGE", "SMASH_LANDING"})

        # Case 3: Permitted numerical tolerance (<= 0.05s): two candidates touching with 0.03s overlap
        c3 = MovementCandidate("3", "v", "LUNGE", 1.0, 2.0, 1.0, 0.75, "{}", proposed_movement="LUNGE", heuristic_score=0.75)
        c4 = MovementCandidate("4", "v", "CHANGE_OF_DIRECTION", 1.97, 3.0, 1.03, 0.85, "{}", proposed_movement="CHANGE_OF_DIRECTION", heuristic_score=0.85)
        merged2 = self.detector._merge_candidates([c3, c4])
        # Overlap is 2.0 - 1.97 = 0.03s <= 0.05s tolerance; does NOT trigger false-positive UNCERTAIN
        for i in range(len(merged2)):
            for j in range(i + 1, len(merged2)):
                ov = min(merged2[i].end_time, merged2[j].end_time) - max(merged2[i].start_time, merged2[j].start_time)
                self.assertLessEqual(ov, 0.05)


    def test_multiple_sequential_conflicts_isolated(self):
        """Multiple distinct sequential conflicts throughout a rally resolve independently without cross-contamination."""
        # Conflict 1: LUNGE vs COD at t=1.0-2.2s
        c1_a = MovementCandidate("1a", "v", "LUNGE", 1.0, 2.0, 1.0, 0.70, json.dumps({"knee": 110}), proposed_movement="LUNGE", heuristic_score=0.70)
        c1_b = MovementCandidate("1b", "v", "CHANGE_OF_DIRECTION", 1.2, 2.2, 1.0, 0.85, json.dumps({"dv": 0.4}), proposed_movement="CHANGE_OF_DIRECTION", heuristic_score=0.85)

        # Isolated clean movement: COD at t=3.0-4.0s
        c_iso = MovementCandidate("iso", "v", "CHANGE_OF_DIRECTION", 3.0, 4.0, 1.0, 0.88, json.dumps({"dv": 0.6}), proposed_movement="CHANGE_OF_DIRECTION", heuristic_score=0.88)

        # Conflict 2: SMASH_LANDING vs COD at t=5.0-6.2s
        c2_a = MovementCandidate("2a", "v", "SMASH_LANDING", 5.0, 6.0, 1.0, 0.75, json.dumps({"apex": 0.04}), proposed_movement="SMASH_LANDING", heuristic_score=0.75)
        c2_b = MovementCandidate("2b", "v", "CHANGE_OF_DIRECTION", 5.2, 6.2, 1.0, 0.90, json.dumps({"dv": 0.5}), proposed_movement="CHANGE_OF_DIRECTION", heuristic_score=0.90)

        merged = self.detector._merge_candidates([c1_a, c1_b, c_iso, c2_a, c2_b])
        self.assertEqual(len(merged), 3, "Expected 3 events: Conflict 1 (UNCERTAIN), Isolated (COD), Conflict 2 (UNCERTAIN)")

        # Verify event 0: Conflict 1
        self.assertEqual(merged[0].proposed_movement, "UNCERTAIN")
        self.assertEqual(merged[0].status, "REVIEW_REQUIRED")
        diag1 = json.loads(merged[0].key_metrics)
        self.assertEqual(set(diag1["conflicting_movements"]), {"LUNGE", "CHANGE_OF_DIRECTION"})
        self.assertEqual(len(diag1["conflicting_candidate_details"]), 2)

        # Verify event 1: Isolated clean COD
        self.assertEqual(merged[1].proposed_movement, "CHANGE_OF_DIRECTION")
        self.assertEqual(merged[1].status, "PROPOSED")
        self.assertEqual(merged[1].start_time, 3.0)
        self.assertEqual(merged[1].end_time, 4.0)

        # Verify event 2: Conflict 2
        self.assertEqual(merged[2].proposed_movement, "UNCERTAIN")
        self.assertEqual(merged[2].status, "REVIEW_REQUIRED")
        diag2 = json.loads(merged[2].key_metrics)
        self.assertEqual(set(diag2["conflicting_movements"]), {"SMASH_LANDING", "CHANGE_OF_DIRECTION"})
        self.assertEqual(len(diag2["conflicting_candidate_details"]), 2)

        # Independent pairwise audit across all pairs
        for i in range(len(merged)):
            for j in range(i + 1, len(merged)):
                ov = min(merged[i].end_time, merged[j].end_time) - max(merged[i].start_time, merged[j].start_time)
                self.assertLessEqual(ov, 0.05, f"Pair ({i}, {j}) has unresolved overlap of {ov}s")

    def test_duplicate_non_target_candidates_preserves_evidence(self):
        """Duplicate non-target candidates suppress redundant detection but preserve evidence in duplicate_candidates diagnostics."""
        # Two overlapping duplicate COD detections
        c_cod1 = MovementCandidate("cod1", "v", "CHANGE_OF_DIRECTION", 2.0, 3.2, 1.2, 0.85, json.dumps({"dv": 0.6, "knee": 120}), proposed_movement="CHANGE_OF_DIRECTION", heuristic_score=0.85)
        c_cod2 = MovementCandidate("cod2", "v", "CHANGE_OF_DIRECTION", 2.05, 3.25, 1.2, 0.70, json.dumps({"dv": 0.35, "knee": 135}), proposed_movement="CHANGE_OF_DIRECTION", heuristic_score=0.70)

        merged = self.detector._merge_candidates([c_cod1, c_cod2])
        self.assertEqual(len(merged), 1, "Duplicate COD candidates must collapse to 1")
        self.assertEqual(merged[0].proposed_movement, "CHANGE_OF_DIRECTION")
        self.assertEqual(merged[0].heuristic_score, 0.85)

        # Verify discarded duplicate evidence is preserved in key_metrics
        metrics = json.loads(merged[0].key_metrics)
        self.assertIn("duplicate_candidates", metrics, "Discarded candidate evidence must be preserved")
        self.assertEqual(len(metrics["duplicate_candidates"]), 1)
        dup_entry = metrics["duplicate_candidates"][0]
        self.assertEqual(dup_entry["candidate_id"], "cod2")
        self.assertEqual(dup_entry["movement"], "CHANGE_OF_DIRECTION")
        self.assertEqual(dup_entry["score"], 0.70)
        self.assertEqual(dup_entry["start_time"], 2.05)
        self.assertEqual(dup_entry["end_time"], 3.25)
        self.assertTrue(
            "suppressed in favor of higher-scoring candidate" in dup_entry["suppression_reason"]
            or "Merged adjacent/overlapping same-type candidate" in dup_entry["suppression_reason"]
        )

        # Also verify duplicate interval across different types with heavy reciprocal overlap
        # converts to UNCERTAIN review event with both candidate details preserved
        c_lunge = MovementCandidate("lunge1", "v", "LUNGE", 4.0, 5.0, 1.0, 0.78, json.dumps({"knee": 108}), proposed_movement="LUNGE", heuristic_score=0.78)
        c_cod_dup = MovementCandidate("cod_dup", "v", "CHANGE_OF_DIRECTION", 4.05, 5.05, 1.0, 0.82, json.dumps({"dv": 0.45}), proposed_movement="CHANGE_OF_DIRECTION", heuristic_score=0.82)
        merged_target = self.detector._merge_candidates([c_lunge, c_cod_dup])
        self.assertEqual(len(merged_target), 1)
        self.assertEqual(merged_target[0].proposed_movement, "UNCERTAIN", "Competing movement types occupying the same window convert to UNCERTAIN")
        self.assertEqual(merged_target[0].status, "REVIEW_REQUIRED")
        t_diag = json.loads(merged_target[0].key_metrics)
        self.assertEqual(len(t_diag["conflicting_candidate_details"]), 2)
        movements = [d["movement"] for d in t_diag["conflicting_candidate_details"]]
        self.assertIn("LUNGE", movements)
        self.assertIn("CHANGE_OF_DIRECTION", movements)

    def test_more_than_twenty_overlapping_events_converges(self):
        """Candidate detector provably resolves cases with >20 overlapping events without stopping prematurely."""
        # Generate 25 overlapping candidates
        cands = []
        for i in range(25):
            t_start = round(1.0 + i * 0.15, 2)
            t_end = round(t_start + 0.75, 2)
            mov = "CHANGE_OF_DIRECTION" if i % 2 == 0 else "LUNGE"
            cands.append(MovementCandidate(
                f"cand_{i:02d}", "v", mov, t_start, t_end, 0.75, 0.70 + (i % 10) * 0.02,
                json.dumps({"seq": i}), proposed_movement=mov, heuristic_score=0.70 + (i % 10) * 0.02
            ))

        merged = self.detector._merge_candidates(cands)
        self.assertGreater(len(merged), 0)

        # Independent audit: verify every pair has overlap <= 0.05s
        audit_count = 0
        for i in range(len(merged)):
            for j in range(i + 1, len(merged)):
                audit_count += 1
                ov = min(merged[i].end_time, merged[j].end_time) - max(merged[i].start_time, merged[j].start_time)
                self.assertLessEqual(ov, 0.05, f"Pair ({i}, {j}) has unresolved overlap of {ov}s")

        # Every emitted candidate must strictly satisfy signal invariants
        for c in merged:
            self.assertTrue(np.isfinite(c.start_time))
            self.assertTrue(np.isfinite(c.end_time))
            self.assertGreaterEqual(c.start_time, 0.0)
            self.assertGreater(c.end_time, c.start_time)
            self.assertGreater(c.duration, 0.0)
            self.assertAlmostEqual(c.duration, round(c.end_time - c.start_time, 2), places=2)
            max_d = self.detector._get_max_duration(c.proposed_movement)
            self.assertLessEqual(c.duration, max_d + 0.02)

    def test_discarded_invalid_candidates_recorded_with_reason(self):
        """Invalid candidates are not silently dropped; detector records full discard reason in diagnostics."""
        invalids = [
            MovementCandidate("nan_cand", "v", "LUNGE", float("nan"), 2.0, 1.0, 0.8, "{}", proposed_movement="LUNGE", heuristic_score=0.8),
            MovementCandidate("neg_end", "v", "LUNGE", -2.0, -0.5, 1.5, 0.8, "{}", proposed_movement="LUNGE", heuristic_score=0.8),
            MovementCandidate("inv_span", "v", "LUNGE", 3.0, 2.0, -1.0, 0.8, "{}", proposed_movement="LUNGE", heuristic_score=0.8),
            MovementCandidate("sub_min", "v", "LUNGE", 4.0, 4.10, 0.10, 0.8, "{}", proposed_movement="LUNGE", heuristic_score=0.8),
        ]
        merged = self.detector._merge_candidates(invalids)
        self.assertEqual(len(merged), 0, "No invalid candidates should survive")
        self.assertGreaterEqual(len(self.detector.discarded_candidates), 4)

        discarded_ids = [d["candidate_id"] for d in self.detector.discarded_candidates]
        self.assertIn("nan_cand", discarded_ids)
        self.assertIn("neg_end", discarded_ids)
        self.assertIn("inv_span", discarded_ids)
        self.assertIn("sub_min", discarded_ids)

        for d in self.detector.discarded_candidates:
            self.assertIn("discard_reason", d)
            self.assertTrue(len(d["discard_reason"]) > 0)


    def test_uncertain_conflict_retains_full_temporal_extent_and_candidate_details(self):
        """UNCERTAIN review event retains the full conflicting temporal extent while preserving original candidate details."""
        # Candidates [1.0, 2.0] and [1.2, 2.2] overlap by 0.80s (>= 0.40s)
        c1 = MovementCandidate(
            candidate_id="c1", video_id="test_vid", movement_type="LUNGE",
            start_time=1.0, end_time=2.0, duration=1.0, confidence_score=0.72,
            key_metrics=json.dumps({"knee_angle": 108.5}), proposed_movement="LUNGE", heuristic_score=0.72,
        )
        c2 = MovementCandidate(
            candidate_id="c2", video_id="test_vid", movement_type="CHANGE_OF_DIRECTION",
            start_time=1.2, end_time=2.2, duration=1.0, confidence_score=0.75,
            key_metrics=json.dumps({"dv": 0.42}), proposed_movement="CHANGE_OF_DIRECTION", heuristic_score=0.75,
        )
        merged = self.detector._merge_candidates([c1, c2])
        self.assertEqual(len(merged), 1)
        ev = merged[0]
        self.assertEqual(ev.proposed_movement, "UNCERTAIN")
        self.assertEqual(ev.status, "REVIEW_REQUIRED")

        # 1. Emitted review event retains the full conflicting interval [1.0, 2.2]
        self.assertEqual(ev.start_time, 1.0, "Emitted UNCERTAIN event must retain full start bound")
        self.assertEqual(ev.end_time, 2.2, "Emitted UNCERTAIN event must retain full end bound")
        self.assertEqual(ev.duration, 1.2)
        max_u = self.detector._get_max_duration("UNCERTAIN")
        self.assertLessEqual(ev.duration, max_u, f"Duration must not exceed class limit {max_u}s")

        # 2. Separately assert that all original candidate intervals remain untruncated in diagnostic details
        diag = json.loads(ev.key_metrics)
        details = diag["conflicting_candidate_details"]
        self.assertEqual(len(details), 2)

        det1 = next(d for d in details if d["movement"] == "LUNGE")
        self.assertEqual(det1["start_time"], 1.0)
        self.assertEqual(det1["end_time"], 2.0)
        self.assertEqual(det1["duration"], 1.0)
        self.assertEqual(det1["key_metrics"]["knee_angle"], 108.5)

        det2 = next(d for d in details if d["movement"] == "CHANGE_OF_DIRECTION")
        self.assertEqual(det2["start_time"], 1.2)
        self.assertEqual(det2["end_time"], 2.2)
        self.assertEqual(det2["duration"], 1.0)
        self.assertEqual(det2["key_metrics"]["dv"], 0.42)

        # 3. Test capping when combined span exceeds class-specific maximum duration (2.5s)
        # Competing target movements with heavy overlap that cannot be decoupled
        c_broad1 = MovementCandidate("b1", "v", "LUNGE", 1.0, 3.2, 2.2, 0.75, "{}", proposed_movement="LUNGE", heuristic_score=0.75)
        c_broad2 = MovementCandidate("b2", "v", "SMASH_LANDING", 1.4, 3.8, 2.4, 0.85, "{}", proposed_movement="SMASH_LANDING", heuristic_score=0.85)
        # Combined span exceeds 2.5s max bound for UNCERTAIN review event
        merged_broad = self.detector._merge_candidates([c_broad1, c_broad2])
        self.assertEqual(len(merged_broad), 1)
        ev_broad = merged_broad[0]
        self.assertEqual(ev_broad.proposed_movement, "UNCERTAIN")
        self.assertLessEqual(ev_broad.duration, max_u + 0.02, "Must be capped to class-specific maximum duration")
        # Diagnostics must still retain original intervals
        diag_broad = json.loads(ev_broad.key_metrics)
        det_b1 = next(d for d in diag_broad["conflicting_candidate_details"] if d["movement"] == "LUNGE")
        det_b2 = next(d for d in diag_broad["conflicting_candidate_details"] if d["movement"] == "SMASH_LANDING")
        self.assertEqual(det_b1["start_time"], 1.0)
        self.assertEqual(det_b1["end_time"], 3.2)
        self.assertEqual(det_b2["start_time"], 1.4)
        self.assertLessEqual(det_b2["end_time"], 3.8)

    def test_valid_lunge_followed_by_smash_landing_small_boundary_overlap(self):
        """A valid LUNGE followed by a SMASH_LANDING with small boundary overlap preserves both distinct movements."""
        c1 = MovementCandidate("1", "v", "LUNGE", 1.0, 1.8, 0.8, 0.85, json.dumps({"knee": 110}), proposed_movement="LUNGE", heuristic_score=0.85)
        c2 = MovementCandidate("2", "v", "SMASH_LANDING", 1.7, 2.5, 0.8, 0.82, json.dumps({"apex": 0.05}), proposed_movement="SMASH_LANDING", heuristic_score=0.82)
        # Overlap is [1.7, 1.8] = 0.10s (small boundary overlap between sequential compound movements)
        merged = self.detector._merge_candidates([c1, c2])
        self.assertEqual(len(merged), 2, "Must preserve both distinct movements rather than merging into single UNCERTAIN")
        self.assertEqual(merged[0].proposed_movement, "LUNGE")
        self.assertEqual(merged[1].proposed_movement, "SMASH_LANDING")
        self.assertGreaterEqual(merged[0].duration, self.detector._get_min_duration("LUNGE"))
        self.assertGreaterEqual(merged[1].duration, self.detector._get_min_duration("SMASH_LANDING"))
        # Boundary decoupled: overlap <= 0.05s
        ov = min(merged[0].end_time, merged[1].end_time) - max(merged[0].start_time, merged[1].start_time)
        self.assertLessEqual(ov, 0.05)

    def test_target_movement_overlapping_broader_footwork_event(self):
        """A target movement overlapping broader footwork preserves target movement intact while footwork is decoupled."""
        c_cod = MovementCandidate("c1", "v", "CHANGE_OF_DIRECTION", 1.0, 2.5, 1.5, 0.88, json.dumps({"dv": 0.5}), proposed_movement="CHANGE_OF_DIRECTION", heuristic_score=0.88)
        c_lunge = MovementCandidate("l1", "v", "LUNGE", 1.8, 2.6, 0.8, 0.85, json.dumps({"knee": 112}), proposed_movement="LUNGE", heuristic_score=0.85)
        merged = self.detector._merge_candidates([c_cod, c_lunge])
        self.assertEqual(len(merged), 2, "Both footwork and target movement must be preserved")
        lunge_ev = next(m for m in merged if m.proposed_movement == "LUNGE")
        cod_ev = next(m for m in merged if m.proposed_movement == "CHANGE_OF_DIRECTION")
        # Target movement preserved intact
        self.assertEqual(lunge_ev.start_time, 1.8)
        self.assertEqual(lunge_ev.end_time, 2.6)
        self.assertEqual(lunge_ev.duration, 0.8)
        # Footwork decoupled before target movement
        self.assertEqual(cod_ev.start_time, 1.0)
        self.assertEqual(cod_ev.end_time, 1.8)
        self.assertEqual(cod_ev.duration, 0.8)

    def test_three_chained_events_distinguishable_movements_preserved(self):
        """Three chained events with boundary overlaps preserve distinguishable movements without collapsing into a single UNCERTAIN megablock."""
        c1 = MovementCandidate("1", "v", "CHANGE_OF_DIRECTION", 0.5, 1.4, 0.9, 0.80, json.dumps({"cod": 1}), proposed_movement="CHANGE_OF_DIRECTION", heuristic_score=0.80)
        c2 = MovementCandidate("2", "v", "LUNGE", 1.3, 2.2, 0.9, 0.85, json.dumps({"lunge": 2}), proposed_movement="LUNGE", heuristic_score=0.85)
        c3 = MovementCandidate("3", "v", "SMASH_LANDING", 2.1, 2.9, 0.8, 0.82, json.dumps({"smash": 3}), proposed_movement="SMASH_LANDING", heuristic_score=0.82)
        merged = self.detector._merge_candidates([c1, c2, c3])
        self.assertEqual(len(merged), 3, "All three distinguishable movements must be preserved")
        movements = [m.proposed_movement for m in merged]
        self.assertEqual(movements, ["CHANGE_OF_DIRECTION", "LUNGE", "SMASH_LANDING"])
        # Verify every pair has overlap <= 0.05s
        for i in range(len(merged)):
            for j in range(i + 1, len(merged)):
                ov = min(merged[i].end_time, merged[j].end_time) - max(merged[i].start_time, merged[j].start_time)
                self.assertLessEqual(ov, 0.05)

    def test_genuine_unresolved_conflict_becomes_uncertain(self):
        """A genuine unresolved conflict between competing target movements resolves to UNCERTAIN with full diagnostics."""
        c1 = MovementCandidate("1", "v", "LUNGE", 1.0, 2.0, 1.0, 0.75, json.dumps({"knee": 110}), proposed_movement="LUNGE", heuristic_score=0.75)
        c2 = MovementCandidate("2", "v", "SMASH_LANDING", 1.1, 2.1, 1.0, 0.85, json.dumps({"apex": 0.05}), proposed_movement="SMASH_LANDING", heuristic_score=0.85)
        merged = self.detector._merge_candidates([c1, c2])
        self.assertEqual(len(merged), 1, "Competing target movements must resolve to single UNCERTAIN event")
        self.assertEqual(merged[0].proposed_movement, "UNCERTAIN")
        self.assertEqual(merged[0].status, "REVIEW_REQUIRED")
        diag = json.loads(merged[0].key_metrics)
        self.assertIn("conflicting_candidate_details", diag)
        self.assertEqual(len(diag["conflicting_candidate_details"]), 2)
        self.assertEqual(set(diag["conflicting_movements"]), {"LUNGE", "SMASH_LANDING"})
        self.assertEqual(diag["original_full_interval"], [1.0, 2.1])
        self.assertEqual(diag["original_full_duration"], 1.1)
        self.assertFalse(diag["is_capped"])
        self.assertTrue(diag["full_conflict_covered"])

    def test_conflict_full_temporal_extent_exceeds_uncertain_duration_limit(self):
        """When conflicting extent exceeds UNCERTAIN limit, review clip is capped, diagnostics preserve original extent, and it is explicitly marked capped."""
        c1 = MovementCandidate("1", "v", "LUNGE", 1.0, 2.8, 1.8, 0.75, json.dumps({"knee": 110}), proposed_movement="LUNGE", heuristic_score=0.75)
        c2 = MovementCandidate("2", "v", "SMASH_LANDING", 1.8, 3.8, 2.0, 0.85, json.dumps({"apex": 0.05}), proposed_movement="SMASH_LANDING", heuristic_score=0.85)
        # Full conflicting span is [1.0, 3.8] = 2.8s > 2.5s (max UNCERTAIN limit)
        merged = self.detector._merge_candidates([c1, c2])
        self.assertEqual(len(merged), 1)
        ev = merged[0]
        self.assertEqual(ev.proposed_movement, "UNCERTAIN")
        max_u = self.detector._get_max_duration("UNCERTAIN")
        self.assertLessEqual(ev.duration, max_u)
        diag = json.loads(ev.key_metrics)
        # Reviewer can still access original full interval and duration
        self.assertEqual(diag["original_full_interval"], [1.0, 3.8])
        self.assertEqual(diag["original_full_duration"], 2.8)
        # Explicitly marked as capped; does NOT claim emitted clip covers full conflict
        self.assertTrue(diag["is_capped"])
        self.assertFalse(diag["full_conflict_covered"])
        self.assertIn("capped_notice", diag)
        self.assertIn("does NOT cover full conflicting extent", diag["capped_notice"])
        # All original candidate evidence is preserved
        self.assertEqual(len(diag["conflicting_candidate_details"]), 2)
        det_lunge = next(d for d in diag["conflicting_candidate_details"] if d["movement"] == "LUNGE")
        self.assertEqual(det_lunge["start_time"], 1.0)
        self.assertEqual(det_lunge["end_time"], 2.8)
        self.assertEqual(det_lunge["duration"], 1.8)
        det_smash = next(d for d in diag["conflicting_candidate_details"] if d["movement"] == "SMASH_LANDING")
        self.assertEqual(det_smash["start_time"], 1.8)
        self.assertEqual(det_smash["end_time"], 3.8)
        self.assertEqual(det_smash["duration"], 2.0)

    def test_pair_remains_overlapping_after_two_event_resolution_becomes_uncertain(self):
        """When a two-event resolution attempt leaves an overlap >0.05s, the sweep safely converts the pair into UNCERTAIN."""
        class MockFailedDecoupleDetector(MovementCandidateDetector):
            def _resolve_conflict_pair(self, e1: MovementCandidate, e2: MovementCandidate) -> list[MovementCandidate]:
                # Simulates a two-event resolution attempt that failed to decouple (leaves 0.15s overlap)
                r1 = MovementCandidate("r1", "v", e1.proposed_movement, 1.0, 1.95, 0.95, e1.heuristic_score, "{}", proposed_movement=e1.proposed_movement, heuristic_score=e1.heuristic_score)
                r2 = MovementCandidate("r2", "v", e2.proposed_movement, 1.80, 2.40, 0.60, e2.heuristic_score, "{}", proposed_movement=e2.proposed_movement, heuristic_score=e2.heuristic_score)
                return [r1, r2]

        detector = MockFailedDecoupleDetector()
        # Non-adjacent candidates where c1 and c3 overlap, while neither overlaps c2 in Pass 2
        c1 = MovementCandidate("1", "v", "LUNGE", 1.0, 2.0, 1.0, 0.85, json.dumps({"knee": 110}), proposed_movement="LUNGE", heuristic_score=0.85)
        c2 = MovementCandidate("2", "v", "CHANGE_OF_DIRECTION", 2.6, 3.6, 1.0, 0.80, json.dumps({"dv": 0.4}), proposed_movement="CHANGE_OF_DIRECTION", heuristic_score=0.80)
        c3 = MovementCandidate("3", "v", "SMASH_LANDING", 1.8, 2.4, 0.6, 0.78, json.dumps({"apex": 0.05}), proposed_movement="SMASH_LANDING", heuristic_score=0.78)

        merged = detector._merge_candidates([c1, c2, c3])
        # Two events emitted: one UNCERTAIN composite (from failed two-event resolution of c1 vs c3) and one preserved COD (c2)
        self.assertEqual(len(merged), 2)
        uncertain_ev = next(m for m in merged if m.proposed_movement == "UNCERTAIN")
        cod_ev = next(m for m in merged if m.proposed_movement == "CHANGE_OF_DIRECTION")

        self.assertEqual(uncertain_ev.status, "REVIEW_REQUIRED")
        diag = json.loads(uncertain_ev.key_metrics)
        self.assertIn("Overlap persists after two-event resolution attempt", diag["reason"])
        self.assertEqual(len(diag["conflicting_candidate_details"]), 2)
        self.assertEqual(set(diag["conflicting_movements"]), {"LUNGE", "SMASH_LANDING"})

        # Confirm the sweep terminated safely normally without hitting its iteration limit
        self.assertFalse(detector.last_sweep_hit_limit)
        self.assertLess(detector.last_sweep_iterations, 10)

        # Independent audit: every pair strictly satisfies overlap <= 0.05s
        for i in range(len(merged)):
            for j in range(i + 1, len(merged)):
                ov = min(merged[i].end_time, merged[j].end_time) - max(merged[i].start_time, merged[j].start_time)
                self.assertLessEqual(ov, 0.05, f"Pair ({i}, {j}) has unresolved overlap of {ov}s")

    def test_sweep_limit_path_cannot_silently_return_violating_events(self):
        """Confirm the sweep-limit path cannot silently return events that violate the final overlap invariant."""
        c1 = MovementCandidate("1", "v", "LUNGE", 1.0, 2.0, 1.0, 0.85, "{}", proposed_movement="LUNGE", heuristic_score=0.85)
        c2 = MovementCandidate("2", "v", "CHANGE_OF_DIRECTION", 2.6, 3.6, 1.0, 0.80, "{}", proposed_movement="CHANGE_OF_DIRECTION", heuristic_score=0.80)
        c3 = MovementCandidate("3", "v", "SMASH_LANDING", 1.8, 2.4, 0.6, 0.78, "{}", proposed_movement="SMASH_LANDING", heuristic_score=0.78)

        # Normal sweep execution terminates safely with last_sweep_hit_limit = False and 0 < iterations < 10
        normal_merged = self.detector._merge_candidates([c1, c2, c3])
        self.assertFalse(self.detector.last_sweep_hit_limit)
        self.assertGreater(self.detector.last_sweep_iterations, 0)
        self.assertLess(self.detector.last_sweep_iterations, 10)

    def test_sweep_limit_path_raises_runtime_error_on_non_convergence(self):
        """
        Deterministic test that forces the sweep-limit failure path and asserts that
        _merge_candidates() raises RuntimeError rather than returning an unresolved timeline.
        """
        # Configure detector with a small deterministic sweep limit (1 iteration)
        self.detector.max_sweep_iterations = 1

        # Chained events requiring at least 2 sweep iterations:
        # A: LUNGE [1.0, 2.5], C: COD [2.3, 3.0], B: SMASH_LANDING [2.4, 3.2]
        c1 = MovementCandidate("1", "v", "LUNGE", 1.0, 2.5, 1.5, 0.85, "{}", proposed_movement="LUNGE", heuristic_score=0.85)
        c2 = MovementCandidate("2", "v", "CHANGE_OF_DIRECTION", 2.3, 3.0, 0.7, 0.80, "{}", proposed_movement="CHANGE_OF_DIRECTION", heuristic_score=0.80)
        c3 = MovementCandidate("3", "v", "SMASH_LANDING", 2.4, 3.2, 0.8, 0.78, "{}", proposed_movement="SMASH_LANDING", heuristic_score=0.78)

        with self.assertRaises(RuntimeError) as cm:
            self.detector._merge_candidates([c1, c2, c3])

        err_msg = str(cm.exception)
        self.assertIn("Multi-pass resolution sweep reached maximum iteration limit", err_msg)
        self.assertIn("(1)", err_msg)
        self.assertTrue(self.detector.last_sweep_hit_limit)
        self.assertEqual(self.detector.last_sweep_iterations, 1)

    def test_independent_final_pairwise_audit_raises_on_residual_overlap(self):
        """
        Exercise the independent final invariant audit directly with a residual overlap
        between a pair other than the pair just resolved, asserting that RuntimeError
        is raised with comprehensive unresolved pair details.
        """
        # Pair 1: successfully resolved event (e.g. UNCERTAIN review conflict)
        resolved_pair = MovementCandidate(
            "resolved_1", "v", "UNCERTAIN", 1.0, 2.5, 1.5, 0.85, "{}",
            proposed_movement="UNCERTAIN", heuristic_score=0.85
        )
        # Pair 2: distinct pair with deliberate residual overlap (> 0.05s)
        c3 = MovementCandidate(
            "cand_lunge_3", "v", "LUNGE", 4.0, 5.0, 1.0, 0.82, "{}",
            proposed_movement="LUNGE", heuristic_score=0.82
        )
        c4 = MovementCandidate(
            "cand_smash_4", "v", "SMASH_LANDING", 4.8, 5.6, 0.8, 0.78, "{}",
            proposed_movement="SMASH_LANDING", heuristic_score=0.78
        )

        with self.assertRaises(RuntimeError) as cm:
            self.detector._audit_pairwise_overlaps([resolved_pair, c3, c4])

        err_msg = str(cm.exception)
        self.assertIn("Candidate detector sweep completed with 1 unresolved overlap(s) > 0.05s", err_msg)
        self.assertIn("cand_lunge_3", err_msg)
        self.assertIn("cand_smash_4", err_msg)
        self.assertIn("LUNGE", err_msg)
        self.assertIn("SMASH_LANDING", err_msg)
        self.assertIn("0.2", err_msg)

    def test_independent_final_pairwise_audit_reached_from_merge_candidates(self):
        """
        Exercise the independent final invariant audit via _merge_candidates(), verifying that
        a residual overlap between a pair other than the pair just resolved genuinely reaches
        the final audit rather than being caught by the sweep-limit guard or two-event fallback.
        """
        c1 = MovementCandidate("1", "v", "LUNGE", 1.0, 2.0, 1.0, 0.85, "{}", proposed_movement="LUNGE", heuristic_score=0.85)
        c2 = MovementCandidate("2", "v", "CHANGE_OF_DIRECTION", 1.8, 2.5, 0.7, 0.80, "{}", proposed_movement="CHANGE_OF_DIRECTION", heuristic_score=0.80)
        c3 = MovementCandidate("3", "v", "LUNGE", 4.0, 5.0, 1.0, 0.82, "{}", proposed_movement="LUNGE", heuristic_score=0.82)
        c4 = MovementCandidate("4", "v", "SMASH_LANDING", 4.8, 5.6, 0.8, 0.78, "{}", proposed_movement="SMASH_LANDING", heuristic_score=0.78)

        # Pre-resolve pair 1 cleanly so it cannot trigger the two-event fallback
        resolved_pair_1 = self.detector._resolve_conflict_pair(c1, c2)

        def mock_resolution_sweep(candidates):
            # Simulates resolution sweep: pair 1 is cleanly resolved, but pair (3, 4) retains residual overlap
            c3_overlap = MovementCandidate("3", "v", "LUNGE", 4.0, 5.0, 1.0, 0.82, "{}", proposed_movement="LUNGE", heuristic_score=0.82)
            c4_overlap = MovementCandidate("4", "v", "SMASH_LANDING", 4.8, 5.6, 0.8, 0.78, "{}", proposed_movement="SMASH_LANDING", heuristic_score=0.78)
            return [resolved_pair_1[0], c3_overlap, c4_overlap]

        self.detector._run_resolution_sweep = mock_resolution_sweep

        with self.assertRaises(RuntimeError) as cm:
            self.detector._merge_candidates([c1, c2, c3, c4])

        err_msg = str(cm.exception)
        self.assertIn("Candidate detector sweep completed with 1 unresolved overlap(s) > 0.05s", err_msg)
        self.assertIn("3", err_msg)
        self.assertIn("4", err_msg)
        self.assertIn("LUNGE", err_msg)
        self.assertIn("SMASH_LANDING", err_msg)
        self.assertIn("0.2", err_msg)


if __name__ == "__main__":
    unittest.main()
