"""Regression coverage for Rubric v0.3 risk assessment gates and outputs."""

import math
import unittest

from src.risk.evaluator import evaluate_movement_risk
from src.detection.manifest import MovementEventRecord
from src.risk.rubric import (
    ACTIVE_RUBRIC_CATEGORIES,
    DEFERRED_RUBRIC_CATEGORIES,
    VALID_RISK_STATUSES,
)


VALID_FEATURES = {
    "left_knee_min": 102.5,
    "right_knee_min": 110.8,
    "trunk_angle_max": 36.4,
}


class RiskEvaluatorTests(unittest.TestCase):
    def test_each_active_movement_has_only_provisional_observations(self):
        for category in ACTIVE_RUBRIC_CATEGORIES:
            with self.subTest(category=category):
                result = evaluate_movement_risk(category, VALID_FEATURES, landmark_quality=0.9)
                self.assertEqual(result.risk_level, "PROVISIONAL")
                self.assertIn(result.risk_level, VALID_RISK_STATUSES)
                self.assertTrue(result.is_assessable)
                self.assertTrue(result.observations)
                self.assertTrue(result.missing_evidence)
                self.assertIn("clinical validation", result.missing_evidence.lower())
                self.assertTrue(any("LOW/MODERATE/HIGH" in line for line in result.observations))

    def test_deferred_and_nonactive_movements_are_unassessed(self):
        for category in [*DEFERRED_RUBRIC_CATEGORIES, "UNCERTAIN", "NO_EVENT", "UNKNOWN"]:
            with self.subTest(category=category):
                result = evaluate_movement_risk(category, VALID_FEATURES, landmark_quality=0.9)
                self.assertEqual(result.risk_level, "UNASSESSED")
                self.assertIn(result.risk_level, VALID_RISK_STATUSES)
                self.assertFalse(result.is_assessable)
                self.assertTrue(result.missing_evidence)

    def test_empty_or_each_missing_required_measurement_is_unassessed(self):
        required = ("left_knee_min", "right_knee_min", "trunk_angle_max")
        cases = [{}, *({**VALID_FEATURES, key: None} for key in required)]
        for features in cases:
            with self.subTest(features=features):
                result = evaluate_movement_risk("LUNGE", features, landmark_quality=0.95)
                self.assertEqual(result.risk_level, "UNASSESSED")
                self.assertFalse(result.is_assessable)
                self.assertTrue(result.missing_evidence)

    def test_low_invalid_and_nonfinite_quality_are_unassessed(self):
        for quality in (None, float("nan"), float("inf"), -0.01, 0.449, 1.001):
            with self.subTest(quality=quality):
                result = evaluate_movement_risk("LUNGE", VALID_FEATURES, landmark_quality=quality)
                self.assertEqual(result.risk_level, "UNASSESSED")
                self.assertFalse(result.is_assessable)
                self.assertTrue(result.missing_evidence)

    def test_quality_boundary_is_inclusive_and_upper_boundary_valid(self):
        for quality in (0.45, 1.0):
            with self.subTest(quality=quality):
                result = evaluate_movement_risk("LUNGE", VALID_FEATURES, landmark_quality=quality)
                self.assertEqual(result.risk_level, "PROVISIONAL")
                self.assertTrue(result.is_assessable)

    def test_angle_boundaries_are_valid_and_out_of_range_values_are_unassessed(self):
        for angle in (0.0, 180.0):
            with self.subTest(valid_boundary=angle):
                features = {**VALID_FEATURES, "left_knee_min": angle}
                result = evaluate_movement_risk("LUNGE", features, landmark_quality=0.9)
                self.assertEqual(result.risk_level, "PROVISIONAL")

        for angle in (-0.01, 180.01, float("nan"), float("inf")):
            with self.subTest(invalid_angle=angle):
                features = {**VALID_FEATURES, "left_knee_min": angle}
                result = evaluate_movement_risk("LUNGE", features, landmark_quality=0.9)
                self.assertEqual(result.risk_level, "UNASSESSED")
                self.assertFalse(result.is_assessable)

    def test_missing_evidence_round_trips_through_event_manifest_record(self):
        record = MovementEventRecord(
            event_id="risk_test_001",
            video_id="sample.mp4",
            source_video="/tmp/sample.mp4",
            start_time=0.0,
            end_time=1.0,
            start_frame=0,
            end_frame=30,
            movement_category="LUNGE",
            risk_missing_evidence="Clinical validation is unavailable.",
        )
        restored = MovementEventRecord.from_dict(record.to_dict())
        self.assertEqual(restored.risk_missing_evidence, "Clinical validation is unavailable.")
        legacy = MovementEventRecord.from_dict({
            "event_id": "legacy",
            "video_id": "sample.mp4",
            "source_video": "/tmp/sample.mp4",
            "start_time": 0.0,
            "end_time": 1.0,
            "start_frame": 0,
            "end_frame": 30,
            "movement_category": "LUNGE",
        })
        self.assertEqual(legacy.risk_missing_evidence, "")

    def test_lower_body_quality_gate_is_used_when_provided(self):
        result = evaluate_movement_risk(
            "LUNGE", VALID_FEATURES, landmark_quality=0.99, lower_body_quality=0.44
        )
        self.assertEqual(result.risk_level, "UNASSESSED")
        self.assertIn("minimum 0.45", result.observations[0])



    def test_score_range_and_determinism(self):
        """Scores must strictly lie in [0.0, 100.0] and be deterministic."""
        for cat in ACTIVE_RUBRIC_CATEGORIES:
            res1 = evaluate_movement_risk(cat, VALID_FEATURES, landmark_quality=0.9)
            res2 = evaluate_movement_risk(cat, VALID_FEATURES, landmark_quality=0.9)
            self.assertIsNotNone(res1.score)
            self.assertTrue(0.0 <= res1.score <= 100.0)
            self.assertEqual(res1.score, res2.score)
            self.assertEqual(res1.screening_category, res2.screening_category)

    def test_score_monotonicity_with_stiff_knee(self):
        """As knee deceleration becomes stiffer (>115 deg), risk score must strictly increase."""
        stiff_angles = [115.0, 125.0, 140.0, 160.0]
        scores = []
        for angle in stiff_angles:
            feats = {"left_knee_min": angle, "right_knee_min": angle, "trunk_angle_max": 15.0}
            res = evaluate_movement_risk("SMASH_LANDING", feats, landmark_quality=0.9)
            scores.append(res.score)
        for i in range(len(scores) - 1):
            self.assertLess(scores[i], scores[i + 1])

    def test_score_monotonicity_with_acute_knee(self):
        """As knee flexion becomes acutely deep (<95 deg), risk score must strictly increase."""
        acute_angles = [95.0, 85.0, 70.0, 55.0]
        scores = []
        for angle in acute_angles:
            feats = {"left_knee_min": angle, "right_knee_min": angle, "trunk_angle_max": 15.0}
            res = evaluate_movement_risk("LUNGE", feats, landmark_quality=0.9)
            scores.append(res.score)
        for i in range(len(scores) - 1):
            self.assertLess(scores[i], scores[i + 1])

    def test_score_monotonicity_with_torso_lean(self):
        """As torso deviation from vertical increases (>15 deg), risk score must strictly increase."""
        trunk_angles = [15.0, 25.0, 35.0, 45.0]
        scores = []
        for trunk in trunk_angles:
            feats = {"left_knee_min": 105.0, "right_knee_min": 105.0, "trunk_angle_max": trunk}
            res = evaluate_movement_risk("CHANGE_OF_DIRECTION", feats, landmark_quality=0.9)
            scores.append(res.score)
        for i in range(len(scores) - 1):
            self.assertLess(scores[i], scores[i + 1])

    def test_score_monotonicity_with_bilateral_asymmetry(self):
        """As bilateral knee flexion asymmetry increases (>10 deg), risk score must strictly increase."""
        asym_differences = [10.0, 20.0, 30.0, 45.0]
        scores = []
        for diff in asym_differences:
            feats = {"left_knee_min": 105.0, "right_knee_min": 105.0 + diff, "trunk_angle_max": 15.0}
            res = evaluate_movement_risk("SMASH_LANDING", feats, landmark_quality=0.9)
            scores.append(res.score)
        for i in range(len(scores) - 1):
            self.assertLess(scores[i], scores[i + 1])

    def test_category_boundaries_and_mappings(self):
        """Scores in [0, 30) -> LOW, [30, 60) -> MODERATE, [60, 100] -> HIGH."""
        # LOW case (all optimal): score 0.0
        low_feats = {"left_knee_min": 105.0, "right_knee_min": 105.0, "trunk_angle_max": 12.0}
        res_low = evaluate_movement_risk("LUNGE", low_feats, landmark_quality=0.9)
        self.assertEqual(res_low.score, 0.0)
        self.assertEqual(res_low.screening_category, "LOW")
        self.assertEqual(res_low.risk_color, "#28a745")

        # MODERATE case: e.g. moderate trunk lean (trunk=35 deg -> 23.3 pts) + mild asymmetry (25 deg -> 10.7 pts) -> 34.0 pts
        mod_feats = {"left_knee_min": 105.0, "right_knee_min": 130.0, "trunk_angle_max": 35.0}
        res_mod = evaluate_movement_risk("SMASH_LANDING", mod_feats, landmark_quality=0.9)
        self.assertGreaterEqual(res_mod.score, 30.0)
        self.assertLess(res_mod.score, 60.0)
        self.assertEqual(res_mod.screening_category, "MODERATE")
        self.assertEqual(res_mod.risk_color, "#ffc107")

        # HIGH case: stiff knee (155 deg -> 35.6 pts) + severe trunk lean (40 deg -> 29.2 pts) -> 64.8 pts
        high_feats = {"left_knee_min": 155.0, "right_knee_min": 155.0, "trunk_angle_max": 40.0}
        res_high = evaluate_movement_risk("CHANGE_OF_DIRECTION", high_feats, landmark_quality=0.9)
        self.assertGreaterEqual(res_high.score, 60.0)
        self.assertEqual(res_high.screening_category, "HIGH")
        self.assertEqual(res_high.risk_color, "#dc3545")

    def test_unassessed_cases_have_none_score_and_unassessed_category(self):
        """Any unassessed movement must have score=None and screening_category='UNASSESSED'."""
        unassessed_cases = [
            ("LUNGE", {}, 0.9),  # empty features
            ("LUNGE", {"left_knee_min": float("nan"), "right_knee_min": 100.0, "trunk_angle_max": 20.0}, 0.9),
            ("LUNGE", VALID_FEATURES, 0.35),  # low quality
            ("SCISSOR_JUMP", VALID_FEATURES, 0.9),  # deferred
            ("UNCERTAIN", VALID_FEATURES, 0.9),  # uncertain
        ]
        for cat, feats, q in unassessed_cases:
            with self.subTest(cat=cat, q=q):
                res = evaluate_movement_risk(cat, feats, landmark_quality=q)
                self.assertIsNone(res.score)
                self.assertEqual(res.screening_category, "UNASSESSED")
                self.assertEqual(res.risk_level, "UNASSESSED")
                self.assertFalse(res.is_assessable)

    def test_contributing_indicators_structure_and_sum(self):
        """Contributing indicators must provide structured breakdowns whose points sum to total score."""
        res = evaluate_movement_risk("LUNGE", VALID_FEATURES, landmark_quality=0.9)
        self.assertEqual(len(res.contributing_indicators), 3)
        expected_names = {"Knee Deceleration Flexion", "Torso Lean Deviation", "Bilateral Knee Asymmetry"}
        actual_names = {c["name"] for c in res.contributing_indicators}
        self.assertEqual(expected_names, actual_names)

        total_pts = round(sum(c["points"] for c in res.contributing_indicators), 1)
        self.assertEqual(total_pts, res.score)

    def test_manifest_record_risk_score_serialization(self):
        """MovementEventRecord serializes and restores risk_score, risk_category, and risk_contributions."""
        record = MovementEventRecord(
            event_id="ev_score_test",
            video_id="test.mp4",
            source_video="/tmp/test.mp4",
            start_time=0.5,
            end_time=1.5,
            start_frame=15,
            end_frame=45,
            movement_category="LUNGE",
            risk_level="PROVISIONAL",
            risk_score=24.5,
            risk_category="LOW",
            risk_contributions=[{"name": "Knee", "points": 10.0}],
        )
        d = record.to_dict()
        self.assertEqual(d["risk_score"], 24.5)
        self.assertEqual(d["risk_category"], "LOW")
        self.assertEqual(len(d["risk_contributions"]), 1)

        restored = MovementEventRecord.from_dict(d)
        self.assertEqual(restored.risk_score, 24.5)
        self.assertEqual(restored.risk_category, "LOW")
        self.assertEqual(restored.risk_contributions[0]["name"], "Knee")


    def test_retired_classes_cannot_enter_active_risk_scoring_pathway(self):
        """Proves retired classes (SCISSOR_JUMP, FOOTWORK_SPLIT_STEP, NO_EVENT) cannot enter active risk scoring."""
        retired_and_nonactive = ["SCISSOR_JUMP", "FOOTWORK_SPLIT_STEP", "NO_EVENT", "scissor_jump", "footwork_split_step"]
        perfect_kinematics = {
            "left_knee_min": 105.0,
            "right_knee_min": 105.0,
            "trunk_angle_max": 10.0,
        }
        for category in retired_and_nonactive:
            with self.subTest(category=category):
                res = evaluate_movement_risk(category, perfect_kinematics, landmark_quality=0.99)
                self.assertFalse(res.is_assessable)
                self.assertEqual(res.risk_level, "UNASSESSED")
                self.assertIsNone(res.score)
                self.assertEqual(res.screening_category, "UNASSESSED")
                self.assertNotEqual(res.screening_category, "LOW")
                self.assertNotEqual(res.screening_category, "MODERATE")
                self.assertNotEqual(res.screening_category, "HIGH")
                self.assertEqual(len(res.contributing_indicators), 0)


if __name__ == "__main__":
    unittest.main()
