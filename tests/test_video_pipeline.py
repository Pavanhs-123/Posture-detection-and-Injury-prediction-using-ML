"""
Unit and integration tests for video pipeline, validation, clip extraction,
5-category classification schema, Rubric v0.3 risk evaluation, and manifest generation.
"""

import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from collections import namedtuple
from pathlib import Path
from unittest.mock import patch, MagicMock
import cv2
import numpy as np

from src.video.validator import validate_video
from src.video.clip_extractor import extract_subclip, extract_keyframe
from src.video.video_pipeline import (
    VideoProcessingPipeline,
    _aggregate_pose_signals,
    _confidence_filtered_measurement,
    _filter_candidates_by_pose_coverage,
)
from src.classification.categories import (
    MOVEMENT_CATEGORIES,
    ACTIVE_RISK_RUBRIC_CATEGORIES,
    DEFERRED_RISK_RUBRIC_CATEGORIES,
)
from src.classification.predictor import MovementClassifier
from src.classification.predictor import EXPECTED_FEATURE_NAMES
from src.risk.evaluator import evaluate_movement_risk
from src.detection.manifest import (
    MovementEventRecord,
    save_event_manifest,
    load_event_manifest,
)


class TestVideoPipeline(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

        # Create a valid synthetic dummy video
        self.dummy_video_path = self.temp_path / "valid_test_video.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(self.dummy_video_path), fourcc, 15.0, (320, 240))
        for _ in range(30):  # 2.0 seconds
            frame = np.zeros((240, 320, 3), dtype=np.uint8)
            writer.write(frame)
        writer.release()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_video_validation_valid(self):
        res = validate_video(str(self.dummy_video_path))
        self.assertTrue(res["is_valid"])
        self.assertIsNone(res["error"])
        self.assertEqual(res["width"], 320)
        self.assertEqual(res["height"], 240)
        self.assertAlmostEqual(res["duration"], 2.0, delta=0.2)

    def test_video_validation_missing_file(self):
        res = validate_video(str(self.temp_path / "non_existent.mp4"))
        self.assertFalse(res["is_valid"])
        self.assertIn("File not found", res["error"])

    def test_video_validation_empty_file(self):
        empty_path = self.temp_path / "empty.mp4"
        empty_path.touch()
        res = validate_video(str(empty_path))
        self.assertFalse(res["is_valid"])
        self.assertIn("empty", res["error"].lower())

    def test_video_validation_unsupported_extension(self):
        bad_ext = self.temp_path / "video.txt"
        bad_ext.write_text("not a video")
        res = validate_video(str(bad_ext))
        self.assertFalse(res["is_valid"])
        self.assertIn("Unsupported", res["error"])

    def test_clip_extraction_and_keyframe(self):
        clip_out = self.temp_path / "extracted_clip.mp4"
        ok_clip = extract_subclip(str(self.dummy_video_path), 0.2, 1.2, str(clip_out))
        self.assertTrue(ok_clip)
        self.assertTrue(clip_out.exists())
        self.assertGreater(clip_out.stat().st_size, 0)

        key_out = self.temp_path / "keyframe.jpg"
        ok_key = extract_keyframe(str(self.dummy_video_path), 0.5, str(key_out))
        self.assertTrue(ok_key)
        self.assertTrue(key_out.exists())
        self.assertGreater(key_out.stat().st_size, 0)

    def test_five_movement_categories_schema(self):
        expected_five = [
            "LUNGE",
            "SMASH_LANDING",
            "SCISSOR_JUMP",
            "CHANGE_OF_DIRECTION",
            "FOOTWORK_SPLIT_STEP",
        ]
        self.assertEqual(MOVEMENT_CATEGORIES, expected_five)
        self.assertEqual(len(MOVEMENT_CATEGORIES), 5)

    def test_risk_rubric_v03_active_and_deferred_distinction(self):
        # Active categories under Rubric v0.3 evaluate to PROVISIONAL with factual observations
        active_sample_features = {
            "left_knee_min": 100.0,
            "right_knee_min": 105.0,
            "trunk_angle_max": 15.0,
        }
        for cat in ACTIVE_RISK_RUBRIC_CATEGORIES:
            res = evaluate_movement_risk(cat, active_sample_features, landmark_quality=0.9)
            self.assertTrue(res.is_assessable)
            self.assertEqual(res.risk_level, "PROVISIONAL")
            self.assertIn("Unvalidated Cutoffs", res.status_message)
            self.assertTrue(len(res.observations) > 0)

        lunge_observations = " ".join(evaluate_movement_risk("LUNGE", active_sample_features, 0.9).observations)
        self.assertIn("Lead leg is not identified", lunge_observations)
        self.assertIn("minima may occur at different frames", lunge_observations)
        smash_observations = " ".join(evaluate_movement_risk("SMASH_LANDING", active_sample_features, 0.9).observations)
        self.assertIn("touchdown timing is not isolated", smash_observations)

        # Deferred categories MUST NOT be evaluated as PROVISIONAL or assigned risk
        for cat in DEFERRED_RISK_RUBRIC_CATEGORIES:
            res = evaluate_movement_risk(cat, active_sample_features, landmark_quality=0.9)
            self.assertFalse(res.is_assessable)
            self.assertEqual(res.risk_level, "UNASSESSED")
            self.assertIn("Deferred", res.status_message)

    def test_risk_evaluation_low_landmark_quality(self):
        feats = {"left_knee_min": 100.0, "right_knee_min": 100.0, "trunk_angle_max": 10.0}
        res = evaluate_movement_risk("LUNGE", feats, landmark_quality=0.3)
        self.assertFalse(res.is_assessable)
        self.assertEqual(res.risk_level, "UNCERTAIN")

    def test_risk_evaluation_missing_measurements_gate(self):
        # Missing measurements must evaluate to UNCERTAIN, never default to LOW
        res = evaluate_movement_risk("LUNGE", {}, landmark_quality=0.9)
        self.assertFalse(res.is_assessable)
        self.assertEqual(res.risk_level, "UNCERTAIN")

    def test_risk_evaluation_nonfinite_measurements_gate(self):
        feats = {"left_knee_min": 100.0, "right_knee_min": float("nan"), "trunk_angle_max": 10.0}
        res = evaluate_movement_risk("LUNGE", feats, landmark_quality=0.9)
        self.assertEqual(res.risk_level, "UNCERTAIN")
        self.assertFalse(res.is_assessable)

    def test_pipeline_defaults_deferred_categories_off(self):
        import inspect
        parameter = inspect.signature(VideoProcessingPipeline.process_video).parameters["include_deferred_categories"]
        self.assertFalse(parameter.default)

    def test_event_features_ignore_no_pose_placeholders(self):
        features = _aggregate_pose_signals([
            {"has_pose": True, "feature_knee_l": 105.0, "feature_knee_r": 110.0, "feature_elbow_l": 140.0, "feature_elbow_r": 145.0, "feature_torso_lean": 22.0},
            {"has_pose": False, "feature_knee_l": 180.0, "feature_knee_r": 180.0, "feature_elbow_l": 180.0, "feature_elbow_r": 180.0, "feature_torso_lean": 0.0},
        ])
        self.assertEqual(features["left_knee_min"], 105.0)
        self.assertEqual(features["left_knee_max"], 105.0)
        self.assertEqual(features["trunk_angle_min"], 22.0)
        self.assertEqual(features["trunk_angle_max"], 22.0)

        missing = _aggregate_pose_signals([{"has_pose": False, "feature_knee_l": 180.0, "feature_torso_lean": 0.0}])
        self.assertIsNone(missing["left_knee_min"])
        self.assertIsNone(missing["trunk_angle_max"])

    def test_low_confidence_measurements_are_unavailable(self):
        self.assertIsNone(_confidence_filtered_measurement(100.0, 0.2))
        self.assertIsNone(_confidence_filtered_measurement(float("nan"), 0.9))
        self.assertEqual(_confidence_filtered_measurement(100.0, 0.9), 100.0)

    def test_candidate_windows_without_pose_support_are_rejected(self):
        candidates = [
            SimpleNamespace(start_time=0.0, end_time=0.8),
            SimpleNamespace(start_time=0.8, end_time=1.0),
        ]
        signals = [{"has_pose": i < 8} for i in range(10)]
        retained = _filter_candidates_by_pose_coverage(candidates, signals, fps=10)
        self.assertEqual(retained, [candidates[0]])

    def test_trained_classifier_rejects_missing_features_and_deferred_outputs(self):
        classifier = MovementClassifier()
        classifier.is_trained_model = True
        classifier.feature_names = list(EXPECTED_FEATURE_NAMES)
        classifier.model = MagicMock()
        classifier.model.predict.return_value = np.array(["SCISSOR_JUMP"])
        classifier.model.predict_proba.return_value = np.array([[1.0]])
        classifier.model.classes_ = np.array(["SCISSOR_JUMP"])

        incomplete = classifier.predict({"left_knee_min": 100.0})
        self.assertEqual(incomplete.predicted_category, "UNCERTAIN")
        classifier.model.predict.assert_not_called()

        complete = {name: 100.0 for name in EXPECTED_FEATURE_NAMES}
        unsupported = classifier.predict(complete)
        self.assertEqual(unsupported.predicted_category, "UNCERTAIN")
        self.assertEqual(unsupported.probabilities, {})

    def test_event_manifest_serialization(self):
        record = MovementEventRecord(
            event_id="ev_001",
            video_id="test.mp4",
            source_video="/path/to/test.mp4",
            start_time=0.5,
            end_time=1.8,
            start_frame=15,
            end_frame=54,
            movement_category="LUNGE",
            detection_status="PROPOSED",
            representative_frame="/path/to/key.jpg",
            clip_path="/path/to/clip.mp4",
            pose_quality=0.88,
            risk_level="LOW",
        )
        manifest_file = self.temp_path / "test_manifest.json"
        save_event_manifest([record], str(manifest_file))
        self.assertTrue(manifest_file.exists())

        loaded = load_event_manifest(str(manifest_file))
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0].event_id, "ev_001")
        self.assertEqual(loaded[0].movement_category, "LUNGE")
        self.assertEqual(loaded[0].start_time, 0.5)
        self.assertEqual(loaded[0].localization_method, "legacy_unknown")

        old_record = record.to_dict()
        old_record.pop("localization_method")
        self.assertEqual(MovementEventRecord.from_dict(old_record).localization_method, "legacy_unknown")

    def test_classifier_predictor_fallback(self):
        classifier = MovementClassifier()
        feats = {
            "left_knee_min": 95.0,
            "right_knee_min": 150.0,
            "trunk_angle_max": 10.0,
        }
        res = classifier.predict(feats)
        self.assertEqual(res.predicted_category, "LUNGE")
        self.assertEqual(res.prediction_source, "heuristic")
        self.assertFalse(res.is_trained_model)
        self.assertIsNone(res.probabilities)  # Must be None, never simulated probabilities

        # Test unmatched features return UNCERTAIN, not defaulting to footwork split-step
        unmatched_feats = {
            "left_knee_min": 150.0,
            "right_knee_min": 150.0,
            "trunk_angle_max": 5.0,
        }
        res_unmatched = classifier.predict(unmatched_feats)
        self.assertEqual(res_unmatched.predicted_category, "UNCERTAIN")
        self.assertIsNone(res_unmatched.probabilities)

        missing = classifier.predict({"left_knee_min": 95.0, "right_knee_min": 150.0})
        self.assertEqual(missing.predicted_category, "UNCERTAIN")
        nonfinite = classifier.predict({"left_knee_min": 95.0, "right_knee_min": 150.0, "trunk_angle_max": float("inf")})
        self.assertEqual(nonfinite.predicted_category, "UNCERTAIN")

    def test_candidate_detector_create_candidate_helper(self):
        """Verifies MovementCandidateDetector._create_candidate constructs valid candidates."""
        from src.detection.candidate_detector import MovementCandidateDetector, MovementCandidate

        detector = MovementCandidateDetector()
        metrics = {"note": "Test candidate", "sample_metric": 12.5}
        cand = detector._create_candidate(
            video_id="test_vid.mp4",
            category="UNCERTAIN",
            start_time=1.234,
            end_time=3.567,
            score=0.45,
            metrics=metrics,
        )

        self.assertIsInstance(cand, MovementCandidate)
        self.assertEqual(cand.candidate_id, "cand_001")
        self.assertEqual(cand.video_id, "test_vid.mp4")
        self.assertEqual(cand.movement_type, "UNCERTAIN")
        self.assertEqual(cand.proposed_movement, "UNCERTAIN")
        self.assertEqual(cand.start_time, 1.23)
        self.assertEqual(cand.end_time, 3.57)
        self.assertEqual(cand.duration, 2.34)
        self.assertEqual(cand.confidence_score, 0.45)
        self.assertEqual(cand.heuristic_score, 0.45)
        self.assertEqual(cand.status, "REVIEW_REQUIRED")
        self.assertEqual(cand.human_decision, "PENDING")
        self.assertEqual(cand.reviewed_movement, "")
        self.assertEqual(cand.reviewer_notes, "")
        self.assertEqual(cand.player_id, "")

        parsed_metrics = json.loads(cand.key_metrics)
        self.assertEqual(parsed_metrics["note"], "Test candidate")
        self.assertEqual(parsed_metrics["sample_metric"], 12.5)

    @patch("src.video.video_pipeline.PoseEstimator")
    def test_video_pipeline_no_candidates_fallback_branch(self, mock_pose_estimator_cls):
        """
        Regression test: verifies that when no candidates are detected from a video
        with > 10 pose samples, the fallback branch creates an UNCERTAIN candidate
        via detector._create_candidate without raising AttributeError, preserves
        valid timestamps/schema, and does not assign injury risk.
        """
        Landmark = namedtuple("Landmark", ["x", "y", "z"])
        dummy_landmarks = [Landmark(x=0.5, y=0.5, z=0.0) for _ in range(33)]
        # Setup specific landmarks for standard standing posture
        dummy_landmarks[11] = Landmark(x=0.55, y=0.3, z=0.0)  # Left shoulder
        dummy_landmarks[12] = Landmark(x=0.65, y=0.3, z=0.0)  # Right shoulder
        dummy_landmarks[13] = Landmark(x=0.40, y=0.4, z=0.0)  # Left elbow
        dummy_landmarks[14] = Landmark(x=0.60, y=0.4, z=0.0)  # Right elbow
        dummy_landmarks[15] = Landmark(x=0.40, y=0.5, z=0.0)  # Left wrist
        dummy_landmarks[16] = Landmark(x=0.60, y=0.5, z=0.0)  # Right wrist
        dummy_landmarks[23] = Landmark(x=0.45, y=0.5, z=0.0)  # Left hip
        dummy_landmarks[24] = Landmark(x=0.55, y=0.5, z=0.0)  # Right hip
        dummy_landmarks[25] = Landmark(x=0.45, y=0.7, z=0.0)  # Left knee
        dummy_landmarks[26] = Landmark(x=0.55, y=0.7, z=0.0)  # Right knee
        dummy_landmarks[27] = Landmark(x=0.45, y=0.9, z=0.0)  # Left ankle
        dummy_landmarks[28] = Landmark(x=0.55, y=0.9, z=0.0)  # Right ankle

        mock_pose_result = MagicMock()
        mock_pose_result.has_pose = True
        mock_pose_result.landmarks = dummy_landmarks
        mock_pose_result.overall_confidence = 0.95

        mock_estimator_instance = MagicMock()
        mock_estimator_instance.process_frame.return_value = mock_pose_result
        mock_pose_estimator_cls.return_value.__enter__.return_value = mock_estimator_instance

        pipeline = VideoProcessingPipeline()
        output_dir = self.temp_path / "fallback_run"
        res = pipeline.process_video(str(self.dummy_video_path), output_dir=str(output_dir))

        self.assertTrue(res["success"])
        self.assertIsNone(res["error"])
        self.assertEqual(res["total_events"], 1)

        event = res["events"][0]
        self.assertEqual(event["movement_category"], "UNCERTAIN")
        self.assertEqual(event["detection_status"], "UNCERTAIN")
        self.assertEqual(event["risk_level"], "UNASSESSED")
        self.assertEqual(event["event_id"], "event_001")
        self.assertEqual(event["localization_method"], "whole_video_fallback")
        self.assertEqual(event["label_source"], "CANDIDATE_DETECTOR")
        self.assertAlmostEqual(event["features"]["trunk_angle_max"], 33.7, delta=0.2)
        self.assertEqual(event["start_time"], 0.0)
        self.assertGreater(event["end_time"], 0.0)
        self.assertAlmostEqual(event["end_time"], 2.0, delta=0.2)

        # Confirm manifest was written and matches
        manifest_path = Path(res["manifest_path"])
        self.assertTrue(manifest_path.exists())
        loaded_events = load_event_manifest(str(manifest_path))
        self.assertEqual(len(loaded_events), 1)
        self.assertEqual(loaded_events[0].movement_category, "UNCERTAIN")
        self.assertEqual(loaded_events[0].detection_status, "UNCERTAIN")
        self.assertEqual(loaded_events[0].risk_level, "UNASSESSED")


if __name__ == "__main__":
    unittest.main()
