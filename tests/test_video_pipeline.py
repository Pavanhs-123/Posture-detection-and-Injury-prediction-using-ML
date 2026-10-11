"""
Unit and integration tests for video pipeline, validation, clip extraction,
5-category classification schema, Rubric v0.3 risk evaluation, and manifest generation.
"""

import os
import tempfile
import unittest
from pathlib import Path
import cv2
import numpy as np

from src.video.validator import validate_video
from src.video.clip_extractor import extract_subclip, extract_keyframe
from src.video.video_pipeline import VideoProcessingPipeline
from src.classification.categories import (
    MOVEMENT_CATEGORIES,
    ACTIVE_RISK_RUBRIC_CATEGORIES,
    DEFERRED_RISK_RUBRIC_CATEGORIES,
)
from src.classification.predictor import MovementClassifier
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


if __name__ == "__main__":
    unittest.main()
