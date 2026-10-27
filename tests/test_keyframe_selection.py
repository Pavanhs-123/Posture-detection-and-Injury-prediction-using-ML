"""Regression tests for quality-gated representative source-frame selection."""

import unittest

from src.detection.candidate_detector import MovementCandidateDetector, DetectorConfig
from src.video.video_pipeline import (
    _relocalize_low_quality_candidates,
    _select_representative_frame,
)


def pose_frame(index, fps=10.0, knee_l=160.0, knee_r=165.0, quality=0.9, **extra):
    timestamp = index / fps
    frame = {
        "frame_idx": index,
        "timestamp": timestamp,
        "has_pose": True,
        "feature_knee_l": knee_l,
        "feature_knee_r": knee_r,
        "lower_body_confidence": quality,
    }
    frame.update(extra)
    return frame


class KeyframeSelectionTests(unittest.TestCase):
    def test_clear_lunge_selects_temporally_supported_deep_flexion(self):
        knees = [160, 150, 140, 120, 122, 145, 160, 165, 165, 165, 165]
        signals = [pose_frame(i, knee_l=knee, knee_r=knee + 8) for i, knee in enumerate(knees)]

        selected = _select_representative_frame(signals, 0.0, 1.0, 10.0, len(signals))

        self.assertIn(selected["representative_frame_index"], (3, 4))
        self.assertIn(selected["kinematic_peak_frame_index"], (3, 4))
        self.assertIn("local median", selected["keyframe_selection_reason"])

    def test_recovery_frame_does_not_replace_reliable_movement_peak(self):
        knees = [165, 145, 120, 122, 150, 165, 170, 170, 170, 170, 170]
        signals = [pose_frame(i, knee_l=knee, knee_r=knee + 5) for i, knee in enumerate(knees)]

        selected = _select_representative_frame(signals, 0.0, 1.0, 10.0, len(signals))

        self.assertLess(selected["representative_frame_index"], 5)
        self.assertEqual(selected["representative_frame_index"], selected["kinematic_peak_frame_index"])

    def test_missing_or_low_confidence_knees_use_quality_fallback(self):
        signals = [pose_frame(i, knee_l=None, knee_r=None, quality=0.8) for i in range(11)]
        signals[5].update({"feature_knee_l": 70.0, "feature_knee_r": 72.0, "lower_body_confidence": 0.2})

        selected = _select_representative_frame(signals, 0.0, 1.0, 10.0, len(signals))

        self.assertIsNone(selected["kinematic_peak_timestamp"])
        self.assertNotEqual(selected["representative_frame_index"], 5)
        self.assertIn("highest-quality pose frame", selected["keyframe_selection_reason"])

    def test_out_of_range_source_indices_and_timestamps_are_ignored_and_bounds_clamped(self):
        signals = [pose_frame(i, knee_l=150.0, knee_r=155.0) for i in range(11)]
        signals.extend([
            {**pose_frame(0, knee_l=60.0, knee_r=62.0), "frame_idx": -1, "timestamp": -0.1},
            {**pose_frame(5, knee_l=60.0, knee_r=62.0), "frame_idx": 25, "timestamp": 0.5},
            {**pose_frame(12, knee_l=60.0, knee_r=62.0), "frame_idx": 12, "timestamp": 1.2},
        ])

        selected = _select_representative_frame(signals, -2.0, 4.0, 10.0, 11)

        self.assertGreaterEqual(selected["representative_frame_index"], 0)
        self.assertLessEqual(selected["representative_frame_index"], 10)
        self.assertAlmostEqual(
            selected["representative_timestamp"],
            selected["representative_frame_index"] / 10.0,
        )
        self.assertIn("clamped to available source frames", selected["keyframe_selection_reason"])

    def test_no_reliable_pose_uses_clamped_midpoint_without_claiming_peak(self):
        signals = [{
            "frame_idx": 3,
            "timestamp": 0.3,
            "has_pose": False,
            "lower_body_confidence": 0.0,
        }]

        selected = _select_representative_frame(signals, 0.2, 0.8, 10.0, 6)

        self.assertEqual(selected["representative_frame_index"], 4)
        self.assertIsNone(selected["kinematic_peak_timestamp"])
        self.assertIn("No reliable pose frame", selected["keyframe_selection_reason"])

    def test_low_quality_off_peak_detector_window_becomes_uncertain_peak_review_window(self):
        detector = MovementCandidateDetector(DetectorConfig(sample_fps=10.0))
        signals = [pose_frame(i, knee_l=165.0, knee_r=168.0, quality=0.9) for i in range(41)]
        for i, knee in ((3, 145.0), (4, 130.0), (5, 132.0), (6, 145.0)):
            signals[i]["feature_knee_l"] = knee
            signals[i]["feature_knee_r"] = knee + 4.0
        for signal in signals[25:40]:
            signal["lower_body_confidence"] = 0.2
        original = detector._create_candidate(
            video_id="test.mp4",
            category="CHANGE_OF_DIRECTION",
            start_time=2.5,
            end_time=4.0,
            score=0.81,
        )

        localized, review_ids = _relocalize_low_quality_candidates(
            [original], signals, 10.0, 41, 4.0, detector
        )

        self.assertEqual(len(localized), 1)
        review = localized[0]
        self.assertEqual(review.proposed_movement, "UNCERTAIN")
        self.assertEqual(review.status, "REVIEW_REQUIRED")
        self.assertIn(id(review), review_ids)
        self.assertAlmostEqual(review.start_time, 0.0)
        self.assertAlmostEqual(review.end_time, 1.0)
        self.assertIn("Low-quality detector window", review.key_metrics)
        self.assertIn("CHANGE_OF_DIRECTION", review.key_metrics)


if __name__ == "__main__":
    unittest.main()
