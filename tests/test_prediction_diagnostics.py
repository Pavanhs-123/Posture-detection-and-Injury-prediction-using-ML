"""Regression coverage for explicit classifier feature-gate diagnostics."""

import unittest
from types import SimpleNamespace

from src.classification.predictor import EXPECTED_FEATURE_NAMES, MovementClassifier
from src.video.video_pipeline import _resolve_event_classification


class PredictionDiagnosticsTests(unittest.TestCase):
    def test_unavailable_classifier_reports_the_exact_missing_feature(self):
        classifier = MovementClassifier()
        classifier.is_trained_model = True
        classifier.feature_names = list(EXPECTED_FEATURE_NAMES)
        classifier.model = object()
        features = {name: 100.0 for name in EXPECTED_FEATURE_NAMES}
        features["right_elbow_min"] = None

        result = classifier.predict(features)

        self.assertEqual(result.predicted_category, "UNCERTAIN")
        self.assertEqual(result.prediction_source, "unavailable")
        self.assertIn("right_elbow_min", result.status_message)
        self.assertIn("missing or invalid", result.status_message)

    def test_pipeline_preserves_classifier_gate_reason_when_event_is_uncertain(self):
        prediction = SimpleNamespace(
            predicted_category="UNCERTAIN",
            status_message="Classifier unavailable: Required feature 'right_elbow_min' is missing or invalid.",
        )
        result = _resolve_event_classification("CHANGE_OF_DIRECTION", prediction, True)
        self.assertEqual(result["movement_category"], "UNCERTAIN")
        self.assertIn("right_elbow_min", result["classification_note"])
        self.assertIn("Quality gate", result["classification_note"])

    def test_detector_ambiguous_proposal_does_not_get_replaced_by_classifier(self):
        prediction = SimpleNamespace(
            predicted_category="LUNGE",
            status_message="Model inference complete.",
            probabilities={"LUNGE": 0.72, "CHANGE_OF_DIRECTION": 0.179, "SMASH_LANDING": 0.101},
        )
        result = _resolve_event_classification("UNCERTAIN", prediction, True)
        self.assertEqual(result["movement_category"], "UNCERTAIN")
        self.assertIn("ambiguous", result["classification_note"])

    def test_recorded_lunge_classifier_suggestion_does_not_override_uncertain_detector(self):
        """Recorded Test_Lunge_2 scores stay separate from the unverified detector fallback."""
        prediction = SimpleNamespace(
            predicted_category="LUNGE",
            status_message="Model inference complete.",
            probabilities={"LUNGE": 0.72, "CHANGE_OF_DIRECTION": 0.179, "SMASH_LANDING": 0.101},
        )
        result = _resolve_event_classification("UNCERTAIN", prediction, True)

        self.assertEqual(result["movement_category"], "UNCERTAIN")
        self.assertEqual(result["label_source"], "CANDIDATE_DETECTOR")
        self.assertIn("classifier suggestion", result["classification_note"])
        self.assertIn("do not resolve detector uncertainty", result["classification_note"])

    def test_classifier_prediction_and_detector_proposal_remain_separately_auditable(self):
        prediction = SimpleNamespace(
            predicted_category="SMASH_LANDING",
            status_message="Model inference complete.",
        )
        result = _resolve_event_classification("CHANGE_OF_DIRECTION", prediction, True)

        self.assertEqual(result["movement_category"], "SMASH_LANDING")
        self.assertEqual(result["label_source"], "CLASSIFIER_MODEL")
        self.assertIn("trained classifier prediction", result["classification_note"])


if __name__ == "__main__":
    unittest.main()
