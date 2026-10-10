import os
import unittest
import numpy as np
import cv2

from pose_analysis import calculate_angle
from src.model.integration_test_model import IntegrationTestModel
from src.model.interface import InjuryRiskModel
from src.inference.pipeline import analyze_video

class TestPosturePipeline(unittest.TestCase):
    def test_angle_calculation(self):
        # 90 degree angle test: (1,0) -> (0,0) -> (0,1)
        a = [1.0, 0.0]
        b = [0.0, 0.0]
        c = [0.0, 1.0]
        angle = calculate_angle(a, b, c)
        self.assertAlmostEqual(angle, 90.0, places=2)

        # 180 degree angle test: (-1,0) -> (0,0) -> (1,0)
        a = [-1.0, 0.0]
        b = [0.0, 0.0]
        c = [1.0, 0.0]
        angle = calculate_angle(a, b, c)
        self.assertAlmostEqual(angle, 180.0, places=2)

    def test_angle_calculation_with_aspect_ratio_scaling(self):
        # In a 1920x1080 (16:9) frame, a point at normalized (0.1, 0.1) is at pixel (192, 108)
        # Without image_dimensions, (0.1, 0) -> (0,0) -> (0.1, 0.1) gives 45 degrees
        b = [0.0, 0.0]
        a = [0.1, 0.0]
        c = [0.1, 0.1]
        raw_angle = calculate_angle(a, b, c)
        self.assertAlmostEqual(raw_angle, 45.0, places=2)

        # With 1920x1080 dimensions, arctan(108 / 192) = arctan(0.5625) ~ 29.358 degrees
        scaled_angle = calculate_angle(a, b, c, image_dimensions=(1920, 1080))
        expected_scaled = np.degrees(np.arctan2(108.0, 192.0))
        self.assertAlmostEqual(scaled_angle, expected_scaled, places=2)
        self.assertNotEqual(raw_angle, scaled_angle)

    def test_integration_test_model(self):
        model = IntegrationTestModel()
        model.load()

        # Test LOW risk features (nominal knees, minimal torso lean)
        low_feats = {
            "left_knee_min": 140.0, "left_knee_max": 160.0,
            "right_knee_min": 140.0, "right_knee_max": 160.0,
            "left_elbow_min": 150.0, "left_elbow_max": 170.0,
            "right_elbow_min": 150.0, "right_elbow_max": 170.0,
            "trunk_angle_min": 5.0, "trunk_angle_max": 12.0
        }
        self.assertEqual(model.predict(low_feats), "LOW")
        low_probs = model.predict_proba(low_feats)
        self.assertGreater(low_probs["LOW"], 0.5)

        # Test MODERATE risk features (intermediate knee bend or torso lean)
        mod_feats = low_feats.copy()
        mod_feats["left_knee_min"] = 95.0 # between 80 and 110
        self.assertEqual(model.predict(mod_feats), "MODERATE")

        # Test HIGH risk features (extreme knee bend or excessive torso lean)
        high_feats = low_feats.copy()
        high_feats["trunk_angle_max"] = 45.0 # > 40
        self.assertEqual(model.predict(high_feats), "HIGH")

    def test_pipeline_on_dummy_video(self):
        # Create a tiny 1-second black dummy video file
        temp_dir = tempfile_dir = os.path.dirname(os.path.abspath(__file__))
        dummy_video_path = os.path.join(temp_dir, "dummy_test_video.mp4")
        
        width, height = 320, 240
        fps = 10
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(dummy_video_path, fourcc, fps, (width, height))
        
        # Write 10 frames of zeros (black screen)
        for _ in range(10):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            writer.write(frame)
        writer.release()

        # Run pipeline. Since there is no person in a black video, it should return gracefully
        # with success=False or an error stating no pose was detected.
        result = analyze_video(dummy_video_path, start_time=0.0, end_time=1.0, sample_fps=5.0)
        
        # Verify it handled the absence of pose detection gracefully instead of throwing exceptions
        self.assertFalse(result["success"])
        self.assertIn("No pose detected", result["error"])

        # Cleanup
        if os.path.exists(dummy_video_path):
            os.remove(dummy_video_path)

    def test_injury_risk_model_interface(self):
        # Test fallback when no model loaded
        model = InjuryRiskModel()
        with self.assertRaises(ValueError):
            model.predict({})

        # Mock a model with predict_proba
        class DummyPipeline:
            classes_ = np.array(["LOW", "HIGH"])
            def predict(self, X):
                return np.array(["LOW"])
            def predict_proba(self, X):
                return np.array([[0.8, 0.2]])

        model.model = DummyPipeline()
        pred = model.predict({"left_knee_min": 140.0})
        self.assertEqual(pred, "LOW")

        probs = model.predict_proba({"left_knee_min": 140.0})
        self.assertIn("LOW", probs)
        self.assertIn("MODERATE", probs)
        self.assertIn("HIGH", probs)
        self.assertEqual(probs["LOW"], 0.8)
        self.assertEqual(probs["HIGH"], 0.2)
        self.assertEqual(probs["MODERATE"], 0.0)


if __name__ == "__main__":
    unittest.main()
