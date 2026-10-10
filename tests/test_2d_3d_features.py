import os
import unittest
import numpy as np
import cv2

from pose_analysis import (
    calculate_angle,
    calculate_angle_3d,
    get_torso_lean_3d,
    get_knee_angle_3d,
    get_elbow_angle_3d,
    calculate_landmark_quality,
)
from compare_2d_3d import process_video_comparison


class MockLandmark:
    def __init__(self, x, y, z=0.0, visibility=1.0, presence=1.0):
        self.x = x
        self.y = y
        self.z = z
        self.visibility = visibility
        self.presence = presence


class MockPoseLandmark:
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28


class Test2D3DFeatures(unittest.TestCase):
    def test_3d_angle_calculation_orthogonal(self):
        # 90 degrees in 3D: (1, 0, 0) -> (0, 0, 0) -> (0, 1, 0)
        a = [1.0, 0.0, 0.0]
        b = [0.0, 0.0, 0.0]
        c = [0.0, 1.0, 0.0]
        angle = calculate_angle_3d(a, b, c)
        self.assertIsNotNone(angle)
        self.assertAlmostEqual(angle, 90.0, places=2)

    def test_3d_angle_calculation_straight(self):
        # 180 degrees in 3D
        a = [-1.0, 0.0, 0.0]
        b = [0.0, 0.0, 0.0]
        c = [1.0, 0.0, 0.0]
        angle = calculate_angle_3d(a, b, c)
        self.assertIsNotNone(angle)
        self.assertAlmostEqual(angle, 180.0, places=2)

    def test_3d_angle_degenerate_vectors(self):
        # Coincident points (zero length vector)
        a = [0.0, 0.0, 0.0]
        b = [0.0, 0.0, 0.0]
        c = [1.0, 0.0, 0.0]
        angle = calculate_angle_3d(a, b, c)
        self.assertIsNone(angle)

    def test_torso_lean_3d_upright(self):
        # 33 dummy landmarks
        lms = [MockLandmark(0, 0, 0) for _ in range(33)]
        # Shoulders at y = 1.0 (vertical), hips at y = 0.0
        lms[MockPoseLandmark.LEFT_SHOULDER] = MockLandmark(-0.2, 1.0, 0.0)
        lms[MockPoseLandmark.RIGHT_SHOULDER] = MockLandmark(0.2, 1.0, 0.0)
        lms[MockPoseLandmark.LEFT_HIP] = MockLandmark(-0.15, 0.0, 0.0)
        lms[MockPoseLandmark.RIGHT_HIP] = MockLandmark(0.15, 0.0, 0.0)

        lean = get_torso_lean_3d(lms, MockPoseLandmark)
        self.assertIsNotNone(lean)
        # Should be approximately 0 degrees (upright)
        self.assertAlmostEqual(lean, 0.0, delta=0.5)

    def test_torso_lean_3d_pitched_forward(self):
        lms = [MockLandmark(0, 0, 0) for _ in range(33)]
        # Shoulders shifted along Z (forward depth = 1.0) while y = 1.0 -> 45 degree tilt
        lms[MockPoseLandmark.LEFT_SHOULDER] = MockLandmark(-0.2, 1.0, 1.0)
        lms[MockPoseLandmark.RIGHT_SHOULDER] = MockLandmark(0.2, 1.0, 1.0)
        lms[MockPoseLandmark.LEFT_HIP] = MockLandmark(-0.15, 0.0, 0.0)
        lms[MockPoseLandmark.RIGHT_HIP] = MockLandmark(0.15, 0.0, 0.0)

        lean = get_torso_lean_3d(lms, MockPoseLandmark)
        self.assertIsNotNone(lean)
        self.assertAlmostEqual(lean, 45.0, delta=0.5)

    def test_confidence_rejection(self):
        lms = [MockLandmark(0, 0, 0) for _ in range(33)]
        # Set left knee with low visibility
        lms[MockPoseLandmark.LEFT_HIP] = MockLandmark(0, 1.0, 0, visibility=0.9)
        lms[MockPoseLandmark.LEFT_KNEE] = MockLandmark(0, 0.5, 0, visibility=0.2)  # Low confidence
        lms[MockPoseLandmark.LEFT_ANKLE] = MockLandmark(0, 0.0, 0, visibility=0.9)

        # Right knee with high visibility
        lms[MockPoseLandmark.RIGHT_HIP] = MockLandmark(0.2, 1.0, 0, visibility=0.9)
        lms[MockPoseLandmark.RIGHT_KNEE] = MockLandmark(0.2, 0.5, 0, visibility=0.9)
        lms[MockPoseLandmark.RIGHT_ANKLE] = MockLandmark(0.2, 0.0, 0, visibility=0.9)

        # With min_confidence=0.5, left knee should be rejected (None)
        left_k, right_k, _ = get_knee_angle_3d(lms, MockPoseLandmark, min_confidence=0.5)
        self.assertIsNone(left_k)
        self.assertIsNotNone(right_k)

        # Quality tracker should report low confidence for left knee
        quality = calculate_landmark_quality(lms, MockPoseLandmark)
        self.assertEqual(quality.left_knee_conf, 0.2)
        self.assertEqual(quality.right_knee_conf, 0.9)

    def test_compare_script_execution_on_dummy_video(self):
        temp_dir = os.path.dirname(os.path.abspath(__file__))
        dummy_video = os.path.join(temp_dir, "temp_compare_test.mp4")
        output_csv = os.path.join(temp_dir, "temp_compare_out.csv")

        width, height = 320, 240
        fps = 10
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(dummy_video, fourcc, fps, (width, height))
        for _ in range(10):
            writer.write(np.zeros((height, width, 3), dtype=np.uint8))
        writer.release()

        try:
            rows = process_video_comparison(
                dummy_video, output_csv, sample_fps=5.0, min_confidence=0.5
            )
            self.assertTrue(os.path.exists(output_csv))
            self.assertEqual(len(rows), 5)  # 1.0s at 5 fps
            # All frames should have pose_detected == 0 on black screen
            self.assertTrue(all(r["pose_detected"] == 0 for r in rows))
        finally:
            if os.path.exists(dummy_video):
                os.remove(dummy_video)
            if os.path.exists(output_csv):
                os.remove(output_csv)


if __name__ == "__main__":
    unittest.main()
