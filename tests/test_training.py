import csv
import os
import tempfile
import unittest
from pathlib import Path
import pandas as pd
import numpy as np

import train


class TestModelTraining(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.csv_path = Path(self.temp_dir.name) / "test_dataset.csv"
        self.model_path = Path(self.temp_dir.name) / "test_model.pkl"

    def tearDown(self):
        self.temp_dir.cleanup()

    def _generate_synthetic_csv(self, num_rows=15, num_videos=3, classes=("LOW", "MODERATE", "HIGH")):
        fieldnames = [
            "sequence_id", "video_id", "movement_type", "camera_viewpoint",
            "frame_start", "frame_end", "duration",
            "left_knee_min", "left_knee_max", "right_knee_min", "right_knee_max",
            "left_elbow_min", "left_elbow_max", "right_elbow_min", "right_elbow_max",
            "trunk_angle_min", "trunk_angle_max", "risk_label"
        ]
        movements = ["LUNGE", "SMASH_LANDING", "CHANGE_OF_DIRECTION"]

        rows = []
        for i in range(num_rows):
            vid = f"vid_{i % num_videos:02d}.mp4"
            risk = classes[i % len(classes)]
            mov = movements[i % len(movements)]

            # Generate plausible biomechanical angles
            rows.append({
                "sequence_id": f"seq_{i:03d}",
                "video_id": vid,
                "movement_type": mov,
                "camera_viewpoint": "side_90deg",
                "frame_start": 0,
                "frame_end": 30,
                "duration": 1.0,
                "left_knee_min": float(np.random.uniform(70.0, 120.0)),
                "left_knee_max": float(np.random.uniform(140.0, 175.0)),
                "right_knee_min": float(np.random.uniform(70.0, 120.0)),
                "right_knee_max": float(np.random.uniform(140.0, 175.0)),
                "left_elbow_min": float(np.random.uniform(60.0, 100.0)),
                "left_elbow_max": float(np.random.uniform(140.0, 175.0)),
                "right_elbow_min": float(np.random.uniform(60.0, 100.0)),
                "right_elbow_max": float(np.random.uniform(140.0, 175.0)),
                "trunk_angle_min": float(np.random.uniform(2.0, 10.0)),
                "trunk_angle_max": float(np.random.uniform(15.0, 45.0)),
                "risk_label": risk,
            })

        with open(self.csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def test_train_with_grouped_videos(self):
        """Tests standard grouped training across 3 videos."""
        self._generate_synthetic_csv(num_rows=15, num_videos=3)

        # Mock args
        class Args:
            csv = str(self.csv_path)
            output = str(self.model_path)
            seed = 42

        # Run training
        orig_parse = train.parse_args
        try:
            train.parse_args = lambda: Args()
            train.main()
            self.assertTrue(self.model_path.exists())
        finally:
            train.parse_args = orig_parse

    def test_train_with_small_single_video_dataset(self):
        """Tests that small single-video dataset trains without throwing exceptions."""
        self._generate_synthetic_csv(num_rows=4, num_videos=1, classes=("LOW", "MODERATE"))

        class Args:
            csv = str(self.csv_path)
            output = str(self.model_path)
            seed = 42

        orig_parse = train.parse_args
        try:
            train.parse_args = lambda: Args()
            train.main()
            self.assertTrue(self.model_path.exists())
        finally:
            train.parse_args = orig_parse

    def test_train_handles_single_class_safely(self):
        """Tests that a single-class dataset warns but trains safely without uncaught exceptions."""
        self._generate_synthetic_csv(num_rows=8, num_videos=2, classes=("LOW",))

        class Args:
            csv = str(self.csv_path)
            output = str(self.model_path)
            seed = 42

        orig_parse = train.parse_args
        try:
            train.parse_args = lambda: Args()
            train.main()
            self.assertTrue(self.model_path.exists())
        finally:
            train.parse_args = orig_parse


if __name__ == "__main__":
    unittest.main()
