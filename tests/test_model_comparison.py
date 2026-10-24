"""
Tests for 3-model training framework, comparative evaluation,
grouped leakage-free splitting, and model registry serialization.
"""

import os
import tempfile
import unittest
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.calibration import CalibratedClassifierCV

from training.compare_models import (
    build_candidate_pipelines,
    compare_and_train_models,
)
from src.evaluation.split import grouped_train_val_test_split
from src.evaluation.metrics import compute_evaluation_metrics
from src.classification.predictor import EXPECTED_FEATURE_NAMES


class TestModelComparisonFramework(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

        # Create a synthetic dataset with 3 distinct videos
        rows = []
        videos = ["vid_A.mp4", "vid_B.mp4", "vid_C.mp4"]
        risk_labels = ["LOW", "MODERATE", "HIGH"]
        mov_labels = ["LUNGE", "SMASH_LANDING", "CHANGE_OF_DIRECTION"]

        rng = np.random.RandomState(42)
        for i, vid in enumerate(videos):
            for j in range(6):  # 6 events per video
                row = {
                    "sequence_id": f"seq_{i}_{j}",
                    "video_id": vid,
                    "risk_label": risk_labels[j % 3],
                    "movement_type": mov_labels[j % 3],
                }
                for f in EXPECTED_FEATURE_NAMES:
                    row[f] = float(rng.uniform(70.0, 160.0))
                rows.append(row)

        self.df = pd.DataFrame(rows)
        self.data_csv = self.temp_path / "test_dataset.csv"
        self.df.to_csv(self.data_csv, index=False)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_build_candidate_pipelines(self):
        pipes = build_candidate_pipelines()
        self.assertIn("RandomForest", pipes)
        self.assertIn("ExtraTrees", pipes)
        self.assertIn("SVM", pipes)
        self.assertEqual(len(pipes), 3)
        self.assertIsInstance(pipes["SVM"].named_steps["classifier"], CalibratedClassifierCV)

    def test_compute_evaluation_metrics(self):
        y_true = ["LOW", "LOW", "MODERATE", "HIGH"]
        y_pred = ["LOW", "MODERATE", "MODERATE", "HIGH"]
        metrics = compute_evaluation_metrics(y_true, y_pred)

        self.assertIn("macro_f1", metrics)
        self.assertIn("weighted_f1", metrics)
        self.assertIn("accuracy", metrics)
        self.assertIn("confusion_matrix", metrics)
        self.assertGreater(metrics["macro_f1"], 0.0)
        self.assertEqual(metrics["accuracy"], 0.75)

    def test_grouped_split_prevents_leakage(self):
        train_df, val_df, test_df = grouped_train_val_test_split(
            self.df, group_col="video_id", train_ratio=0.7, val_ratio=0.15, test_ratio=0.15
        )
        train_vids = set(train_df["video_id"].unique())
        val_vids = set(val_df["video_id"].unique())
        test_vids = set(test_df["video_id"].unique())

        # Assert no video overlaps between any split pairs
        self.assertTrue(train_vids.isdisjoint(val_vids), "Train and Val videos must be disjoint")
        self.assertTrue(train_vids.isdisjoint(test_vids), "Train and Test videos must be disjoint")
        self.assertTrue(val_vids.isdisjoint(test_vids), "Val and Test videos must be disjoint")

    def test_compare_and_train_models_end_to_end(self):
        output_dir = self.temp_path / "models_out"
        registry = compare_and_train_models(
            data_path=str(self.data_csv),
            target_col="risk_label",
            output_dir=str(output_dir),
            random_state=42,
        )

        self.assertIn("selected_model", registry)
        self.assertIn(registry["selected_model"], ["RandomForest", "ExtraTrees", "SVM"])
        self.assertEqual(registry["selection_metric"], "macro_f1")
        self.assertIn("candidate_comparison", registry)
        self.assertEqual(len(registry["candidate_comparison"]), 3)

        # Verify artifacts were written to disk
        registry_file = output_dir / "model_registry.json"
        self.assertTrue(registry_file.exists())
        primary_model = output_dir / "injury_risk_model.pkl"
        self.assertTrue(primary_model.exists())

    def test_insufficient_groups_raises_error(self):
        """Verifies that datasets with fewer than 3 groups raise a clear ValueError."""
        two_group_df = self.df[self.df["video_id"].isin(["vid_A.mp4", "vid_B.mp4"])].copy()
        with self.assertRaises(ValueError) as ctx:
            grouped_train_val_test_split(two_group_df, group_col="video_id")
        self.assertIn("Insufficient independent groups", str(ctx.exception))

    def test_player_id_grouped_split_prevents_subject_leakage(self):
        """Verifies player_id grouping keeps a player's events in one split even across multiple videos."""
        rows = []
        # Player P1 has events across vid_1 and vid_2
        # Player P2 has events across vid_3
        # Player P3 has events across vid_4
        players = ["P1", "P1", "P2", "P3"]
        vids = ["vid_1.mp4", "vid_2.mp4", "vid_3.mp4", "vid_4.mp4"]
        for p, v in zip(players, vids):
            for i in range(3):
                row = {
                    "sequence_id": f"{p}_{v}_{i}",
                    "player_id": p,
                    "video_id": v,
                    "movement_type": "LUNGE",
                    "risk_label": "LOW",
                }
                for f in EXPECTED_FEATURE_NAMES:
                    row[f] = 100.0
                rows.append(row)
        df_players = pd.DataFrame(rows)

        train_df, val_df, test_df = grouped_train_val_test_split(df_players, group_col="player_id")

        p_train = set(train_df["player_id"].unique())
        p_val = set(val_df["player_id"].unique())
        p_test = set(test_df["player_id"].unique())

        # No player in multiple splits
        self.assertTrue(p_train.isdisjoint(p_val))
        self.assertTrue(p_train.isdisjoint(p_test))
        self.assertTrue(p_val.isdisjoint(p_test))

        # Check all P1 events (from both vid_1 and vid_2) are in the same split
        p1_splits = [
            "train" if "P1" in p_train else None,
            "val" if "P1" in p_val else None,
            "test" if "P1" in p_test else None,
        ]
        active_p1_splits = [s for s in p1_splits if s is not None]
        self.assertEqual(len(active_p1_splits), 1, "Player P1 must only exist in exactly 1 split partition")


if __name__ == "__main__":
    unittest.main()
