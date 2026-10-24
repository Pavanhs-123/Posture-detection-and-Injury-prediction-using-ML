"""
Single model training pipeline.
Supports training Random Forest, Extra Trees, or SVM on sequence dataset.
"""

import argparse
import pickle
import sys
from pathlib import Path
import pandas as pd

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from training.compare_models import build_candidate_pipelines, prepare_features_and_labels
from src.evaluation.split import grouped_train_val_test_split
from src.evaluation.metrics import compute_evaluation_metrics
from src.classification.predictor import EXPECTED_FEATURE_NAMES


def train_single_model(
    data_path: str,
    output_path: str = "models/injury_risk_model.pkl",
    model_type: str = "RandomForest",
    target_col: str = "risk_label",
    random_state: int = 42,
):
    df = pd.read_csv(data_path)
    if "risk_label" in target_col:
        valid_mask = df[target_col].isin(["LOW", "MODERATE", "HIGH"])
    else:
        valid_mask = ~df[target_col].isin(["UNCERTAIN", "NO_EVENT", "PENDING", ""])
    clean_df = df[valid_mask].copy()

    feature_cols = [c for c in EXPECTED_FEATURE_NAMES if c in clean_df.columns]
    if len(feature_cols) < 5:
        feature_cols = EXPECTED_FEATURE_NAMES

    train_df, val_df, test_df = grouped_train_val_test_split(
        clean_df, group_col="video_id", random_state=random_state
    )

    X_train, y_train, _ = prepare_features_and_labels(train_df, target_col, feature_cols)
    pipelines = build_candidate_pipelines(random_state=random_state)
    pipe = pipelines.get(model_type, pipelines["RandomForest"])

    print(f"Training {model_type} pipeline on {len(X_train)} samples...")
    pipe.fit(X_train, y_train)

    if len(val_df) > 0:
        X_val, y_val, _ = prepare_features_and_labels(val_df, target_col, feature_cols)
        y_pred = pipe.predict(X_val)
        metrics = compute_evaluation_metrics(y_val, y_pred)
        print(f"Validation Macro F1: {metrics['macro_f1']:.4f} | Accuracy: {metrics['accuracy']:.4f}")

    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "wb") as f:
        pickle.dump(pipe, f)
    print(f"Saved model to: {out_p}")


def main():
    parser = argparse.ArgumentParser(description="Train a single model pipeline")
    parser.add_argument("--data", type=str, default="data/dataset.csv")
    parser.add_argument("--output", type=str, default="models/injury_risk_model.pkl")
    parser.add_argument("--model-type", type=str, choices=["RandomForest", "ExtraTrees", "SVM"], default="RandomForest")
    parser.add_argument("--target", type=str, default="risk_label")
    args = parser.parse_args()

    train_single_model(
        data_path=args.data,
        output_path=args.output,
        model_type=args.model_type,
        target_col=args.target,
    )


if __name__ == "__main__":
    main()
