"""
Model evaluation utility.
Loads a model artifact and generates a full metrics report and confusion matrix on test data.
"""

import argparse
import pickle
import sys
from pathlib import Path
import pandas as pd

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from src.evaluation.metrics import compute_evaluation_metrics
from src.classification.predictor import EXPECTED_FEATURE_NAMES


def evaluate_model_file(
    model_path: str,
    test_data_path: str,
    target_col: str = "risk_label",
):
    with open(model_path, "rb") as f:
        data = pickle.load(f)
    model = data.get("model", data) if isinstance(data, dict) else data

    df = pd.read_csv(test_data_path)
    for col in EXPECTED_FEATURE_NAMES:
        if col not in df.columns:
            df[col] = 180.0 if "knee" in col or "elbow" in col else 0.0

    X = df[EXPECTED_FEATURE_NAMES].values
    y_true = df[target_col].astype(str).str.strip().str.upper().values

    y_pred = model.predict(X)
    metrics = compute_evaluation_metrics(y_true, y_pred)

    print("\n" + "=" * 60)
    print(f"EVALUATION REPORT: {Path(model_path).name} on {Path(test_data_path).name}")
    print("=" * 60)
    print(f"Macro F1-Score:    {metrics['macro_f1']:.4f}")
    print(f"Weighted F1-Score: {metrics['weighted_f1']:.4f}")
    print(f"Accuracy:          {metrics['accuracy']:.4f}")
    print(f"Total Samples:     {metrics['support']}")
    print("\nPer-Class Breakdown:")
    for lbl, stats in metrics["per_class"].items():
        print(f"  • {lbl:<15}: Precision={stats['precision']:.3f}, Recall={stats['recall']:.3f}, F1={stats['f1']:.3f} (N={stats['support']})")
    print("\nConfusion Matrix:")
    print(f"Labels: {metrics['labels']}")
    for row in metrics["confusion_matrix"]:
        print(f"  {row}")
    print("=" * 60 + "\n")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate a saved model artifact")
    parser.add_argument("--model", type=str, default="models/injury_risk_model.pkl")
    parser.add_argument("--data", type=str, default="data/dataset.csv")
    parser.add_argument("--target", type=str, default="risk_label")
    args = parser.parse_args()

    evaluate_model_file(args.model, args.data, args.target)


if __name__ == "__main__":
    main()
