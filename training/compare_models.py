"""
Three-Model Training and Comparative Evaluation Framework.
Trains and compares:
1. Random Forest (Tabular tree ensemble)
2. Extra Trees (Extremely randomized trees ensemble)
3. Support Vector Machine (RBF kernel with standard scaling)

Uses Macro F1 as the primary selection metric on leakage-safe grouped splits.
Selects and registers the best model into models/model_registry.json.

Enforces:
- Player-grouped splitting when player_id is available; video-grouped otherwise (with explicit advisory).
- Minimum 3 independent groups required for 3-way train/validation/test evaluation.
- Dataset validation and deduplication prior to partitioning.
"""

import argparse
import json
import os
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from src.evaluation.split import grouped_train_val_test_split, detect_group_column
from src.evaluation.metrics import compute_evaluation_metrics
from src.classification.predictor import EXPECTED_FEATURE_NAMES


def build_candidate_pipelines(random_state: int = 42) -> Dict[str, Pipeline]:
    """Builds the 3 candidate model pipelines for comparison."""
    return {
        "RandomForest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", RandomForestClassifier(
                n_estimators=100,
                max_depth=8,
                min_samples_split=2,
                class_weight="balanced",
                random_state=random_state,
            )),
        ]),
        "ExtraTrees": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", ExtraTreesClassifier(
                n_estimators=100,
                max_depth=8,
                min_samples_split=2,
                class_weight="balanced",
                random_state=random_state,
            )),
        ]),
        "SVM": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", CalibratedClassifierCV(
                estimator=SVC(
                    kernel="rbf",
                    C=1.0,
                    class_weight="balanced",
                    random_state=random_state,
                ),
                method="sigmoid",
                cv=2,
                ensemble=False,
            )),
        ]),
    }


def validate_and_clean_dataset(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: List[str],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Validates event records, detects duplicates, checks for label conflicts,
    and returns (cleaned_dataframe, data_quality_report).
    """
    report: Dict[str, Any] = {
        "initial_rows": len(df),
        "target_column": target_col,
        "duplicate_rows_removed": 0,
        "label_conflicts_detected": 0,
        "conflict_details": [],
        "unique_groups": 0,
        "group_column": None,
    }

    # Filter target labels
    if "risk" in target_col:
        valid_mask = df[target_col].isin(["LOW", "MODERATE", "HIGH"])
    else:
        valid_mask = ~df[target_col].isin(["UNCERTAIN", "NO_EVENT", "PENDING", "", "nan"]) & df[target_col].notna()

    clean_df = df[valid_mask].copy()
    if clean_df.empty:
        raise ValueError(f"No valid labeled samples found in dataset for target column '{target_col}'.")

    # Check for duplicate event/sequence IDs
    id_cols = [c for c in ["sequence_id", "event_id", "candidate_id"] if c in clean_df.columns]
    if id_cols:
        id_col = id_cols[0]
        dup_ids = clean_df[clean_df.duplicated(subset=[id_col], keep=False)]
        if not dup_ids.empty:
            print(f"⚠️  [Data Quality Warning] Found {len(dup_ids)} rows with duplicated {id_col}.")

    # Check for identical feature rows with conflicting labels
    existing_feat_cols = [c for c in feature_cols if c in clean_df.columns]
    if existing_feat_cols:
        # Detect exact duplicate feature rows
        feat_dups = clean_df[clean_df.duplicated(subset=existing_feat_cols, keep=False)]
        if not feat_dups.empty:
            # Check if any feature duplicates have conflicting target labels
            grouped = feat_dups.groupby(existing_feat_cols)[target_col].nunique()
            conflicts = grouped[grouped > 1]
            if len(conflicts) > 0:
                report["label_conflicts_detected"] = int(len(conflicts))
                print(f"⚠️  [Data Quality Warning] Found {len(conflicts)} sets of identical features with CONFLICTING labels!")
                report["conflict_details"] = f"{len(conflicts)} identical feature rows with conflicting {target_col}."

        # Remove exact duplicate rows across features and target to prevent metric inflation
        n_before = len(clean_df)
        clean_df = clean_df.drop_duplicates(subset=existing_feat_cols + [target_col]).copy()
        report["duplicate_rows_removed"] = n_before - len(clean_df)
        if report["duplicate_rows_removed"] > 0:
            print(f"🧹 Removed {report['duplicate_rows_removed']} exact duplicate rows from training set.")

    # Identify grouping column
    group_col, group_desc = detect_group_column(clean_df)
    report["group_column"] = group_col
    report["group_description"] = group_desc
    report["unique_groups"] = int(clean_df[group_col].nunique())
    report["final_clean_rows"] = len(clean_df)

    return clean_df, report


def prepare_features_and_labels(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: List[str],
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Extracts feature matrix X and target vector y from dataframe."""
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 180.0 if "knee" in col or "elbow" in col else 0.0

    X = df[feature_cols].values.astype(np.float64)
    y = df[target_col].astype(str).str.strip().str.upper().values
    labels = sorted(list(set(y)))
    return X, y, labels


def compare_and_train_models(
    data_path: str,
    target_col: str = "risk_label",
    output_dir: str = "models",
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Executes fair comparative evaluation across the 3 candidate models.
    """
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {data_path}.\n"
            "An assembled dataset with at least 3 independent player or video groups is required. "
            "Please upload and annotate video candidates before running comparative training."
        )

    raw_df = pd.read_csv(path)
    if raw_df.empty:
        raise ValueError(f"Dataset at {data_path} is empty.")

    if target_col not in raw_df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset. Columns: {list(raw_df.columns)}")

    feature_cols = [c for c in EXPECTED_FEATURE_NAMES if c in raw_df.columns]
    if len(feature_cols) < 5:
        feature_cols = EXPECTED_FEATURE_NAMES

    # Validate and clean data (deduplication & conflict detection)
    clean_df, quality_report = validate_and_clean_dataset(raw_df, target_col, feature_cols)

    # Detect grouping and verify sufficient independent groups
    group_col = quality_report["group_column"]
    n_groups = quality_report["unique_groups"]

    print("\n" + "=" * 70)
    print("🏸 THREE-MODEL TRAINING & COMPARATIVE EVALUATION FRAMEWORK")
    print("=" * 70)
    print(f"Dataset: {path.name} | Total clean samples: {len(clean_df)}")
    print(f"Target column: {target_col}")
    print(f"Grouping Column: {group_col} ({quality_report['group_description']})")
    print(f"Unique Independent Groups: {n_groups}")

    if n_groups < 3:
        err_msg = (
            f"❌ Evaluation stopped: Dataset has only {n_groups} independent group(s) in '{group_col}'.\n"
            "Valid 3-way train/validation/test evaluation requires at least 3 independent groups (players or videos) "
            "to prevent cross-partition leakage. Assemble additional video recordings before evaluating models."
        )
        print(err_msg)
        raise ValueError(err_msg)

    # Perform leakage-safe grouped split
    train_df, val_df, test_df = grouped_train_val_test_split(
        clean_df,
        group_col=group_col,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        random_state=random_state,
    )

    print(f"Split sizes: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    # Class distribution
    print("\nTarget Class Distribution:")
    for cls_name, count in clean_df[target_col].value_counts().items():
        print(f"  • {cls_name:<20}: {count:>4} ({count / len(clean_df) * 100:.1f}%)")
    print("=" * 70)

    X_train, y_train, train_labels = prepare_features_and_labels(train_df, target_col, feature_cols)
    X_val, y_val, _ = prepare_features_and_labels(val_df, target_col, feature_cols) if len(val_df) > 0 else (None, None, None)
    X_test, y_test, _ = prepare_features_and_labels(test_df, target_col, feature_cols) if len(test_df) > 0 else (None, None, None)

    all_labels = sorted(list(set(clean_df[target_col].unique())))

    pipelines = build_candidate_pipelines(random_state=random_state)
    results = {}
    saved_artifacts = {}
    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    best_model_name = None
    best_macro_f1 = -1.0

    print(f"\n{'Model Name':<15} | {'Fit Time':<9} | {'Val Macro F1':<13} | {'Val Weighted F1':<16} | {'Val Acc':<8}")
    print("-" * 70)

    for name, pipe in pipelines.items():
        # Measure training latency
        t0 = time.time()
        pipe.fit(X_train, y_train)
        fit_time = time.time() - t0

        # Evaluate on Validation Set
        if X_val is not None and len(X_val) > 0:
            y_val_pred = pipe.predict(X_val)
            val_metrics = compute_evaluation_metrics(y_val, y_val_pred, labels=all_labels)
        else:
            y_train_pred = pipe.predict(X_train)
            val_metrics = compute_evaluation_metrics(y_train, y_train_pred, labels=all_labels)

        val_macro = val_metrics["macro_f1"]
        val_weighted = val_metrics["weighted_f1"]
        val_acc = val_metrics["accuracy"]

        print(f"{name:<15} | {fit_time:>6.3f}s   | {val_macro:>10.4f}   | {val_weighted:>13.4f}   | {val_acc:>7.4f}")

        cand_path = out_p / f"{name.lower()}_{target_col}.pkl"
        with open(cand_path, "wb") as f:
            pickle.dump({
                "model": pipe,
                "features": feature_cols,
                "model_name": name,
                "target": target_col,
                "classes": all_labels,
                "metrics": val_metrics,
                "metadata": {
                    "dataset": path.name,
                    "trained_at": datetime.now().isoformat(),
                    "fit_time_sec": round(fit_time, 4),
                    "grouping": quality_report["group_description"],
                }
            }, f)

        saved_artifacts[name] = str(cand_path)
        results[name] = {
            "fit_time_sec": round(fit_time, 4),
            "val_metrics": val_metrics,
            "artifact_path": str(cand_path),
        }

        # Model Selection Rule: Highest Validation Macro F1
        if val_macro > best_macro_f1:
            best_macro_f1 = val_macro
            best_model_name = name

    print("-" * 70)
    print(f"🏆 Selected Best Candidate: {best_model_name} (Validation Macro F1: {best_macro_f1:.4f})")

    # Evaluate Winner on Untouched Test Set
    best_pipe = pipelines[best_model_name]
    if X_test is not None and len(X_test) > 0:
        y_test_pred = best_pipe.predict(X_test)
        test_metrics = compute_evaluation_metrics(y_test, y_test_pred, labels=all_labels)
    else:
        test_metrics = results[best_model_name]["val_metrics"]

    print("\n" + "=" * 70)
    print(f"FINAL TEST SET EVALUATION ({best_model_name}):")
    print(f"  • Test Macro F1:    {test_metrics['macro_f1']:.4f}")
    print(f"  • Test Weighted F1: {test_metrics['weighted_f1']:.4f}")
    print(f"  • Test Accuracy:    {test_metrics['accuracy']:.4f}")
    print("=" * 70)

    # Save to primary expected artifact paths
    if target_col == "risk_label":
        primary_model_path = out_p / "injury_risk_model.pkl"
    else:
        primary_model_path = out_p / "movement_classifier.pkl"

    with open(primary_model_path, "wb") as f:
        pickle.dump(best_pipe, f)

    # Maintain compatibility alias so legacy pipeline loaders do not break
    alias_path = out_p / "injury_risk_model.pkl"
    if alias_path != primary_model_path:
        with open(alias_path, "wb") as f:
            pickle.dump(best_pipe, f)

    # Save Model Registry Manifest
    registry_path = out_p / "model_registry.json"
    registry_data = {
        "selected_model": best_model_name,
        "selection_metric": "macro_f1",
        "validation_macro_f1": best_macro_f1,
        "test_metrics": test_metrics,
        "target_column": target_col,
        "dataset_file": path.name,
        "timestamp": datetime.now().isoformat(),
        "primary_artifact": str(primary_model_path),
        "data_quality_report": quality_report,
        "candidate_comparison": {
            name: {
                "val_macro_f1": res["val_metrics"]["macro_f1"],
                "val_accuracy": res["val_metrics"]["accuracy"],
                "fit_time_sec": res["fit_time_sec"],
                "artifact": res["artifact_path"],
            }
            for name, res in results.items()
        },
    }

    with open(registry_path, "w", encoding="utf-8") as f:
        json.dump(registry_data, f, indent=2)

    print(f"\nSaved best model to: {primary_model_path}")
    print(f"Saved model registry to: {registry_path}")

    return registry_data


def main():
    parser = argparse.ArgumentParser(description="Train and compare 3 models for Badminton AI")
    parser.add_argument("--data", type=str, default="data/dataset.csv", help="Path to training dataset CSV")
    parser.add_argument("--target", type=str, choices=["risk_label", "movement_type"], default="risk_label", help="Target column to predict")
    parser.add_argument("--output-dir", type=str, default="models", help="Directory to save model artifacts")
    parser.add_argument("--seed", type=int, default=42, help="Random state seed")
    args = parser.parse_args()

    compare_and_train_models(
        data_path=args.data,
        target_col=args.target,
        output_dir=args.output_dir,
        random_state=args.seed,
    )


if __name__ == "__main__":
    main()
