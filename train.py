import os
import pickle
import random
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

def parse_args():
    parser = argparse.ArgumentParser(description="Train the badminton injury risk prediction model")
    parser.add_argument("--csv", default="data/dataset.csv", help="Path to input dataset CSV")
    parser.add_argument("--output", default="models/injury_risk_model.pkl", help="Output path for trained model pickle")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Set random seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    csv_path = args.csv
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        print(f"\n[Error] Dataset file not found or empty at: {csv_path}")
        print("Please run 'python dataset_builder.py' to collect and label movement sequences first.")
        return

    # Load dataset
    df = pd.read_csv(csv_path)
    required_cols = [
        "sequence_id", "video_id", "movement_type",
        "left_knee_min", "left_knee_max", "right_knee_min", "right_knee_max",
        "left_elbow_min", "left_elbow_max", "right_elbow_min", "right_elbow_max",
        "trunk_angle_min", "trunk_angle_max", "risk_label"
    ]
    
    # Check for missing columns
    for col in required_cols:
        if col not in df.columns:
            print(f"[Error] Required column '{col}' is missing from the dataset.")
            return

    print(f"\nLoaded dataset with {len(df)} movement sequences.")
    
    # Drop rows with missing values in key columns
    df = df.dropna(subset=required_cols)
    if len(df) == 0:
        print("[Error] No valid rows left after dropping missing values.")
        return

    # Filter out any unapproved/malformed rows
    valid_risks = {"LOW", "MODERATE", "HIGH"}
    valid_risk_mask = df["risk_label"].str.upper().isin(valid_risks)
    if not valid_risk_mask.all():
        invalid_cnt = (~valid_risk_mask).sum()
        print(f"[Warning] Dropped {invalid_cnt} row(s) with invalid risk labels (must be LOW, MODERATE, or HIGH).")
        df = df[valid_risk_mask]
        if len(df) == 0:
            print("[Error] No valid rows left after filtering invalid risk labels.")
            return

    # Target features
    feature_cols = [
        "left_knee_min", "left_knee_max", "right_knee_min", "right_knee_max",
        "left_elbow_min", "left_elbow_max", "right_elbow_min", "right_elbow_max",
        "trunk_angle_min", "trunk_angle_max"
    ]

    # Report class & movement distribution
    print("\n" + "=" * 50)
    print("DATASET COMPOSITION & DISTRIBUTION")
    print("=" * 50)
    class_counts = df["risk_label"].str.upper().value_counts()
    print("Injury-Risk Class Distribution:")
    for cls_name, cnt in class_counts.items():
        print(f"  {cls_name:<10}: {cnt:3d} sample(s) ({cnt / len(df) * 100:.1f}%)")

    movement_counts = df["movement_type"].value_counts()
    print("\nMovement Class Distribution:")
    for mov_name, cnt in movement_counts.items():
        print(f"  {mov_name:<22}: {cnt:3d} sample(s) ({cnt / len(df) * 100:.1f}%)")
    print("=" * 50)

    if len(df) < 30:
        print("\n⚠️  [DATASET SIZE LIMITATION ADVISORY]")
        print(f"The training dataset contains only {len(df)} sequences.")
        print("Statistical metrics (precision, recall, F1) on very small samples (< 30-50 per class) have high variance.")
        print("Note: This trained model is an engineering prototype for biomechanical screening,")
        print("NOT a clinically certified diagnostic tool.")

    X = df[feature_cols]
    y = df["risk_label"].str.upper()
    groups = df["video_id"]

    # 1. GROUP-BASED SPLITTING (Preventing source video leakage)
    unique_videos = list(groups.unique())
    random.shuffle(unique_videos)
    n_videos = len(unique_videos)

    if n_videos < 3 or len(df) < 6:
        print(f"\n[Notice] Small sample pool ({n_videos} video(s), {len(df)} sequence(s)).")
        # Check if stratification is feasible
        can_stratify = (y.value_counts().min() >= 2) and (len(y.unique()) > 1) and (len(y) >= 6)
        strat_arg = y if can_stratify else None

        if len(df) >= 6:
            X_train_val, X_test, y_train_val, y_test = train_test_split(
                X, y, test_size=0.15, random_state=args.seed, stratify=strat_arg
            )
            can_strat_val = (y_train_val.value_counts().min() >= 2) and (len(y_train_val.unique()) > 1) if strat_arg is not None else False
            strat_val_arg = y_train_val if can_strat_val else None
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_val, y_train_val, test_size=0.18, random_state=args.seed, stratify=strat_val_arg
            )
        else:
            # Under 6 samples: train on all data; validation and test sets remain empty
            print("Dataset has fewer than 6 samples. Training on all samples without validation split.")
            X_train, y_train = X, y
            X_val, y_val = pd.DataFrame(columns=feature_cols), pd.Series(dtype=str)
            X_test, y_test = pd.DataFrame(columns=feature_cols), pd.Series(dtype=str)
    else:
        # Grouped video split ensuring every split has at least 1 video
        if n_videos == 3:
            train_vids = [unique_videos[0]]
            val_vids = [unique_videos[1]]
            test_vids = [unique_videos[2]]
        else:
            train_idx = max(1, int(0.70 * n_videos))
            val_idx = max(train_idx + 1, int(0.85 * n_videos))
            if val_idx >= n_videos:
                val_idx = n_videos - 1
            train_vids = unique_videos[:train_idx]
            val_vids = unique_videos[train_idx:val_idx]
            test_vids = unique_videos[val_idx:]

        print(f"Grouped split by video: Train={len(train_vids)}, Val={len(val_vids)}, Test={len(test_vids)}")
        
        train_mask = groups.isin(train_vids)
        val_mask = groups.isin(val_vids)
        test_mask = groups.isin(test_vids)
        
        X_train, y_train = X[train_mask], y[train_mask]
        X_val, y_val = X[val_mask], y[val_mask]
        X_test, y_test = X[test_mask], y[test_mask]

    print(f"Sample count - Train: {len(X_train)}, Validation: {len(X_val)}, Test: {len(X_test)}")
    if len(X_train) == 0:
        print("[Error] Train set has 0 samples. Cannot proceed with model training.")
        return

    # Check for single-class training
    unique_train_classes = list(np.unique(y_train))
    if len(unique_train_classes) < 2:
        print(f"\n[Warning] Training set contains only 1 unique class: {unique_train_classes}.")
        print("Classification decision boundaries require multiple classes to generalize effectively.")

    # 2. DEFINE TRAINING PIPELINE
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(
            n_estimators=100,
            max_depth=6,
            min_samples_split=3,
            random_state=args.seed,
            class_weight="balanced" if len(unique_train_classes) > 1 else None
        ))
    ])

    print("\nTraining Random Forest model pipeline...")
    pipeline.fit(X_train, y_train)

    # 3. EVALUATION
    if len(X_val) > 0 and len(y_val) > 0:
        val_preds = pipeline.predict(X_val)
        print("\n" + "=" * 50)
        print("VALIDATION SET PERFORMANCE")
        print("=" * 50)
        report_labels = sorted(list(valid_risks))
        print(classification_report(y_val, val_preds, labels=report_labels, zero_division=0))
    else:
        print("\n[Notice] Validation set has 0 samples; skipping validation evaluation.")

    if len(X_test) > 0 and len(y_test) > 0:
        test_preds = pipeline.predict(X_test)
        print("\n" + "=" * 50)
        print("TEST SET PERFORMANCE")
        print("=" * 50)
        print(f"Accuracy: {accuracy_score(y_test, test_preds):.4f}")
        report_labels = sorted(list(valid_risks))
        print(classification_report(y_test, test_preds, labels=report_labels, zero_division=0))
        print(f"Confusion Matrix (labels: {report_labels}):")
        print(confusion_matrix(y_test, test_preds, labels=report_labels))
        print("=" * 50)
    else:
        print("[Notice] Test set has 0 samples; skipping test evaluation.")

    # 4. SAVE MODEL
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save the pipeline and feature list
    save_data = {
        "model": pipeline,
        "features": feature_cols
    }
    
    with open(output_path, "wb") as f:
        pickle.dump(save_data, f)
        
    print(f"\nTrained model pipeline saved successfully to: {output_path}")
    print("You can now run the web interface, which will automatically detect and load this model.")

if __name__ == "__main__":
    main()
