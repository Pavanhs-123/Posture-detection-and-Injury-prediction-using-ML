import os
import pickle
import random
import argparse
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

    print(f"Loaded dataset with {len(df)} movement sequences.")
    
    # Drop rows with missing values in key columns
    df = df.dropna(subset=required_cols)
    if len(df) == 0:
        print("[Error] No valid rows left after dropping missing values.")
        return

    # Target features
    feature_cols = [
        "left_knee_min", "left_knee_max", "right_knee_min", "right_knee_max",
        "left_elbow_min", "left_elbow_max", "right_elbow_min", "right_elbow_max",
        "trunk_angle_min", "trunk_angle_max"
    ]
    
    X = df[feature_cols]
    y = df["risk_label"].str.upper()
    groups = df["video_id"]

    # 1. GROUP-BASED SPLITTING (No frame or sequence leakage from the same video source)
    unique_videos = list(groups.unique())
    random.shuffle(unique_videos)
    
    # Target split: 70% Train, 15% Val, 15% Test
    n_videos = len(unique_videos)
    if n_videos < 3:
        print(f"\n[Warning] Very few unique video sources ({n_videos}). Splitting at sequence-level instead.")
        # Fallback to random sequence split if video sources are extremely limited
        X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.15, random_state=args.seed, stratify=y)
        X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.176, random_state=args.seed, stratify=y_train_val)
    else:
        train_idx = int(0.70 * n_videos)
        val_idx = int(0.85 * n_videos)
        
        train_vids = unique_videos[:max(1, train_idx)]
        val_vids = unique_videos[max(1, train_idx):max(2, val_idx)]
        test_vids = unique_videos[max(2, val_idx):]
        
        print(f"Split video sources: Train={len(train_vids)}, Val={len(val_vids)}, Test={len(test_vids)}")
        
        train_mask = groups.isin(train_vids)
        val_mask = groups.isin(val_vids)
        test_mask = groups.isin(test_vids)
        
        X_train, y_train = X[train_mask], y[train_mask]
        X_val, y_val = X[val_mask], y[val_mask]
        X_test, y_test = X[test_mask], y[test_mask]

    print(f"Samples count - Train: {len(X_train)}, Validation: {len(X_val)}, Test: {len(X_test)}")
    if len(X_train) == 0:
        print("[Error] Train set has 0 samples. Please add more sequences.")
        return

    # 2. DEFINE TRAINING PIPELINE
    # Random Forest is simple, stable and runs well on CPU locally
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(
            n_estimators=100,
            max_depth=6,
            min_samples_split=3,
            random_state=args.seed,
            class_weight="balanced" # Handles class imbalance in risk labels
        ))
    ])

    print("\nTraining Random Forest model pipeline...")
    pipeline.fit(X_train, y_train)

    # 3. EVALUATION
    # Validation evaluation
    val_preds = pipeline.predict(X_val)
    print("\n" + "="*50)
    print("VALIDATION SET PERFORMANCE")
    print(f"Accuracy: {accuracy_score(y_val, val_preds):.4f}")
    if len(y_val.unique()) > 0:
        print(classification_report(y_val, val_preds, zero_division=0))
    
    # Test evaluation
    test_preds = pipeline.predict(X_test)
    print("="*50)
    print("TEST SET PERFORMANCE")
    print(f"Accuracy: {accuracy_score(y_test, test_preds):.4f}")
    if len(y_test.unique()) > 0:
        print(classification_report(y_test, test_preds, zero_division=0))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, test_preds))
    print("="*50)

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
