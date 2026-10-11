"""
Evaluation metrics computation for model selection and reporting.
Calculates Macro F1 (primary metric), Weighted F1, Per-class metrics, and Confusion Matrix.
"""

from typing import Dict, Any, List, Optional
import numpy as np
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    confusion_matrix,
    classification_report,
)


def compute_evaluation_metrics(
    y_true: List[str],
    y_pred: List[str],
    labels: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Computes a comprehensive evaluation dictionary.
    Primary selection metric: macro_f1.
    """
    if len(y_true) == 0:
        return {
            "macro_f1": 0.0,
            "weighted_f1": 0.0,
            "accuracy": 0.0,
            "support": 0,
            "per_class": {},
            "confusion_matrix": [],
            "labels": labels or [],
        }

    # Normalize case
    y_true_clean = [str(y).strip().upper() for y in y_true]
    y_pred_clean = [str(y).strip().upper() for y in y_pred]

    if labels is None:
        target_labels = sorted(list(set(y_true_clean) | set(y_pred_clean)))
    else:
        target_labels = [str(l).strip().upper() for l in labels]

    macro_f1 = float(f1_score(y_true_clean, y_pred_clean, labels=target_labels, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true_clean, y_pred_clean, labels=target_labels, average="weighted", zero_division=0))
    acc = float(accuracy_score(y_true_clean, y_pred_clean))

    cm = confusion_matrix(y_true_clean, y_pred_clean, labels=target_labels).tolist()

    precisions = precision_score(y_true_clean, y_pred_clean, labels=target_labels, average=None, zero_division=0)
    recalls = recall_score(y_true_clean, y_pred_clean, labels=target_labels, average=None, zero_division=0)
    f1s = f1_score(y_true_clean, y_pred_clean, labels=target_labels, average=None, zero_division=0)

    per_class = {}
    for idx, lbl in enumerate(target_labels):
        sup = int(sum(1 for y in y_true_clean if y == lbl))
        per_class[lbl] = {
            "precision": round(float(precisions[idx]), 3),
            "recall": round(float(recalls[idx]), 3),
            "f1": round(float(f1s[idx]), 3),
            "support": sup,
        }

    return {
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "accuracy": round(acc, 4),
        "support": len(y_true),
        "labels": target_labels,
        "per_class": per_class,
        "confusion_matrix": cm,
    }
