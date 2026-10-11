"""
Temporal aggregation of frame-level pose features across event windows.
Computes min/max joint angles, asymmetries, and visibility quality summaries.
"""

from typing import Dict, List, Any
import numpy as np


def aggregate_event_features(frame_features: List[Any]) -> Dict[str, float]:
    """
    Aggregates a sequence of frame-level pose features into an event-level tabular feature vector.

    Accepts a list of PoseFeatures objects or dictionaries containing:
    left_knee_angle, right_knee_angle, left_elbow_angle, right_elbow_angle, torso_lean, etc.
    """
    if not frame_features:
        return {
            "left_knee_min": 180.0,
            "left_knee_max": 180.0,
            "right_knee_min": 180.0,
            "right_knee_max": 180.0,
            "left_elbow_min": 180.0,
            "left_elbow_max": 180.0,
            "right_elbow_min": 180.0,
            "right_elbow_max": 180.0,
            "trunk_angle_min": 0.0,
            "trunk_angle_max": 0.0,
            "asymmetry_knee": 0.0,
            "asymmetry_elbow": 0.0,
            "mean_confidence": 0.0,
        }

    left_knees = []
    right_knees = []
    left_elbows = []
    right_elbows = []
    trunk_leans = []
    confidences = []

    for f in frame_features:
        if isinstance(f, dict):
            lk = f.get("left_knee_angle")
            rk = f.get("right_knee_angle")
            le = f.get("left_elbow_angle")
            re = f.get("right_elbow_angle")
            tl = f.get("torso_lean")
            conf = f.get("confidence", 1.0)
        else:
            lk = getattr(f, "left_knee_angle", None)
            rk = getattr(f, "right_knee_angle", None)
            le = getattr(f, "left_elbow_angle", None)
            re = getattr(f, "right_elbow_angle", None)
            tl = getattr(f, "torso_lean", None)
            conf = getattr(f, "confidence", 1.0)

        if lk is not None:
            left_knees.append(lk)
        if rk is not None:
            right_knees.append(rk)
        if le is not None:
            left_elbows.append(le)
        if re is not None:
            right_elbows.append(re)
        if tl is not None:
            trunk_leans.append(tl)
        if conf is not None:
            confidences.append(conf)

    lk_min = float(np.min(left_knees)) if left_knees else 180.0
    lk_max = float(np.max(left_knees)) if left_knees else 180.0
    rk_min = float(np.min(right_knees)) if right_knees else 180.0
    rk_max = float(np.max(right_knees)) if right_knees else 180.0

    le_min = float(np.min(left_elbows)) if left_elbows else 180.0
    le_max = float(np.max(left_elbows)) if left_elbows else 180.0
    re_min = float(np.min(right_elbows)) if right_elbows else 180.0
    re_max = float(np.max(right_elbows)) if right_elbows else 180.0

    t_min = float(np.min(trunk_leans)) if trunk_leans else 0.0
    t_max = float(np.max(trunk_leans)) if trunk_leans else 0.0

    mean_conf = float(np.mean(confidences)) if confidences else 0.0

    return {
        "left_knee_min": round(lk_min, 2),
        "left_knee_max": round(lk_max, 2),
        "right_knee_min": round(rk_min, 2),
        "right_knee_max": round(rk_max, 2),
        "left_elbow_min": round(le_min, 2),
        "left_elbow_max": round(le_max, 2),
        "right_elbow_min": round(re_min, 2),
        "right_elbow_max": round(re_max, 2),
        "trunk_angle_min": round(t_min, 2),
        "trunk_angle_max": round(t_max, 2),
        "asymmetry_knee": round(abs(lk_min - rk_min), 2),
        "asymmetry_elbow": round(abs(le_min - re_min), 2),
        "mean_confidence": round(mean_conf, 3),
    }
