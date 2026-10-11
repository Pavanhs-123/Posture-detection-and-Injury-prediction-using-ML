"""
Dedicated prediction service for movement classification.
Validates feature schemas, loads trained model pipelines from registry,
and predicts movement categories across the 5 target classes.

Enforces clear separation between genuine trained-model predictions and heuristic baselines.
Never generates simulated probabilities for heuristic outputs.
"""

import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import numpy as np

from .categories import (
    MOVEMENT_CATEGORIES,
    CATEGORY_METADATA,
)

EXPECTED_FEATURE_NAMES = [
    "left_knee_min",
    "left_knee_max",
    "right_knee_min",
    "right_knee_max",
    "left_elbow_min",
    "left_elbow_max",
    "right_elbow_min",
    "right_elbow_max",
    "trunk_angle_min",
    "trunk_angle_max",
]


@dataclass
class ClassificationResult:
    predicted_category: str  # LUNGE, SMASH_LANDING, SCISSOR_JUMP, CHANGE_OF_DIRECTION, FOOTWORK_SPLIT_STEP, UNCERTAIN
    prediction_source: str   # "trained_model", "heuristic", "unavailable"
    probabilities: Optional[Dict[str, float]]  # Only present when trained_model supports predict_proba; None for heuristic
    is_trained_model: bool
    status_message: str

    def __iter__(self):
        """Allows tuple unpacking (category, probabilities) for backward compatibility."""
        yield self.predicted_category
        yield self.probabilities


class MovementClassifier:
    """Predicts one of the 5 movement categories given event-level kinematic features."""

    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.classes_ = list(MOVEMENT_CATEGORIES)
        self.feature_names = list(EXPECTED_FEATURE_NAMES)
        self.model_name = "Heuristic Biomechanical Baseline"
        self.is_trained_model = False
        self.metadata = {}

        if model_path and os.path.exists(model_path):
            self.load(model_path)

    def load(self, model_path: str) -> None:
        """Loads a serialized model artifact and metadata."""
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Classifier model file not found: {model_path}")

        with open(path, "rb") as f:
            data = pickle.load(f)

        if isinstance(data, dict):
            self.model = data.get("model")
            self.feature_names = data.get("features", EXPECTED_FEATURE_NAMES)
            self.model_name = data.get("model_name", path.stem)
            self.metadata = data.get("metadata", {})
            if "classes" in data:
                self.classes_ = list(data["classes"])
        else:
            self.model = data
            self.model_name = path.stem

        self.is_trained_model = True

    def validate_features(self, features: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Checks whether input features dictionary is valid and non-empty."""
        if not features:
            return False, "Features dictionary is empty."
        return True, None

    def _prepare_vector(self, features: Dict[str, Any]) -> np.ndarray:
        vec = []
        for name in self.feature_names:
            val = features.get(name, 180.0 if "knee" in name or "elbow" in name else 0.0)
            vec.append(float(val) if val is not None else 0.0)
        return np.array(vec, dtype=np.float64).reshape(1, -1)

    def predict(self, features: Dict[str, Any]) -> ClassificationResult:
        """
        Predicts movement category.
        Returns ClassificationResult with prediction_source and probabilities.
        """
        valid, err = self.validate_features(features)
        if not valid:
            return ClassificationResult(
                predicted_category="UNCERTAIN",
                prediction_source="unavailable",
                probabilities=None,
                is_trained_model=False,
                status_message="Features dictionary empty or invalid.",
            )

        x = self._prepare_vector(features)

        if self.is_trained_model and self.model is not None:
            pred = self.model.predict(x)[0]
            pred_str = str(pred).strip().upper()

            probs = None
            if hasattr(self.model, "predict_proba"):
                try:
                    p_vals = self.model.predict_proba(x)[0]
                    m_classes = []
                    if hasattr(self.model, "classes_"):
                        m_classes = [str(c).upper() for c in self.model.classes_]
                    elif hasattr(self.model, "named_steps") and hasattr(
                        self.model.named_steps.get("classifier", None), "classes_"
                    ):
                        m_classes = [
                            str(c).upper()
                            for c in self.model.named_steps["classifier"].classes_
                        ]

                    probs = {}
                    for c, pv in zip(m_classes, p_vals):
                        probs[c] = round(float(pv), 3)

                    # Normalize if sum > 0
                    total_p = sum(probs.values())
                    if total_p > 0:
                        probs = {k: round(v / total_p, 3) for k, v in probs.items()}
                except Exception:
                    probs = None

            return ClassificationResult(
                predicted_category=pred_str,
                prediction_source="trained_model",
                probabilities=probs,
                is_trained_model=True,
                status_message=f"Model Inference: {self.model_name}",
            )

        # Fallback heuristic prediction when no trained model is loaded
        return self._heuristic_fallback_prediction(features)

    def _heuristic_fallback_prediction(
        self, features: Dict[str, Any]
    ) -> ClassificationResult:
        """
        Deterministic kinematic baseline rules for pipeline integration testing.
        Explicitly marked as uncalibrated heuristic; NEVER produces simulated probabilities.
        """
        lk_min = features.get("left_knee_min")
        rk_min = features.get("right_knee_min")
        trunk_max = features.get("trunk_angle_max")

        if lk_min is None or rk_min is None or trunk_max is None:
            return ClassificationResult(
                predicted_category="UNCERTAIN",
                prediction_source="heuristic",
                probabilities=None,
                is_trained_model=False,
                status_message="Missing kinematic angles for heuristic classification.",
            )

        min_knee = min(float(lk_min), float(rk_min))
        asym = abs(float(lk_min) - float(rk_min))
        trunk_deg = float(trunk_max)

        pred = "UNCERTAIN"
        reason = "Kinematic signals did not meet specific heuristic rules."

        if min_knee < 110.0 and asym > 20.0:
            pred = "LUNGE"
            reason = "Deep asymmetric knee flexion consistent with lunge plant."
        elif min_knee < 125.0 and trunk_deg < 20.0 and asym < 15.0:
            pred = "SMASH_LANDING"
            reason = "Bilateral knee cushioning and upright torso consistent with jump touchdown."
        elif trunk_deg > 25.0:
            pred = "CHANGE_OF_DIRECTION"
            reason = "Substantial torso tilt consistent with lateral deceleration cut."
        else:
            # Do NOT default to FOOTWORK_SPLIT_STEP or SCISSOR_JUMP
            # Those require dynamic temporal footwork/flight tracking
            pred = "UNCERTAIN"
            reason = "Summary angles insufficient to distinguish aerial/footwork mechanics. Marked UNCERTAIN."

        return ClassificationResult(
            predicted_category=pred,
            prediction_source="heuristic",
            probabilities=None,  # Never fabricate probabilities for heuristics
            is_trained_model=False,
            status_message=f"Heuristic Baseline: {reason}",
        )
