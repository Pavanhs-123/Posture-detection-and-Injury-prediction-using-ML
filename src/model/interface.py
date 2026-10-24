import os
import pickle
import numpy as np


class InjuryRiskModel:
    """
    Abstraction layer for the injury-risk prediction model.
    Enables loading model pipelines and making predictions.
    """
    def __init__(self):
        self.model = None
        self.feature_names = None
        self.classes_ = ["LOW", "MODERATE", "HIGH"]
        self.is_integration_test = False

    def load(self, model_path: str) -> None:
        """
        Loads a serialized model pipeline or dictionary from path.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")
            
        with open(model_path, "rb") as f:
            data = pickle.load(f)
            
        if isinstance(data, dict):
            self.model = data.get("model")
            self.feature_names = data.get("features")
        else:
            self.model = data
            self.feature_names = None
            
        self.is_integration_test = False

    def predict(self, features: dict) -> str:
        """
        Risk classification is disabled under Rubric v0.3 pending credible validation.
        """
        raise RuntimeError("Automated LOW/MODERATE/HIGH injury-risk classification is disabled under Rubric v0.3.")

    def predict_proba(self, features: dict) -> dict:
        """
        Risk probability output is disabled under Rubric v0.3 pending credible validation.
        """
        raise RuntimeError("Automated LOW/MODERATE/HIGH injury-risk probabilities are disabled under Rubric v0.3.")

    def _prepare_features(self, features: dict) -> np.ndarray:
        """
        Align feature values without inventing missing or invalid measurements.
        """
        if not isinstance(features, dict):
            raise TypeError("Model features must be provided as a dictionary.")

        keys = self.feature_names or [
                "left_knee_min", "left_knee_max",
                "right_knee_min", "right_knee_max",
                "left_elbow_min", "left_elbow_max",
                "right_elbow_min", "right_elbow_max",
                "trunk_angle_min", "trunk_angle_max"
            ]
        missing = [name for name in keys if name not in features]
        if missing:
            raise ValueError(f"Missing required model feature(s): {', '.join(missing)}")

        vector = []
        for name in keys:
            value = features[name]
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
                raise TypeError(f"Model feature '{name}' must be a finite numeric measurement.")
            value = float(value)
            if not np.isfinite(value):
                raise ValueError(f"Model feature '{name}' must be finite; got {value!r}.")
            vector.append(value)

        return np.asarray([vector], dtype=float)
