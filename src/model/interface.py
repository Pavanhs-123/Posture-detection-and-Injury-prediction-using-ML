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
        Predicts the risk label ('LOW', 'MODERATE', 'HIGH') given a dictionary of sequence features.
        """
        if self.model is None:
            raise ValueError("Model is not loaded. Load a model file first or use Demo Mode.")
            
        x = self._prepare_features(features)
        pred = self.model.predict(x)[0]
        return str(pred).upper()

    def predict_proba(self, features: dict) -> dict:
        """
        Predicts the class probabilities for 'LOW', 'MODERATE', and 'HIGH' risk.
        """
        if self.model is None:
            raise ValueError("Model is not loaded.")
            
        x = self._prepare_features(features)
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(x)[0]
            classes = self.model.classes_ if hasattr(self.model, "classes_") else self.classes_
            return {str(c).upper(): float(p) for c, p in zip(classes, probs)}
            
        # Default fallback if predict_proba is not available
        pred = self.predict(features)
        return {c: 1.0 if c == pred else 0.0 for c in self.classes_}

    def _prepare_features(self, features: dict) -> np.ndarray:
        """
        Aligns feature dictionary keys into a 2D numpy array for the model.
        """
        if self.feature_names:
            vector = [features.get(name, 0.0) for name in self.feature_names]
        else:
            # Fallback default feature column order
            keys = [
                "left_knee_min", "left_knee_max",
                "right_knee_min", "right_knee_max",
                "left_elbow_min", "left_elbow_max",
                "right_elbow_min", "right_elbow_max",
                "trunk_angle_min", "trunk_angle_max"
            ]
            vector = [features.get(k, 180.0 if "knee" in k or "elbow" in k else 0.0) for k in keys]
        return np.array([vector])
