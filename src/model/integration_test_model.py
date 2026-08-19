from .interface import InjuryRiskModel

class IntegrationTestModel(InjuryRiskModel):
    """
    Dummy/Integration Test Model used purely to verify integration plumbing and UI flows.
    This runs deterministic biomechanical threshold checks on the features.
    """
    def __init__(self):
        super().__init__()
        self.is_integration_test = True
        self.feature_names = [
            "left_knee_min", "left_knee_max",
            "right_knee_min", "right_knee_max",
            "left_elbow_min", "left_elbow_max",
            "right_elbow_min", "right_elbow_max",
            "trunk_angle_min", "trunk_angle_max"
        ]

    def load(self, model_path: str = None) -> None:
        # No-op, bypass loading from disk
        pass

    def predict(self, features: dict) -> str:
        """
        Determines risk label ('LOW', 'MODERATE', 'HIGH') based on biomechanical rules.
        """
        knee_min = min(features.get("left_knee_min", 180.0), features.get("right_knee_min", 180.0))
        trunk_max = features.get("trunk_angle_max", 0.0)

        # High risk threshold: knee bend is extremely deep/exaggerated (under 80 deg) 
        # or torso lean is severe (greater than 40 deg).
        if knee_min < 80.0 or trunk_max > 40.0:
            return "HIGH"
        # Moderate risk threshold: knee bend is moderately deep (80-110 deg) 
        # or torso lean is intermediate (25-40 deg).
        elif knee_min < 110.0 or trunk_max > 25.0:
            return "MODERATE"
        else:
            return "LOW"

    def predict_proba(self, features: dict) -> dict:
        """
        Generates simulated class probabilities matching the rule-based prediction.
        """
        pred = self.predict(features)
        if pred == "HIGH":
            return {"LOW": 0.1, "MODERATE": 0.2, "HIGH": 0.7}
        elif pred == "MODERATE":
            return {"LOW": 0.2, "MODERATE": 0.6, "HIGH": 0.2}
        else:
            return {"LOW": 0.8, "MODERATE": 0.1, "HIGH": 0.1}
