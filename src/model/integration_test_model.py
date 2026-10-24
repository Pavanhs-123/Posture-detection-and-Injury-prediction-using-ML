from .interface import InjuryRiskModel

class IntegrationTestModel(InjuryRiskModel):
    """
    Compatibility shim retained for old tests and imports. Rubric v0.3 disables
    automated LOW/MODERATE/HIGH risk classification, so it never assigns a risk.
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
        """Return an explicit unassessed status instead of an unsupported risk label."""
        return "UNASSESSED"

    def predict_proba(self, features: dict) -> dict:
        """No probabilities are available for an unassessed event."""
        return {}
