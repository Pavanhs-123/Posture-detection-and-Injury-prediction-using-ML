from dataclasses import dataclass
from collections import deque


@dataclass(frozen=True)
class PoseSample:
    knee_angle: float | None
    elbow_angle: float | None


@dataclass(frozen=True)
class RiskAssessment:
    score: int
    level: str
    feedback: str


class InjuryRiskPredictor:
    def __init__(self, window_size=15):
        self._scores = deque(maxlen=window_size)

    def assess(self, sample: PoseSample) -> RiskAssessment:
        score = 0
        reasons = []

        if sample.knee_angle is not None:
            if sample.knee_angle < 90:
                score += 45
                reasons.append("very deep knee bend")
            elif sample.knee_angle < 110:
                score += 25
                reasons.append("knee bend is still quite sharp")
            elif sample.knee_angle < 140:
                score += 10
                reasons.append("moderate knee load")

        if sample.elbow_angle is not None:
            if sample.elbow_angle < 60:
                score += 25
                reasons.append("upper-limb position is highly flexed")
            elif sample.elbow_angle < 95:
                score += 15
                reasons.append("upper-limb position is moderately flexed")

        if score == 0:
            reasons.append("pose looks within a safer range")

        self._scores.append(score)
        smoothed_score = round(sum(self._scores) / len(self._scores))
        final_score = max(score, smoothed_score)

        if final_score >= 60:
            level = "HIGH"
        elif final_score >= 30:
            level = "MEDIUM"
        else:
            level = "LOW"

        if level == "HIGH":
            feedback = "Reduce load immediately and correct the movement pattern."
        elif level == "MEDIUM":
            feedback = "Monitor form and adjust posture before repeating."
        else:
            feedback = "Movement currently looks acceptable."

        if reasons:
            feedback = f"{feedback} {'; '.join(reasons)}."

        return RiskAssessment(score=final_score, level=level, feedback=feedback)