"""
Biomechanical injury-risk evaluator based strictly on Rubric v0.3.

CRITICAL DIRECTIVE:
Under Rubric v0.3, numerical indicator cutoffs remain unspecified and the H1 cluster-scoring
hypothesis is unvalidated. Consequently, this evaluator outputs raw kinematic observations
and marks assessable active movements as PROVISIONAL rather than assigning unvalidated
LOW, MODERATE, or HIGH decisions.

SCISSOR_JUMP and FOOTWORK_SPLIT_STEP remain explicitly DEFERRED/UNASSESSED.
Low landmark quality or missing measurements result in UNCERTAIN (never defaulting to LOW).
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

from .rubric import (
    RUBRIC_VERSION,
    RUBRIC_STATUS_PROVISIONAL,
    RUBRIC_STATUS_UNASSESSED,
    RUBRIC_STATUS_UNCERTAIN,
    ACTIVE_RUBRIC_CATEGORIES,
    DEFERRED_RUBRIC_CATEGORIES,
)
from src.classification.categories import (
    ACTIVE_RISK_RUBRIC_CATEGORIES,
    DEFERRED_RISK_RUBRIC_CATEGORIES,
)


@dataclass
class RiskAssessmentResult:
    risk_level: str  # PROVISIONAL, UNASSESSED, UNCERTAIN
    is_assessable: bool
    rubric_version: str = RUBRIC_VERSION
    observations: List[str] = field(default_factory=list)
    risk_color: str = "#6c757d"  # Default neutral gray
    status_message: str = ""
    missing_evidence: str = ""


def evaluate_movement_risk(
    movement_category: str,
    features: Dict[str, Any],
    landmark_quality: float = 1.0,
) -> RiskAssessmentResult:
    """
    Evaluates biomechanical posture observations based strictly on Rubric v0.3.
    Does not assign arbitrary LOW/MODERATE/HIGH decisions.
    """
    category = (movement_category or "").strip().upper()

    # 1. Landmark Quality Gate (Never default failed tracking to LOW)
    if landmark_quality < 0.45:
        return RiskAssessmentResult(
            risk_level=RUBRIC_STATUS_UNCERTAIN,
            is_assessable=False,
            observations=[
                f"Pose tracking quality was low ({landmark_quality:.2f} < 0.45). "
                "Insufficient keypoint confidence for reliable biomechanical assessment."
            ],
            risk_color="#6c757d",
            status_message="UNCERTAIN (Low Landmark Quality)",
            missing_evidence="Sufficient keypoint visibility and tracking confidence across key lower-limb and trunk joints.",
        )

    # 2. Check for Deferred Categories under Rubric v0.3
    if category in DEFERRED_RUBRIC_CATEGORIES or category in DEFERRED_RISK_RUBRIC_CATEGORIES:
        return RiskAssessmentResult(
            risk_level=RUBRIC_STATUS_UNASSESSED,
            is_assessable=False,
            observations=[
                f"Movement '{category}' is a recognized badminton movement class, "
                "but its injury-risk scoring criteria are deferred under Rubric v0.3.",
                "Risk scoring is not clinically calibrated for this category. Do not infer automatic risk.",
            ],
            risk_color="#17a2b8",  # Info cyan
            status_message="UNASSESSED (Rubric v0.3 Deferred)",
            missing_evidence=f"Validated biomechanical screening criteria and injury indicators for '{category}'.",
        )

    # 3. Check for Unrecognized / Non-active Categories
    if category not in ACTIVE_RUBRIC_CATEGORIES and category not in ACTIVE_RISK_RUBRIC_CATEGORIES:
        return RiskAssessmentResult(
            risk_level=RUBRIC_STATUS_UNASSESSED,
            is_assessable=False,
            observations=[f"Movement '{category}' is not an active assessable category under Rubric v0.3."],
            risk_color="#6c757d",
            status_message="UNASSESSED (Non-Active Movement)",
            missing_evidence="Category inclusion in active rubric specification.",
        )

    # 4. Check for Missing Kinematic Measurements (Never default missing data to LOW)
    if not features:
        return RiskAssessmentResult(
            risk_level=RUBRIC_STATUS_UNCERTAIN,
            is_assessable=False,
            observations=["No kinematic features available for this event."],
            risk_color="#6c757d",
            status_message="UNCERTAIN (Missing Kinematic Measurements)",
            missing_evidence="Extracted joint angles and kinematic measurements.",
        )

    lk_min = features.get("left_knee_min")
    rk_min = features.get("right_knee_min")
    t_max = features.get("trunk_angle_max")

    if lk_min is None or rk_min is None or t_max is None:
        return RiskAssessmentResult(
            risk_level=RUBRIC_STATUS_UNCERTAIN,
            is_assessable=False,
            observations=["Incomplete joint angle measurements (missing knee flexion or torso lean)."],
            risk_color="#6c757d",
            status_message="UNCERTAIN (Incomplete Kinematics)",
            missing_evidence="Complete bilateral knee flexion and torso lean angle values.",
        )

    min_knee = min(float(lk_min), float(rk_min))
    asym_knee = abs(float(lk_min) - float(rk_min))
    trunk_max = float(t_max)

    obs: List[str] = []

    # 5. Extract Factual Biomechanical Observations (No invented risk cutoffs)
    if category == "LUNGE":
        obs.append(f"• Measured Lead Knee Flexion: {min_knee:.1f}° (Left: {float(lk_min):.1f}°, Right: {float(rk_min):.1f}°).")
        obs.append(f"• Measured Bilateral Knee Difference: {asym_knee:.1f}°.")
        obs.append(f"• Measured Torso Forward Lean: {trunk_max:.1f}°.")
    elif category == "SMASH_LANDING":
        obs.append(f"• Measured Touchdown Knee Flexion: {min_knee:.1f}° (Left: {float(lk_min):.1f}°, Right: {float(rk_min):.1f}°).")
        obs.append(f"• Measured Bilateral Touchdown Asymmetry: {asym_knee:.1f}°.")
        obs.append(f"• Measured Trunk Deviation: {trunk_max:.1f}°.")
    elif category == "CHANGE_OF_DIRECTION":
        obs.append(f"• Measured Deceleration Knee Angle: {min_knee:.1f}°.")
        obs.append(f"• Measured Lateral Trunk Deviation: {trunk_max:.1f}°.")
        obs.append(f"• Bilateral Knee Flexion Delta: {asym_knee:.1f}°.")

    obs.append(
        "ℹ️ Notice: Numerical cutoffs for Rubric v0.3 remain unspecified pending empirical sports medicine trials. "
        "Reported values are biomechanical observations only; automated LOW/MODERATE/HIGH risk classification is disabled."
    )

    return RiskAssessmentResult(
        risk_level=RUBRIC_STATUS_PROVISIONAL,
        is_assessable=True,
        observations=obs,
        risk_color="#ffc107",  # Warning amber for provisional
        status_message="PROVISIONAL (Unvalidated Cutoffs)",
        missing_evidence="Clinical validation of H1 cluster-scoring hypothesis and empirical cutoffs for angle indicators.",
    )
