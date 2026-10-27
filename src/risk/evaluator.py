"""
Biomechanical injury-risk evaluator based on Rubric v0.3 with deterministic numerical scoring.

CRITICAL DIRECTIVE:
Under Rubric v0.3, numerical scores (0-100) and screening categories (LOW, MODERATE, HIGH)
are provisional engineering screening benchmarks based on published sports biomechanics literature.
They are NOT clinically validated injury probabilities or diagnostic medical claims.

SCISSOR_JUMP and FOOTWORK_SPLIT_STEP remain explicitly DEFERRED/UNASSESSED.
Low landmark quality or missing/invalid required measurements return UNASSESSED (never defaulting to LOW or 0 risk).
The UNCERTAIN rubric status is retained for legacy records; assessable evaluations return PROVISIONAL.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
import math

from .rubric import (
    RUBRIC_VERSION,
    RUBRIC_STATUS_PROVISIONAL,
    RUBRIC_STATUS_UNASSESSED,
    RUBRIC_STATUS_UNCERTAIN,
    ACTIVE_RUBRIC_CATEGORIES,
    DEFERRED_RUBRIC_CATEGORIES,
    CATEGORY_LOW,
    CATEGORY_MODERATE,
    CATEGORY_HIGH,
    CATEGORY_UNASSESSED,
    MAX_KNEE_POINTS,
    MAX_TRUNK_POINTS,
    MAX_ASYMMETRY_POINTS,
    SCORE_BAND_LOW_MAX,
    SCORE_BAND_MODERATE_MAX,
)
from src.classification.categories import (
    ACTIVE_RISK_RUBRIC_CATEGORIES,
    DEFERRED_RISK_RUBRIC_CATEGORIES,
)


@dataclass
class RiskAssessmentResult:
    risk_level: str  # New evaluations: PROVISIONAL or UNASSESSED; UNCERTAIN is legacy-only
    is_assessable: bool
    score: Optional[float] = None  # 0.0 to 100.0; None when UNASSESSED
    screening_category: str = CATEGORY_UNASSESSED  # LOW, MODERATE, HIGH, UNASSESSED
    rubric_version: str = RUBRIC_VERSION
    observations: List[str] = field(default_factory=list)
    contributing_indicators: List[Dict[str, Any]] = field(default_factory=list)
    category_bands: Dict[str, str] = field(default_factory=lambda: {
        "LOW": f"0.0 - {SCORE_BAND_LOW_MAX - 0.1:.1f}",
        "MODERATE": f"{SCORE_BAND_LOW_MAX:.1f} - {SCORE_BAND_MODERATE_MAX - 0.1:.1f}",
        "HIGH": f"{SCORE_BAND_MODERATE_MAX:.1f} - 100.0",
    })
    risk_color: str = "#6c757d"  # Default neutral gray
    status_message: str = ""
    missing_evidence: str = ""


def compute_knee_flexion_contribution(min_knee: float) -> Tuple[float, str]:
    """
    Computes knee flexion risk contribution (0.0 to 40.0 pts).
    Optimal cushioning window: [95.0, 115.0] deg -> 0.0 pts.
    Stiff deceleration (> 115.0 deg): up to 40.0 pts at 160.0 deg.
    Acute hyperflexion (< 95.0 deg): up to 40.0 pts at 60.0 deg.
    """
    if 95.0 <= min_knee <= 115.0:
        return 0.0, f"Optimal knee deceleration cushioning ({min_knee:.1f}° within 95°–115°)."
    elif min_knee > 115.0:
        pts = min(MAX_KNEE_POINTS, (min_knee - 115.0) * (MAX_KNEE_POINTS / 45.0))
        return round(pts, 1), f"Stiff knee deceleration ({min_knee:.1f}° > 115°); reduced shock cushioning."
    else:
        pts = min(MAX_KNEE_POINTS, (95.0 - min_knee) * (MAX_KNEE_POINTS / 35.0))
        return round(pts, 1), f"Acute knee flexion ({min_knee:.1f}° < 95°); elevated patellofemoral compressive stress."


def compute_torso_lean_contribution(trunk_max: float) -> Tuple[float, str]:
    """
    Computes torso lean risk contribution (0.0 to 35.0 pts).
    Optimal upright alignment: <= 15.0 deg -> 0.0 pts.
    Excessive lean (> 15.0 deg): up to 35.0 pts at 45.0 deg.
    """
    if trunk_max <= 15.0:
        return 0.0, f"Controlled upright torso alignment ({trunk_max:.1f}° <= 15°)."
    else:
        pts = min(MAX_TRUNK_POINTS, (trunk_max - 15.0) * (MAX_TRUNK_POINTS / 30.0))
        return round(pts, 1), f"Elevated torso lean ({trunk_max:.1f}° > 15°); center of mass displaced outside base."


def compute_asymmetry_contribution(asym_knee: float) -> Tuple[float, str]:
    """
    Computes bilateral knee asymmetry risk contribution (0.0 to 25.0 pts).
    Symmetrical / balanced mechanics: <= 10.0 deg -> 0.0 pts.
    Asymmetric impact / plant (> 10.0 deg): up to 25.0 pts at 45.0 deg.
    """
    if asym_knee <= 10.0:
        return 0.0, f"Balanced bilateral knee flexion ({asym_knee:.1f}° asymmetry <= 10°)."
    else:
        pts = min(MAX_ASYMMETRY_POINTS, (asym_knee - 10.0) * (MAX_ASYMMETRY_POINTS / 35.0))
        return round(pts, 1), f"Unbalanced bilateral knee asymmetry ({asym_knee:.1f}° > 10°); unequal shock distribution."


def determine_screening_category(score: float) -> Tuple[str, str]:
    """
    Maps numerical score [0, 100] to category and hex color badge.
    LOW: [0.0, 30.0) -> #28a745 (Green)
    MODERATE: [30.0, 60.0) -> #ffc107 (Amber)
    HIGH: [60.0, 100.0] -> #dc3545 (Red)
    """
    if score < SCORE_BAND_LOW_MAX:
        return CATEGORY_LOW, "#28a745"
    elif score < SCORE_BAND_MODERATE_MAX:
        return CATEGORY_MODERATE, "#ffc107"
    else:
        return CATEGORY_HIGH, "#dc3545"


def evaluate_movement_risk(
    movement_category: str,
    features: Dict[str, Any],
    landmark_quality: float = 1.0,
    lower_body_quality: Optional[float] = None,
) -> RiskAssessmentResult:
    """
    Evaluates biomechanical posture observations and computes a deterministic numerical
    screening score (0-100) and provisional category (LOW, MODERATE, HIGH) based on Rubric v0.3.
    """
    category = (movement_category or "").strip().upper()

    # Determine effective landmark quality for Rubric v0.3 (lower-body and trunk kinematics)
    eval_quality = lower_body_quality if lower_body_quality is not None else landmark_quality

    # 1. Landmark Quality Gate (Never default failed tracking to LOW or 0 risk)
    try:
        quality = float(eval_quality)
    except (TypeError, ValueError):
        quality = float("nan")
    if not math.isfinite(quality) or not 0.0 <= quality <= 1.0 or quality < 0.45:
        return RiskAssessmentResult(
            risk_level=RUBRIC_STATUS_UNASSESSED,
            is_assessable=False,
            score=None,
            screening_category=CATEGORY_UNASSESSED,
            observations=[
                f"Pose tracking quality was unavailable or low ({quality:.2f}; minimum 0.45). "
                "Insufficient keypoint confidence for reliable biomechanical assessment."
            ],
            risk_color="#6c757d",
            status_message="UNASSESSED (Low Landmark Quality)",
            missing_evidence="Sufficient keypoint visibility and tracking confidence across key lower-limb and trunk joints.",
        )

    # 2. Check for Deferred Categories under Rubric v0.3
    if category in DEFERRED_RUBRIC_CATEGORIES or category in DEFERRED_RISK_RUBRIC_CATEGORIES:
        return RiskAssessmentResult(
            risk_level=RUBRIC_STATUS_UNASSESSED,
            is_assessable=False,
            score=None,
            screening_category=CATEGORY_UNASSESSED,
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
            score=None,
            screening_category=CATEGORY_UNASSESSED,
            observations=[f"Movement '{category}' is not an active assessable category under Rubric v0.3."],
            risk_color="#6c757d",
            status_message="UNASSESSED (Non-Active Movement)",
            missing_evidence="Category inclusion in active rubric specification.",
        )

    # 4. Check for Missing Kinematic Measurements (Never default missing data to LOW or 0 risk)
    if not features:
        return RiskAssessmentResult(
            risk_level=RUBRIC_STATUS_UNASSESSED,
            is_assessable=False,
            score=None,
            screening_category=CATEGORY_UNASSESSED,
            observations=["No kinematic features available for this event."],
            risk_color="#6c757d",
            status_message="UNASSESSED (Missing Kinematic Measurements)",
            missing_evidence="Extracted joint angles and kinematic measurements.",
        )

    lk_min = features.get("left_knee_min")
    rk_min = features.get("right_knee_min")
    t_max = features.get("trunk_angle_max")

    try:
        lk_min, rk_min, t_max = float(lk_min), float(rk_min), float(t_max)
    except (TypeError, ValueError):
        lk_min = rk_min = t_max = float("nan")
    if (
        not all(math.isfinite(v) for v in (lk_min, rk_min, t_max))
        or not all(0.0 <= v <= 180.0 for v in (lk_min, rk_min, t_max))
    ):
        return RiskAssessmentResult(
            risk_level=RUBRIC_STATUS_UNASSESSED,
            is_assessable=False,
            score=None,
            screening_category=CATEGORY_UNASSESSED,
            observations=["Joint angle measurements are missing or invalid (non-finite or outside 0-180 degrees)."],
            risk_color="#6c757d",
            status_message="UNASSESSED (Incomplete Kinematics)",
            missing_evidence="Complete bilateral knee flexion and torso lean angle values.",
        )

    min_knee = min(float(lk_min), float(rk_min))
    asym_knee = abs(float(lk_min) - float(rk_min))
    trunk_max = float(t_max)

    # Compute deterministic points for each contributing indicator
    knee_pts, knee_exp = compute_knee_flexion_contribution(min_knee)
    trunk_pts, trunk_exp = compute_torso_lean_contribution(trunk_max)
    asym_pts, asym_exp = compute_asymmetry_contribution(asym_knee)

    raw_total = knee_pts + trunk_pts + asym_pts
    total_score = round(min(100.0, max(0.0, raw_total)), 1)
    screening_cat, cat_color = determine_screening_category(total_score)

    contributing_indicators = [
        {
            "name": "Knee Deceleration Flexion",
            "measured_value": round(min_knee, 1),
            "unit": "°",
            "points": knee_pts,
            "max_points": MAX_KNEE_POINTS,
            "direction_of_concern": "Stiff landing (>115°) or acute hyperflexion (<95°)",
            "explanation": knee_exp,
        },
        {
            "name": "Torso Lean Deviation",
            "measured_value": round(trunk_max, 1),
            "unit": "°",
            "points": trunk_pts,
            "max_points": MAX_TRUNK_POINTS,
            "direction_of_concern": "Excessive forward or lateral tilt (>15°)",
            "explanation": trunk_exp,
        },
        {
            "name": "Bilateral Knee Asymmetry",
            "measured_value": round(asym_knee, 1),
            "unit": "°",
            "points": asym_pts,
            "max_points": MAX_ASYMMETRY_POINTS,
            "direction_of_concern": "Unbalanced bilateral ground impact absorption (>10°)",
            "explanation": asym_exp,
        },
    ]

    obs: List[str] = []

    # 5. Extract Factual Biomechanical Observations
    if category == "LUNGE":
        obs.append(f"Minimum side-specific event knee angle: {min_knee:.1f}° (Left: {lk_min:.1f}°, Right: {rk_min:.1f}°). Lead leg is not identified.")
        obs.append(f"Difference between left/right event minima: {asym_knee:.1f}°; minima may occur at different frames.")
        obs.append(f"Maximum image-plane torso deviation from vertical: {trunk_max:.1f}°.")
    elif category == "SMASH_LANDING":
        obs.append(f"Minimum side-specific event knee angle: {min_knee:.1f}° (Left: {lk_min:.1f}°, Right: {rk_min:.1f}°); touchdown timing is not isolated.")
        obs.append(f"Difference between left/right event minima: {asym_knee:.1f}°; minima may occur at different frames.")
        obs.append(f"Maximum image-plane torso deviation from vertical: {trunk_max:.1f}°.")
    elif category == "CHANGE_OF_DIRECTION":
        obs.append(f"Minimum side-specific event knee angle: {min_knee:.1f}°; braking phase is not isolated.")
        obs.append(f"Maximum image-plane torso deviation from vertical: {trunk_max:.1f}°.")
        obs.append(f"Difference between left/right event minima: {asym_knee:.1f}°; minima may occur at different frames.")

    obs.append(f"Numerical Screening Score: {total_score:.1f} / 100 ({screening_cat} Risk Category). Breakdown: Knee {knee_pts:.1f}/40, Torso {trunk_pts:.1f}/35, Asymmetry {asym_pts:.1f}/25.")
    obs.append(
        "ℹ️ Notice: Numerical cutoffs and LOW/MODERATE/HIGH categories are provisional engineering screening benchmarks. "
        "Reported values are biomechanical screening indicators only, not clinically validated injury outcome predictions or medical diagnoses."
    )

    return RiskAssessmentResult(
        risk_level=RUBRIC_STATUS_PROVISIONAL,
        is_assessable=True,
        score=total_score,
        screening_category=screening_cat,
        observations=obs,
        contributing_indicators=contributing_indicators,
        risk_color=cat_color,
        status_message=f"PROVISIONAL (Unvalidated Cutoffs) — {screening_cat} ({total_score:.1f}/100)",
        missing_evidence="Clinical validation of H1 cluster-scoring hypothesis and empirical cutoffs for angle indicators in prospective injury-outcome trials.",
    )
