"""
Biomechanical Risk Rubric v0.3 specification, status constants, and scoring thresholds.

IMPORTANT ARCHITECTURAL DIRECTIVE:
Under Biomechanical Risk Rubric v0.3, numerical scores (0-100) and screening categories
(LOW, MODERATE, HIGH) are provisional engineering screening benchmarks derived from published
sports science kinematics. They are NOT certified medical diagnoses or clinical injury outcome predictions.
"""

from typing import List

RUBRIC_VERSION = "v0.3"

# Official Rubric v0.3 Assessment Statuses
RUBRIC_STATUS_PROVISIONAL = "PROVISIONAL"
RUBRIC_STATUS_UNASSESSED = "UNASSESSED"
RUBRIC_STATUS_UNCERTAIN = "UNCERTAIN"

VALID_RISK_STATUSES: List[str] = [
    RUBRIC_STATUS_PROVISIONAL,
    RUBRIC_STATUS_UNASSESSED,
    RUBRIC_STATUS_UNCERTAIN,
]

# Active categories eligible for kinematic observations under Rubric v0.3
ACTIVE_RUBRIC_CATEGORIES: List[str] = [
    "LUNGE",
    "SMASH_LANDING",
    "CHANGE_OF_DIRECTION",
]

# Deferred categories explicitly unassessed for injury risk in Rubric v0.3
DEFERRED_RUBRIC_CATEGORIES: List[str] = [
    "SCISSOR_JUMP",
    "FOOTWORK_SPLIT_STEP",
]

# Numerical Screening Categories
CATEGORY_LOW = "LOW"
CATEGORY_MODERATE = "MODERATE"
CATEGORY_HIGH = "HIGH"
CATEGORY_UNASSESSED = "UNASSESSED"

SCREENING_CATEGORIES: List[str] = [
    CATEGORY_LOW,
    CATEGORY_MODERATE,
    CATEGORY_HIGH,
    CATEGORY_UNASSESSED,
]

# Indicator Weight Capacities (Total = 100.0 max)
MAX_KNEE_POINTS = 40.0
MAX_TRUNK_POINTS = 35.0
MAX_ASYMMETRY_POINTS = 25.0

# Provisional Engineering Score Category Boundaries
SCORE_BAND_LOW_MAX = 30.0       # [0.0, 30.0) -> LOW
SCORE_BAND_MODERATE_MAX = 60.0  # [30.0, 60.0) -> MODERATE
                                # [60.0, 100.0] -> HIGH
