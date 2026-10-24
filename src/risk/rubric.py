"""
Biomechanical Risk Rubric v0.3 specification and status constants.

IMPORTANT ARCHITECTURAL DIRECTIVE:
Under Biomechanical Risk Rubric v0.3, the H1 cluster-scoring hypothesis remains unvalidated
and numerical indicator cutoffs remain intentionally unspecified pending empirical biomechanical trials.
No automated LOW, MODERATE, or HIGH risk decisions may be assigned by heuristics.
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
