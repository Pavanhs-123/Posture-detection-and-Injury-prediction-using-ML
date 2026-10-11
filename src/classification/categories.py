"""
Movement categories definitions for badminton AI.
Defines the five target movement categories, their descriptions,
and their coverage status under Biomechanical Risk Rubric v0.3.
"""

from typing import List, Dict, Any

# The 5 primary movement categories
MOVEMENT_CATEGORIES: List[str] = [
    "LUNGE",
    "SMASH_LANDING",
    "SCISSOR_JUMP",
    "CHANGE_OF_DIRECTION",
    "FOOTWORK_SPLIT_STEP",
]

# Non-movement / Review statuses
NON_TARGET_STATUSES: List[str] = [
    "UNCERTAIN",
    "NO_EVENT",
]

# Active vs. Deferred under Biomechanical Risk Rubric v0.3
ACTIVE_RISK_RUBRIC_CATEGORIES: List[str] = [
    "LUNGE",
    "SMASH_LANDING",
    "CHANGE_OF_DIRECTION",
]

DEFERRED_RISK_RUBRIC_CATEGORIES: List[str] = [
    "SCISSOR_JUMP",
    "FOOTWORK_SPLIT_STEP",
]

CATEGORY_METADATA: Dict[str, Dict[str, Any]] = {
    "LUNGE": {
        "label": "Lunge Plant",
        "icon": "🦵",
        "description": "Front or lateral lunge with lead leg deceleration and weight transfer.",
        "risk_assessable": True,
        "rubric_status": "Active in Rubric v0.3",
    },
    "SMASH_LANDING": {
        "label": "Overhead Smash Landing",
        "icon": "🏸",
        "description": "Landing phase following high-velocity airborne overhead jump.",
        "risk_assessable": True,
        "rubric_status": "Active in Rubric v0.3",
    },
    "SCISSOR_JUMP": {
        "label": "Scissor Kick / Jump",
        "icon": "✂️",
        "description": "Mid-air hitch-kick or scissors leg rotation during rear-court stroke.",
        "risk_assessable": False,
        "rubric_status": "Deferred in Rubric v0.3 (Unassessed)",
    },
    "CHANGE_OF_DIRECTION": {
        "label": "Change of Direction",
        "icon": "🔄",
        "description": "Rapid deceleration, braking plant, and directional recovery push-off.",
        "risk_assessable": True,
        "rubric_status": "Active in Rubric v0.3",
    },
    "FOOTWORK_SPLIT_STEP": {
        "label": "Footwork Split-Step",
        "icon": "⚡",
        "description": "Preparatory bounce hop to pre-stretch leg musculature before reaction.",
        "risk_assessable": False,
        "rubric_status": "Deferred in Rubric v0.3 (Unassessed)",
    },
    "UNCERTAIN": {
        "label": "Uncertain Action",
        "icon": "❓",
        "description": "Ambiguous movement pattern requiring manual human inspection.",
        "risk_assessable": False,
        "rubric_status": "Unassessed",
    },
    "NO_EVENT": {
        "label": "No Significant Event",
        "icon": "⏹️",
        "description": "Stationary stance, idle walking, or non-badminton motion.",
        "risk_assessable": False,
        "rubric_status": "Excluded",
    },
}
