"""
Structured Event Manifest definition and serialization for badminton movement analysis.
Manages event-level metadata, clip paths, key frames, predicted categories, and risk assessments.
"""

import json
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional


@dataclass
class MovementEventRecord:
    event_id: str
    video_id: str
    source_video: str
    start_time: float
    end_time: float
    start_frame: int
    end_frame: int
    movement_category: str  # LUNGE, SMASH_LANDING, SCISSOR_JUMP, CHANGE_OF_DIRECTION, FOOTWORK_SPLIT_STEP, UNCERTAIN
    localization_method: str = "legacy_unknown"  # candidate_detector, whole_video_fallback, legacy_unknown
    detection_status: str = "PROPOSED"  # PROPOSED, ACCEPTED, REJECTED, UNCERTAIN
    representative_frame: str = ""  # path to keyframe image
    representative_timestamp: Optional[float] = None  # source-video timestamp of selected image
    representative_frame_index: Optional[int] = None  # zero-based source-video frame index
    kinematic_peak_timestamp: Optional[float] = None  # source-video timestamp; None when unsupported
    kinematic_peak_frame_index: Optional[int] = None
    keyframe_selection_reason: str = ""
    clip_path: str = ""  # path to cut video clip
    pose_quality: float = 1.0
    feature_extraction_status: str = "SUCCESS"  # SUCCESS, PARTIAL, FAILED
    annotation_status: str = "AUTO_PROPOSED"  # AUTO_PROPOSED, HUMAN_VERIFIED, RECLASSIFIED
    label_source: str = "CANDIDATE_DETECTOR"  # CANDIDATE_DETECTOR, HUMAN_REVIEW, CLASSIFIER_MODEL, CLASSIFIER_HEURISTIC
    rubric_version: str = "v0.3"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    risk_level: str = "UNASSESSED"  # PROVISIONAL, UNASSESSED, UNCERTAIN
    risk_observations: List[str] = field(default_factory=list)
    risk_missing_evidence: str = ""
    risk_score: Optional[float] = None
    risk_category: str = "UNASSESSED"
    risk_contributions: List[Dict[str, Any]] = field(default_factory=list)
    features: Dict[str, Any] = field(default_factory=dict)
    probabilities: Optional[Dict[str, float]] = None
    prediction_source: str = "unavailable"  # trained_model, heuristic, unavailable
    player_id: Optional[str] = None
    lower_body_quality: Optional[float] = None
    detector_proposal: Optional[str] = None
    detector_metrics: Dict[str, Any] = field(default_factory=dict)
    classifier_prediction: Optional[str] = None
    classification_note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MovementEventRecord":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


def save_event_manifest(
    events: List[MovementEventRecord],
    output_path: str,
) -> None:
    """Saves a list of movement event records as a JSON manifest."""
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    records = [e.to_dict() for e in events]
    with open(p, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)


def load_event_manifest(manifest_path: str) -> List[MovementEventRecord]:
    """Loads movement event records from a JSON manifest file."""
    p = Path(manifest_path)
    if not p.exists():
        return []
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [MovementEventRecord.from_dict(item) for item in data]
