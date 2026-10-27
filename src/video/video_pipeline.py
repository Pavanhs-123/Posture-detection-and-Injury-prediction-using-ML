"""
End-to-end video-to-movement-clips processing pipeline.
Performs video validation, frame extraction, pose estimation, feature calculation,
event candidate detection, 5-class movement classification, clip/keyframe extraction,
rubric v0.3 risk evaluation, and manifest generation.
"""

import json
import os
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
import cv2
import numpy as np

from .validator import validate_video
from .clip_extractor import extract_subclip, extract_keyframe
from src.pose.estimator import PoseEstimator, FramePoseResult
from src.pose.overlay import draw_skeleton_overlay
from src.features.angles_2d import (
    get_knee_angle_2d,
    get_elbow_angle_2d,
    get_torso_lean_2d,
)
from src.features.world_3d import (
    get_knee_angle_3d,
    get_elbow_angle_3d,
    get_torso_lean_3d,
)
from src.detection.candidate_detector import (
    MovementCandidateDetector,
    DetectorConfig,
)
from src.detection.manifest import (
    MovementEventRecord,
    save_event_manifest,
)
from src.classification.predictor import MovementClassifier
from src.risk.evaluator import evaluate_movement_risk

MIN_RELIABLE_PEAK_QUALITY = 0.45
REVIEW_WINDOW_CONTEXT_SECONDS = 0.5


def _aggregate_pose_signals(signals: List[Dict[str, Any]]) -> Dict[str, Optional[float]]:
    """Aggregate finite pose measurements only; no-pose placeholders are not observations."""
    pose_signals = [s for s in signals if s.get("has_pose")]
    mappings = {
        "left_knee_min": "feature_knee_l", "left_knee_max": "feature_knee_l",
        "right_knee_min": "feature_knee_r", "right_knee_max": "feature_knee_r",
        "left_elbow_min": "feature_elbow_l", "left_elbow_max": "feature_elbow_l",
        "right_elbow_min": "feature_elbow_r", "right_elbow_max": "feature_elbow_r",
        "trunk_angle_min": "feature_torso_lean", "trunk_angle_max": "feature_torso_lean",
    }
    result: Dict[str, Optional[float]] = {}
    for output_name, signal_name in mappings.items():
        values = []
        for signal in pose_signals:
            try:
                value = float(signal[signal_name])
            except (KeyError, TypeError, ValueError):
                continue
            if np.isfinite(value):
                values.append(value)
        if not values:
            result[output_name] = None
        elif output_name.endswith("_min"):
            result[output_name] = round(float(np.min(values)), 1)
        else:
            result[output_name] = round(float(np.max(values)), 1)
    return result


def _resolve_event_classification(
    candidate_category: Optional[str],
    prediction: Any,
    classifier_is_trained: bool,
) -> Dict[str, str]:
    """Keep the final event label and classifier suggestion separately auditable."""
    detector_proposal = (candidate_category or "").strip().upper()
    classifier_prediction = str(
        getattr(prediction, "predicted_category", "UNCERTAIN") or "UNCERTAIN"
    ).strip().upper()

    if detector_proposal == "UNCERTAIN":
        return {
            "movement_category": "UNCERTAIN",
            "label_source": "CANDIDATE_DETECTOR",
            "classification_note": (
                "The detector marked this event ambiguous. The classifier suggestion and "
                "probabilities are shown for review, but do not resolve detector uncertainty."
            ),
        }

    if classifier_is_trained:
        if classifier_prediction == "UNCERTAIN":
            note = "The trained classifier could not assign an active movement class to this event."
            gate_reason = getattr(prediction, "status_message", "")
            if gate_reason:
                note += f" Quality gate: {gate_reason}"
        else:
            note = "Final event class follows the trained classifier prediction."
        return {
            "movement_category": classifier_prediction,
            "label_source": "CLASSIFIER_MODEL",
            "classification_note": note,
        }

    if detector_proposal:
        return {
            "movement_category": detector_proposal,
            "label_source": "CANDIDATE_DETECTOR",
            "classification_note": "No trained classifier was loaded; the detector proposal is shown.",
        }

    prediction_source = getattr(prediction, "prediction_source", "heuristic")
    return {
        "movement_category": classifier_prediction,
        "label_source": (
            "CLASSIFIER_HEURISTIC" if prediction_source == "heuristic" else "CLASSIFIER_MODEL"
        ),
        "classification_note": "Final event class follows the available classifier prediction.",
    }


def _confidence_filtered_measurement(value: Optional[float], confidence: Any, threshold: float = 0.4) -> Optional[float]:
    """Discard missing, non-finite, or low-visibility joint measurements."""
    try:
        value = float(value)
        confidence = float(confidence)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(value) or not np.isfinite(confidence) or confidence < threshold:
        return None
    return value


def _filter_candidates_by_pose_coverage(candidates: List[Any], signals: List[Dict[str, Any]], fps: float, minimum: float = 0.7) -> List[Any]:
    """Reject detector windows dominated by frames without a detected pose."""
    retained = []
    for candidate in candidates:
        start = max(0, int(round(candidate.start_time * fps)))
        end = min(len(signals), int(round(candidate.end_time * fps)))
        window = signals[start:end]
        if window and sum(bool(frame.get("has_pose")) for frame in window) / len(window) >= minimum:
            retained.append(candidate)
    return retained


def _select_representative_frame(
    signals: List[Dict[str, Any]],
    start_time: float,
    end_time: float,
    fps: float,
    total_frames: int,
    min_quality: float = 0.45,
    context_seconds: float = 0.12,
) -> Dict[str, Any]:
    """Choose a source-frame keyframe from a temporally supported, quality-gated pose peak."""
    try:
        fps = float(fps)
    except (TypeError, ValueError):
        fps = 0.0
    if not np.isfinite(fps) or fps <= 0:
        fps = 30.0

    try:
        frame_count = int(total_frames)
    except (TypeError, ValueError):
        frame_count = 0
    if frame_count <= 0:
        frame_count = len(signals)
    if frame_count <= 0:
        return {
            "representative_timestamp": 0.0,
            "representative_frame_index": 0,
            "kinematic_peak_timestamp": None,
            "kinematic_peak_frame_index": None,
            "keyframe_selection_reason": "No source frames were available; used frame 0 fallback.",
        }

    max_frame = frame_count - 1
    max_timestamp = max_frame / fps
    try:
        raw_start = float(start_time)
    except (TypeError, ValueError):
        raw_start = 0.0
    try:
        raw_end = float(end_time)
    except (TypeError, ValueError):
        raw_end = max_timestamp
    if not np.isfinite(raw_start):
        raw_start = 0.0
    if not np.isfinite(raw_end):
        raw_end = max_timestamp
    lower = min(max(raw_start, 0.0), max_timestamp)
    upper = min(max(raw_end, 0.0), max_timestamp)
    if upper < lower:
        lower, upper = upper, lower
    midpoint = (lower + upper) / 2.0
    was_clamped = lower != raw_start or upper != raw_end

    pose_frames: List[Dict[str, Any]] = []
    knee_frames: List[Dict[str, Any]] = []
    frame_tolerance = max(0.51 / fps, 0.01)
    for signal in signals:
        if not signal.get("has_pose"):
            continue
        try:
            raw_index = float(signal.get("frame_idx"))
            frame_index = int(raw_index)
            timestamp = float(signal.get("timestamp"))
            quality = float(signal.get("lower_body_confidence", signal.get("confidence", 0.0)))
        except (TypeError, ValueError, OverflowError):
            continue
        if (
            not np.isfinite(raw_index)
            or raw_index != frame_index
            or frame_index < 0
            or frame_index > max_frame
            or not np.isfinite(timestamp)
            or not np.isfinite(quality)
            or abs(timestamp - frame_index / fps) > frame_tolerance
            or timestamp < lower - 1e-6
            or timestamp > upper + 1e-6
            or quality < min_quality
        ):
            continue

        frame = {
            "timestamp": frame_index / fps,
            "frame_index": frame_index,
            "quality": quality,
        }
        pose_frames.append(frame)
        try:
            knee_l = float(signal.get("feature_knee_l"))
            knee_r = float(signal.get("feature_knee_r"))
        except (TypeError, ValueError):
            continue
        if (
            np.isfinite(knee_l)
            and np.isfinite(knee_r)
            and 0.0 < knee_l <= 180.0
            and 0.0 < knee_r <= 180.0
        ):
            knee_frames.append({**frame, "flexion_angle": min(knee_l, knee_r)})

    peak_candidates = []
    for frame in knee_frames:
        neighborhood = [
            candidate["flexion_angle"]
            for candidate in knee_frames
            if abs(candidate["timestamp"] - frame["timestamp"]) <= context_seconds
        ]
        if len(neighborhood) >= 3:
            peak_candidates.append({
                **frame,
                "smoothed_flexion": float(np.median(neighborhood)),
            })

    if peak_candidates:
        selected = min(
            peak_candidates,
            key=lambda frame: (
                frame["smoothed_flexion"],
                -frame["quality"],
                abs(frame["timestamp"] - midpoint),
            ),
        )
        reason = (
            "Most flexed reliable knee frame; local median over "
            f"±{context_seconds:.2f}s with lower-body/trunk quality ≥ {min_quality:.2f}."
        )
        peak_timestamp: Optional[float] = selected["timestamp"]
        peak_frame_index: Optional[int] = selected["frame_index"]
    elif pose_frames:
        selected = min(
            pose_frames,
            key=lambda frame: (-frame["quality"], abs(frame["timestamp"] - midpoint)),
        )
        reason = (
            "No reliable temporally supported knee peak; selected the highest-quality pose "
            "frame nearest the event midpoint."
        )
        peak_timestamp = None
        peak_frame_index = None
    else:
        frame_index = min(max(int(round(midpoint * fps)), 0), max_frame)
        selected = {"timestamp": frame_index / fps, "frame_index": frame_index}
        reason = "No reliable pose frame was available; used the clamped event midpoint."
        peak_timestamp = None
        peak_frame_index = None

    if was_clamped:
        reason += " Event bounds were clamped to available source frames."
    return {
        "representative_timestamp": round(float(selected["timestamp"]), 4),
        "representative_frame_index": int(selected["frame_index"]),
        "kinematic_peak_timestamp": round(float(peak_timestamp), 4) if peak_timestamp is not None else None,
        "kinematic_peak_frame_index": int(peak_frame_index) if peak_frame_index is not None else None,
        "keyframe_selection_reason": reason,
    }


def _relocalize_low_quality_candidates(
    candidates: List[Any],
    signals: List[Dict[str, Any]],
    fps: float,
    total_frames: int,
    duration: float,
    detector: MovementCandidateDetector,
) -> tuple[List[Any], set[int]]:
    """Replace low-quality, off-peak detector windows with explicit uncertain review windows."""
    localized: List[Any] = []
    review_window_ids: set[int] = set()
    reliable_peak = _select_representative_frame(
        signals, 0.0, duration, fps, total_frames,
        min_quality=MIN_RELIABLE_PEAK_QUALITY,
    )
    peak_time = reliable_peak.get("kinematic_peak_timestamp")

    for candidate in candidates:
        window = [
            signal for signal in signals
            if candidate.start_time - 1e-6 <= signal.get("timestamp", -1.0) <= candidate.end_time + 1e-6
        ]
        quality_values = []
        for signal in window:
            try:
                quality = float(signal.get("lower_body_confidence", signal.get("confidence", 0.0)))
            except (TypeError, ValueError):
                quality = 0.0
            quality_values.append(quality if np.isfinite(quality) else 0.0)
        mean_quality = float(np.mean(quality_values)) if quality_values else 0.0

        if (
            mean_quality < MIN_RELIABLE_PEAK_QUALITY
            and peak_time is not None
            and not (candidate.start_time <= peak_time <= candidate.end_time)
        ):
            source_max_time = max(0.0, (max(total_frames, 1) - 1) / fps)
            review_start = max(0.0, peak_time - REVIEW_WINDOW_CONTEXT_SECONDS)
            review_end = min(source_max_time, peak_time + REVIEW_WINDOW_CONTEXT_SECONDS)
            if review_end - review_start >= 0.6:
                metrics = {
                    "reason": "Low-quality detector window did not contain the best reliable full-source knee-flexion peak.",
                    "original_detector_proposal": candidate.proposed_movement,
                    "original_candidate_score": candidate.heuristic_score,
                    "original_candidate_start_time": candidate.start_time,
                    "original_candidate_end_time": candidate.end_time,
                    "original_candidate_lower_body_quality": round(mean_quality, 3),
                    "review_peak_timestamp": peak_time,
                    "review_peak_frame_index": reliable_peak["kinematic_peak_frame_index"],
                    "peak_quality_threshold": MIN_RELIABLE_PEAK_QUALITY,
                    "review_context_seconds": REVIEW_WINDOW_CONTEXT_SECONDS,
                }
                replacement = detector._create_candidate(
                    video_id=candidate.video_id,
                    category="UNCERTAIN",
                    start_time=review_start,
                    end_time=review_end,
                    score=0.4,
                    metrics=metrics,
                    status="REVIEW_REQUIRED",
                )
                localized.append(replacement)
                review_window_ids.add(id(replacement))
                continue
        localized.append(candidate)

    return localized, review_window_ids


class VideoProcessingPipeline:
    """Orchestrates the entire video-to-clips biomechanical analysis pipeline."""

    def __init__(
        self,
        classifier: Optional[MovementClassifier] = None,
        model_path: Optional[str] = None,
    ):
        if classifier is not None:
            self.classifier = classifier
        else:
            self.classifier = MovementClassifier(model_path=model_path)

    def process_video(
        self,
        video_path: str,
        output_dir: str,
        sample_fps: float = 15.0,
        include_deferred_categories: bool = False,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> Dict[str, Any]:
        """
        Executes complete pipeline on a video file.
        Returns a dictionary with status, metadata, events, manifest path, and annotated video.
        """
        def update_progress(pct: float, msg: str):
            if progress_callback:
                progress_callback(pct, msg)

        update_progress(0.05, "Validating video file...")
        meta = validate_video(video_path)
        if not meta["is_valid"]:
            return {
                "success": False,
                "error": meta["error"],
                "events": [],
                "metadata": meta,
            }

        out_path = Path(output_dir)
        clips_dir = out_path / "clips"
        keyframes_dir = out_path / "keyframes"
        clips_dir.mkdir(parents=True, exist_ok=True)
        keyframes_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return {
                "success": False,
                "error": "Failed to open video capture device.",
                "events": [],
                "metadata": meta,
            }

        src_fps = meta["fps"] if meta["fps"] > 0 else 30.0
        width = meta["width"]
        height = meta["height"]
        total_frames = meta["frame_count"]
        duration = meta["duration"]

        effective_sample_fps = min(sample_fps, src_fps)
        sample_step = max(1, int(round(src_fps / effective_sample_fps)))

        # Phase 1: Frame Extraction & Pose Tracking
        update_progress(0.15, "Running pose tracking across video frames...")
        signals = []
        frame_poses: List[FramePoseResult] = []
        frame_idx = 0

        # Also write annotated video with skeleton overlays
        annotated_video_path = out_path / f"{Path(video_path).stem}_annotated.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        temp_annotated = out_path / f"temp_{annotated_video_path.name}"
        ann_writer = cv2.VideoWriter(str(temp_annotated), fourcc, src_fps, (width, height))

        prev_hip_x = None
        prev_hip_y = None

        with PoseEstimator() as estimator:
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                timestamp_sec = frame_idx / src_fps
                timestamp_ms = int(timestamp_sec * 1000)

                # Detect pose for this frame
                pose_res = estimator.process_frame(frame, timestamp_ms, frame_index=frame_idx)
                frame_poses.append(pose_res)

                # Extract kinematic features for detection signal if pose detected
                if pose_res.has_pose and pose_res.landmarks:
                    lms = pose_res.landmarks
                    # Landmarks: 11:LS, 12:RS, 13:LE, 14:RE, 15:LW, 16:RW, 23:LH, 24:RH, 25:LK, 26:RK, 27:LA, 28:RA
                    hip_x = float((lms[23].x + lms[24].x) / 2.0)
                    hip_y = float((lms[23].y + lms[24].y) / 2.0)
                    hip_z = float((lms[23].z + lms[24].z) / 2.0)

                    hip_vx = (hip_x - prev_hip_x) * src_fps if prev_hip_x is not None else 0.0
                    hip_vy = (hip_y - prev_hip_y) * src_fps if prev_hip_y is not None else 0.0
                    prev_hip_x, prev_hip_y = hip_x, hip_y

                    fallback_conf = pose_res.overall_confidence
                    lk_feature = _confidence_filtered_measurement(
                        get_knee_angle_2d((lms[23].x, lms[23].y), (lms[25].x, lms[25].y), (lms[27].x, lms[27].y), (width, height)),
                        getattr(pose_res, "left_knee_conf", fallback_conf),
                    )
                    rk_feature = _confidence_filtered_measurement(
                        get_knee_angle_2d((lms[24].x, lms[24].y), (lms[26].x, lms[26].y), (lms[28].x, lms[28].y), (width, height)),
                        getattr(pose_res, "right_knee_conf", fallback_conf),
                    )
                    le_feature = _confidence_filtered_measurement(
                        get_elbow_angle_2d((lms[11].x, lms[11].y), (lms[13].x, lms[13].y), (lms[15].x, lms[15].y), (width, height)),
                        getattr(pose_res, "left_elbow_conf", fallback_conf),
                    )
                    re_feature = _confidence_filtered_measurement(
                        get_elbow_angle_2d((lms[12].x, lms[12].y), (lms[14].x, lms[14].y), (lms[16].x, lms[16].y), (width, height)),
                        getattr(pose_res, "right_elbow_conf", fallback_conf),
                    )
                    shoulder_x = float((lms[11].x + lms[12].x) / 2.0)
                    shoulder_y = float((lms[11].y + lms[12].y) / 2.0)
                    torso_feature = _confidence_filtered_measurement(
                        get_torso_lean_2d((shoulder_x, shoulder_y), (hip_x, hip_y), (width, height)),
                        getattr(pose_res, "torso_conf", fallback_conf),
                    )
                    # Neutral numeric sentinels are used only by the temporal detector;
                    # feature aggregation keeps the raw missing measurement as None.
                    lk = lk_feature if lk_feature is not None else 180.0
                    rk = rk_feature if rk_feature is not None else 180.0
                    le = le_feature if le_feature is not None else 180.0
                    re = re_feature if re_feature is not None else 180.0
                    torso = torso_feature if torso_feature is not None else 0.0

                    ankle_dist = float(np.linalg.norm([lms[27].x - lms[28].x, lms[27].y - lms[28].y]))

                    signals.append({
                        "frame_idx": frame_idx,
                        "timestamp": timestamp_sec,
                        "has_pose": True,
                        "hip_x": hip_x,
                        "hip_y": hip_y,
                        "hip_z": hip_z,
                        "ankle_l_x": float(lms[27].x),
                        "ankle_l_y": float(lms[27].y),
                        "ankle_r_x": float(lms[28].x),
                        "ankle_r_y": float(lms[28].y),
                        "ankle_dist": ankle_dist,
                        "wrist_l_y": float(lms[15].y),
                        "wrist_r_y": float(lms[16].y),
                        "shoulder_y": float(min(lms[11].y, lms[12].y)),
                        "knee_l": lk,
                        "knee_r": rk,
                        "feature_knee_l": lk_feature,
                        "feature_knee_r": rk_feature,
                        "min_knee": min(lk, rk),
                        "elbow_l": le,
                        "elbow_r": re,
                        "feature_elbow_l": le_feature,
                        "feature_elbow_r": re_feature,
                        "max_elbow": max(le, re),
                        "torso_lean": torso,
                        "feature_torso_lean": torso_feature,
                        "confidence": pose_res.overall_confidence,
                        "risk_confidence": min(
                            float(getattr(pose_res, key, fallback_conf))
                            for key in ("left_knee_conf", "right_knee_conf", "left_elbow_conf", "right_elbow_conf", "torso_conf")
                        ),
                        "lower_body_confidence": min(
                            float(getattr(pose_res, key, fallback_conf))
                            for key in ("left_knee_conf", "right_knee_conf", "torso_conf")
                        ),
                        "hip_vx": hip_vx,
                        "hip_vy": hip_vy,
                    })

                    # Draw skeleton on overlay video
                    overlay_frame = draw_skeleton_overlay(frame, lms, {"left_knee": lk, "right_knee": rk})
                    ann_writer.write(overlay_frame)
                else:
                    prev_hip_x = prev_hip_y = None
                    signals.append({
                        "frame_idx": frame_idx,
                        "timestamp": timestamp_sec,
                        "has_pose": False,
                        "hip_x": None, "hip_y": None, "hip_z": None,
                        "ankle_l_x": None, "ankle_l_y": None, "ankle_r_x": None, "ankle_r_y": None,
                        "ankle_dist": None, "wrist_l_y": None, "wrist_r_y": None, "shoulder_y": None,
                        "knee_l": None, "knee_r": None, "min_knee": None,
                        "feature_knee_l": None, "feature_knee_r": None,
                        "elbow_l": None, "elbow_r": None, "max_elbow": None,
                        "feature_elbow_l": None, "feature_elbow_r": None,
                        "torso_lean": None, "feature_torso_lean": None,
                        "confidence": 0.0,
                        "risk_confidence": 0.0,
                        "lower_body_confidence": 0.0,
                        "hip_vx": 0.0, "hip_vy": 0.0,
                    })
                    ann_writer.write(frame)

                frame_idx += 1
                if total_frames > 0 and frame_idx % 20 == 0:
                    prog = 0.15 + 0.45 * (frame_idx / total_frames)
                    update_progress(prog, f"Tracking posture: {frame_idx}/{total_frames} frames...")

        cap.release()
        ann_writer.release()

        # Finalize annotated video transcode
        try:
            import subprocess
            cmd = [
                "ffmpeg", "-y", "-i", str(temp_annotated),
                "-vcodec", "libx264", "-pix_fmt", "yuv420p",
                "-crf", "23", "-preset", "veryfast",
                "-movflags", "+faststart",
                str(annotated_video_path)
            ]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if annotated_video_path.exists() and annotated_video_path.stat().st_size > 0:
                temp_annotated.unlink()
            else:
                temp_annotated.rename(annotated_video_path)
        except Exception:
            if temp_annotated.exists():
                temp_annotated.rename(annotated_video_path)

        # Phase 2: Candidate Event Detection
        update_progress(0.65, "Detecting candidate movement events...")
        config = DetectorConfig(
            sample_fps=src_fps,
            include_deferred_categories=include_deferred_categories,
        )
        detector = MovementCandidateDetector(config)
        candidates = detector.detect_candidates_from_signals(signals, Path(video_path).name)
        candidates = _filter_candidates_by_pose_coverage(candidates, signals, src_fps)
        candidates, review_window_candidate_ids = _relocalize_low_quality_candidates(
            candidates,
            signals,
            src_fps,
            total_frames,
            duration,
            detector,
        )
        fallback_candidate_ids = set()

        # Fallback if no specific event was triggered in a video with poses
        poses_count = sum(1 for s in signals if s["has_pose"])
        if not candidates and poses_count > 10:
            # Create a single holistic event covering the video duration
            fallback_candidate = detector._create_candidate(
                video_id=Path(video_path).name,
                category="UNCERTAIN",
                start_time=0.0,
                end_time=round(duration, 2),
                score=0.4,
                metrics={"note": "Holistic sequence event fallback"},
            )
            candidates.append(fallback_candidate)
            fallback_candidate_ids.add(id(fallback_candidate))

        # Phase 3: Event Clip Extraction, Classification & Risk Assessment
        update_progress(0.75, "Extracting movement clips and evaluating biomechanics...")
        event_records: List[MovementEventRecord] = []

        for idx, cand in enumerate(candidates, start=1):
            e_id = f"event_{idx:03d}"
            clip_name = f"{e_id}.mp4"
            keyframe_name = f"{e_id}_key.jpg"
            clip_path = clips_dir / clip_name
            keyframe_path = keyframes_dir / keyframe_name

            # Sub-clip extraction
            extract_subclip(video_path, cand.start_time, cand.end_time, str(clip_path))

            # Select and extract an auditable source frame from the full event interval.
            keyframe_info = _select_representative_frame(
                signals,
                cand.start_time,
                cand.end_time,
                src_fps,
                total_frames,
            )
            extract_keyframe(
                video_path,
                keyframe_info["representative_timestamp"],
                str(keyframe_path),
            )

            # Extract event frame features
            start_f = max(0, int(round(cand.start_time * src_fps)))
            end_f = min(len(signals), int(round(cand.end_time * src_fps)))
            event_signals = signals[start_f:end_f] if start_f < end_f else signals
            pose_event_signals = [s for s in event_signals if s.get("has_pose")]
            feature_dict = _aggregate_pose_signals(event_signals)
            if not pose_event_signals:
                feature_status = "FAILED"
            elif len(pose_event_signals) == len(event_signals) and all(
                value is not None for value in feature_dict.values()
            ):
                feature_status = "SUCCESS"
            else:
                feature_status = "PARTIAL"

            avg_conf = float(np.mean([s.get("risk_confidence", s["confidence"]) for s in event_signals])) if event_signals else 0.0
            avg_lower_body_conf = float(
                np.mean([s.get("lower_body_confidence", s.get("confidence", 0.0)) for s in event_signals])
            ) if event_signals else 0.0

            # Classification
            pred_res = self.classifier.predict(feature_dict)

            # Preserve detector uncertainty for human review while recording the
            # classifier's separate suggestion so displayed probabilities are interpretable.
            classification = _resolve_event_classification(
                cand.proposed_movement,
                pred_res,
                self.classifier.is_trained_model,
            )
            if id(cand) in fallback_candidate_ids:
                classification["classification_note"] = (
                    "No movement-specific detector criteria produced an event, so the "
                    "pipeline created a whole-video fallback and kept its label UNCERTAIN. "
                    "The classifier suggestion and probabilities are shown for review; "
                    "they do not verify the movement or override detector uncertainty."
                )
            elif id(cand) in review_window_candidate_ids:
                classification["classification_note"] = (
                    "The original detector window had low lower-body quality and did not "
                    "contain the best reliable full-source knee-flexion peak. A focused "
                    "UNCERTAIN review window was placed around that peak; this does not "
                    "confirm a movement class. Review the full event clip and source video."
                )
            final_category = classification["movement_category"]
            label_source = classification["label_source"]
            try:
                detector_metrics = json.loads(cand.key_metrics) if cand.key_metrics else {}
                if not isinstance(detector_metrics, dict):
                    detector_metrics = {"details": detector_metrics}
            except (TypeError, ValueError):
                detector_metrics = {"details": str(cand.key_metrics)}

            # Biomechanical Risk Assessment (Rubric v0.3)
            risk_result = evaluate_movement_risk(
                final_category,
                feature_dict,
                landmark_quality=avg_conf,
                lower_body_quality=avg_lower_body_conf,
            )

            record = MovementEventRecord(
                event_id=e_id,
                video_id=Path(video_path).name,
                source_video=str(video_path),
                start_time=round(cand.start_time, 2),
                end_time=round(cand.end_time, 2),
                start_frame=start_f,
                end_frame=end_f,
                movement_category=final_category,
                localization_method=(
                    "whole_video_fallback" if id(cand) in fallback_candidate_ids
                    else "kinematic_peak_review_window" if id(cand) in review_window_candidate_ids
                    else "candidate_detector"
                ),
                detection_status="UNCERTAIN" if final_category == "UNCERTAIN" else "PROPOSED",
                representative_frame=str(keyframe_path),
                representative_timestamp=keyframe_info["representative_timestamp"],
                representative_frame_index=keyframe_info["representative_frame_index"],
                kinematic_peak_timestamp=keyframe_info["kinematic_peak_timestamp"],
                kinematic_peak_frame_index=keyframe_info["kinematic_peak_frame_index"],
                keyframe_selection_reason=keyframe_info["keyframe_selection_reason"],
                clip_path=str(clip_path),
                pose_quality=round(avg_conf, 3),
                lower_body_quality=round(avg_lower_body_conf, 3),
                detector_proposal=(cand.proposed_movement or "") or None,
                detector_metrics=detector_metrics,
                classifier_prediction=pred_res.predicted_category,
                classification_note=classification["classification_note"],
                feature_extraction_status=feature_status,
                annotation_status="AUTO_PROPOSED",
                label_source=label_source,
                rubric_version="v0.3",
                risk_level=risk_result.risk_level,
                risk_observations=risk_result.observations,
                risk_missing_evidence=risk_result.missing_evidence,
                risk_score=risk_result.score,
                risk_category=risk_result.screening_category,
                risk_contributions=risk_result.contributing_indicators,
                features=feature_dict,
                probabilities=pred_res.probabilities,
                prediction_source=pred_res.prediction_source,
                player_id=getattr(cand, "player_id", None) or None,
            )
            event_records.append(record)

        # Save Structured Manifest
        manifest_path = out_path / "event_manifest.json"
        save_event_manifest(event_records, str(manifest_path))

        update_progress(1.0, "Analysis complete!")
        return {
            "success": True,
            "error": None,
            "metadata": meta,
            "events": [e.to_dict() for e in event_records],
            "manifest_path": str(manifest_path),
            "annotated_video_path": str(annotated_video_path),
            "clips_dir": str(clips_dir),
            "keyframes_dir": str(keyframes_dir),
            "total_events": len(event_records),
        }
