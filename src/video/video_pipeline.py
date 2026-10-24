"""
End-to-end video-to-movement-clips processing pipeline.
Performs video validation, frame extraction, pose estimation, feature calculation,
event candidate detection, 5-class movement classification, clip/keyframe extraction,
rubric v0.3 risk evaluation, and manifest generation.
"""

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
                        "hip_vx": hip_vx,
                        "hip_vy": hip_vy,
                    })

                    # Draw skeleton on overlay video
                    overlay_frame = draw_skeleton_overlay(frame, lms, {"left_knee": lk, "right_knee": rk})
                    ann_writer.write(overlay_frame)
                else:
                    signals.append({
                        "frame_idx": frame_idx,
                        "timestamp": timestamp_sec,
                        "has_pose": False,
                        "hip_x": 0.5, "hip_y": 0.5, "hip_z": 0.0,
                        "ankle_l_x": 0.5, "ankle_l_y": 0.9, "ankle_r_x": 0.5, "ankle_r_y": 0.9,
                        "ankle_dist": 0.1, "wrist_l_y": 0.5, "wrist_r_y": 0.5, "shoulder_y": 0.3,
                        "knee_l": 180.0, "knee_r": 180.0, "min_knee": 180.0,
                        "elbow_l": 180.0, "elbow_r": 180.0, "max_elbow": 180.0,
                        "torso_lean": 0.0, "confidence": 0.0,
                        "risk_confidence": 0.0,
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

            # Keyframe extraction (at center or peak of event)
            key_time = (cand.start_time + cand.end_time) / 2.0
            extract_keyframe(video_path, key_time, str(keyframe_path))

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

            # Classification
            pred_res = self.classifier.predict(feature_dict)

            # Preserve detector uncertainty for human review.
            # A classifier prediction alone must not resolve an ambiguous detection.
            if cand.proposed_movement == "UNCERTAIN":
                final_category = "UNCERTAIN"
                label_source = "CANDIDATE_DETECTOR"
            elif self.classifier.is_trained_model:
                final_category = pred_res.predicted_category
                label_source = "CLASSIFIER_MODEL"
            elif cand.proposed_movement:
                final_category = cand.proposed_movement
                label_source = "CANDIDATE_DETECTOR"
            else:
                final_category = pred_res.predicted_category
                label_source = "CLASSIFIER_HEURISTIC"

            # Biomechanical Risk Assessment (Rubric v0.3)
            risk_result = evaluate_movement_risk(final_category, feature_dict, landmark_quality=avg_conf)

            record = MovementEventRecord(
                event_id=e_id,
                video_id=Path(video_path).name,
                source_video=str(video_path),
                start_time=round(cand.start_time, 2),
                end_time=round(cand.end_time, 2),
                start_frame=start_f,
                end_frame=end_f,
                movement_category=final_category,
                localization_method=("whole_video_fallback" if id(cand) in fallback_candidate_ids else "candidate_detector"),
                detection_status="UNCERTAIN" if final_category == "UNCERTAIN" else "PROPOSED",
                representative_frame=str(keyframe_path),
                clip_path=str(clip_path),
                pose_quality=round(avg_conf, 3),
                feature_extraction_status=feature_status,
                annotation_status="AUTO_PROPOSED",
                label_source=label_source,
                rubric_version="v0.3",
                risk_level=risk_result.risk_level,
                risk_observations=risk_result.observations,
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
