"""
Automatic Badminton Movement Candidate Detector

Scans video footage using MediaPipe pose tracking and temporal biomechanical heuristics
to propose start/end timestamps and candidate categories for exactly three target classes:
- LUNGE
- SMASH_LANDING
- CHANGE_OF_DIRECTION

Ambiguous observations or conflicting signals are classified as UNCERTAIN for human review.
This module generates candidate suggestions for human review; it does not assign
injury-risk labels.
"""

import csv
import json
import os
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path
import cv2
import mediapipe as mp
import numpy as np

from pose_analysis import (
    MODEL_PATH,
    ensure_model_file,
    get_knee_angle,
    get_elbow_angle,
    get_torso_lean,
    get_landmark_confidence,
)

TARGET_MOVEMENT_CLASSES = ["LUNGE", "SMASH_LANDING", "CHANGE_OF_DIRECTION"]


@dataclass
class MovementCandidate:
    candidate_id: str
    video_id: str
    movement_type: str  # Kept for backward compatibility (maps to proposed_movement)
    start_time: float
    end_time: float
    duration: float
    confidence_score: float  # Kept for backward compatibility (maps to heuristic_score)
    key_metrics: str  # JSON-formatted summary of trigger metrics
    status: str = "PROPOSED"  # Kept for backward compatibility
    # Human Review Workflow Fields
    proposed_movement: str = ""
    heuristic_score: float = 0.0  # Explicit uncalibrated heuristic score (0.0 - 1.0, NOT a probability)
    human_decision: str = "PENDING"  # PENDING, ACCEPT, REJECT, UNCERTAIN, NO_EVENT
    reviewed_movement: str = ""  # Confirmed movement class upon human review
    reviewer_notes: str = ""
    player_id: str = ""  # Optional player identity

    def __post_init__(self):
        if not self.proposed_movement:
            self.proposed_movement = self.movement_type
        if not self.movement_type:
            self.movement_type = self.proposed_movement

        if self.heuristic_score <= 0.0 and self.confidence_score > 0.0:
            self.heuristic_score = self.confidence_score
        elif self.confidence_score <= 0.0 and self.heuristic_score > 0.0:
            self.confidence_score = self.heuristic_score

        if not self.status:
            self.status = "PROPOSED"


@dataclass
class DetectorConfig:
    sample_fps: float = 15.0
    min_landmark_confidence: float = 0.4

    # Baseline tracking
    baseline_window_seconds: float = 3.0  # Rolling window for local ground baseline

    # Lunge parameters
    lunge_knee_threshold: float = 120.0  # Lead knee flex angle in degrees
    lunge_knee_asymmetry_min: float = 20.0  # Min difference between lead and trail knee to reject bilateral squats
    lunge_stance_ratio_min: float = 0.70  # Min ankle distance relative to torso length
    lunge_hip_drop_min: float = 0.04      # Min vertical drop of hip from local baseline
    lunge_min_duration: float = 0.6
    lunge_max_duration: float = 2.5

    # Smash landing parameters
    jump_elevation_min: float = 0.045   # Min vertical hip rise off local baseline
    jump_vertical_vel_min: float = 0.10 # Min upward/downward velocity pulse
    jump_foot_clearance_min: float = 0.028 # Min foot clearance off floor plane
    overhead_arm_margin: float = 0.05   # Wrist must be at least this far above shoulder in y
    landing_knee_flex_threshold: float = 160.0  # Supports cushioning in both flexed and stiff-legged landings (up to 160 deg)
    landing_upright_knee_min: float = 165.0     # Standing leg extension threshold to detect recovery to upright stance
    landing_upright_consecutive_frames: int = 2 # Consecutive frames of upright stance after touchdown required to reset search
    smash_min_duration: float = 0.5
    smash_max_duration: float = 2.2

    # Change of direction parameters
    cod_velocity_threshold: float = 0.16  # Significant lateral approach velocity
    cod_reversal_window: float = 0.6      # Time window for direction reversal
    cod_braking_knee_max: float = 150.0   # Must have athletic braking knee bend
    cod_min_duration: float = 0.4
    cod_max_duration: float = 1.8

    # Non-maximum suppression and merging
    overlap_iou_threshold: float = 0.30
    merge_gap_seconds: float = 0.35
    cross_type_conflict_iou: float = 0.35

    # Extended categories flag (supports SCISSOR_JUMP & FOOTWORK_SPLIT_STEP)
    include_deferred_categories: bool = False


CANDIDATE_CSV_FIELDS = [
    "candidate_id",
    "video_id",
    "start_time",
    "end_time",
    "duration",
    "proposed_movement",
    "heuristic_score",
    "key_metrics",
    "human_decision",
    "reviewed_movement",
    "reviewer_notes",
    "player_id",
    # Legacy compatibility fields
    "movement_type",
    "confidence_score",
    "status",
]


class MovementCandidateDetector:
    def __init__(self, config: DetectorConfig | None = None):
        self.config = config or DetectorConfig()

    def extract_temporal_signals(self, video_path: str) -> list[dict]:
        """
        Extracts temporal landmark and angle time-series signals from video.
        """
        ensure_model_file()

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open video: {video_path}")

        source_fps = cap.get(cv2.CAP_PROP_FPS)
        if source_fps <= 0:
            source_fps = 30.0

        BaseOptions = mp.tasks.BaseOptions
        PoseLandmarker = mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode
        PoseLandmark = mp.tasks.vision.PoseLandmark

        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
            running_mode=VisionRunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        sample_interval = 1.0 / max(self.config.sample_fps, 1.0)
        next_sample_time = 0.0
        frame_idx = 0
        signals: list[dict] = []

        with PoseLandmarker.create_from_options(options) as landmarker:
            while True:
                success, frame = cap.read()
                if not success:
                    break

                timestamp_sec = frame_idx / source_fps

                if timestamp_sec >= next_sample_time - 1e-6:
                    next_sample_time += sample_interval

                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                    timestamp_ms = int(timestamp_sec * 1000)
                    result = landmarker.detect_for_video(mp_image, timestamp_ms)

                    sig = {
                        "frame_idx": frame_idx,
                        "timestamp": timestamp_sec,
                        "has_pose": False,
                        "hip_x": None,
                        "hip_y": None,
                        "hip_z": None,
                        "ankle_l_x": None,
                        "ankle_l_y": None,
                        "ankle_r_x": None,
                        "ankle_r_y": None,
                        "ankle_dist": None,
                        "wrist_l_y": None,
                        "wrist_r_y": None,
                        "shoulder_y": None,
                        "knee_l": None,
                        "knee_r": None,
                        "min_knee": None,
                        "elbow_l": None,
                        "elbow_r": None,
                        "max_elbow": None,
                        "torso_lean": None,
                        "confidence": 0.0,
                    }

                    if result.pose_landmarks and len(result.pose_landmarks) > 0:
                        lms = result.pose_landmarks[0]
                        sig["has_pose"] = True

                        # Extract key points
                        hl = lms[PoseLandmark.LEFT_HIP]
                        hr = lms[PoseLandmark.RIGHT_HIP]
                        al = lms[PoseLandmark.LEFT_ANKLE]
                        ar = lms[PoseLandmark.RIGHT_ANKLE]
                        sl = lms[PoseLandmark.LEFT_SHOULDER]
                        sr = lms[PoseLandmark.RIGHT_SHOULDER]
                        wl = lms[PoseLandmark.LEFT_WRIST]
                        wr = lms[PoseLandmark.RIGHT_WRIST]

                        sig["hip_x"] = float((hl.x + hr.x) / 2.0)
                        sig["hip_y"] = float((hl.y + hr.y) / 2.0)
                        sig["hip_z"] = float((hl.z + hr.z) / 2.0)

                        sig["ankle_l_x"] = float(al.x)
                        sig["ankle_l_y"] = float(al.y)
                        sig["ankle_r_x"] = float(ar.x)
                        sig["ankle_r_y"] = float(ar.y)
                        sig["ankle_dist"] = float(abs(al.x - ar.x))

                        sig["wrist_l_y"] = float(wl.y)
                        sig["wrist_r_y"] = float(wr.y)
                        sig["shoulder_y"] = float((sl.y + sr.y) / 2.0)

                        # Angles
                        kl, kr, _ = get_knee_angle(lms, PoseLandmark)
                        el, er, _ = get_elbow_angle(lms, PoseLandmark)
                        sig["knee_l"] = kl
                        sig["knee_r"] = kr
                        valid_knees = [k for k in (kl, kr) if k is not None]
                        sig["min_knee"] = float(min(valid_knees)) if valid_knees else None

                        sig["elbow_l"] = el
                        sig["elbow_r"] = er
                        valid_elbows = [e for e in (el, er) if e is not None]
                        sig["max_elbow"] = float(max(valid_elbows)) if valid_elbows else None

                        sig["torso_lean"] = get_torso_lean(lms, PoseLandmark)
                        confs = [get_landmark_confidence(lm) for lm in (hl, hr, al, ar, sl, sr)]
                        sig["confidence"] = float(np.mean(confs))

                    signals.append(sig)

                frame_idx += 1

        cap.release()
        return self._smooth_and_interpolate_signals(signals)

    def _smooth_and_interpolate_signals(self, signals: list[dict]) -> list[dict]:
        """
        Interpolates short tracking losses, computes normalized stance ratio,
        determines local dynamic ground baseline, and computes smoothed velocities.
        """
        n = len(signals)
        if n == 0:
            return signals

        # Linear interpolation for isolated missing frames (up to 4 frames gap)
        numeric_keys = [
            "hip_x", "hip_y", "hip_z", "ankle_l_x", "ankle_l_y",
            "ankle_r_x", "ankle_r_y", "ankle_dist", "wrist_l_y",
            "wrist_r_y", "shoulder_y", "knee_l", "knee_r",
            "min_knee", "max_elbow", "torso_lean"
        ]
        for key in numeric_keys:
            valid_indices = [i for i, s in enumerate(signals) if s.get(key) is not None]
            if len(valid_indices) >= 2:
                for idx in range(len(valid_indices) - 1):
                    i1 = valid_indices[idx]
                    i2 = valid_indices[idx + 1]
                    gap = i2 - i1
                    if 1 < gap <= 4:  # small gap
                        v1 = signals[i1][key]
                        v2 = signals[i2][key]
                        for step in range(1, gap):
                            frac = step / gap
                            signals[i1 + step][key] = float(v1 + frac * (v2 - v1))

        # Torso length and normalized stance ratio (body-scale invariant)
        for i in range(n):
            s = signals[i]
            torso_len = 0.35  # default approximate scale
            if s["shoulder_y"] is not None and s["hip_y"] is not None:
                torso_len = max(0.12, abs(s["hip_y"] - s["shoulder_y"]))
            s["torso_len"] = torso_len

            if s["ankle_dist"] is not None:
                s["stance_ratio"] = float(s["ankle_dist"] / torso_len)
            else:
                s["stance_ratio"] = 0.0

            # Knee asymmetry (for distinguishing unilateral lunge vs bilateral squat)
            knees = [k for k in (s.get("knee_l"), s.get("knee_r")) if k is not None]
            min_k = s.get("min_knee")
            if min_k is not None and min_k not in knees:
                knees.append(min_k)
            if len(knees) >= 2:
                s["knee_asym"] = float(abs(max(knees) - min(knees)))
            else:
                s["knee_asym"] = 0.0

        # Standing ground baseline for hip_y
        # In image coords (0=top, 1=bottom), standing hips have a smaller y value than crouched/lunging hips.
        # Jumps elevate ABOVE standing height (hip_y < standing_hip_y).
        # Lunges/squats drop BELOW standing height (hip_y > standing_hip_y).
        half_win = int(self.config.baseline_window_seconds * self.config.sample_fps / 2)
        valid_all_hips = [s["hip_y"] for s in signals if s["hip_y"] is not None]
        global_fallback_standing = float(np.percentile(valid_all_hips, 30)) if valid_all_hips else 0.5

        for i in range(n):
            w_start = max(0, i - half_win)
            w_end = min(n, i + half_win + 1)
            # Find frames in local window where knees are extended (upright standing)
            standing_hips = [
                signals[k]["hip_y"]
                for k in range(w_start, w_end)
                if signals[k]["hip_y"] is not None
                and (signals[k].get("min_knee") is None or signals[k]["min_knee"] >= 145.0)
            ]
            if len(standing_hips) >= 3:
                signals[i]["standing_hip_y"] = float(np.median(standing_hips))
            else:
                local_hips = [signals[k]["hip_y"] for k in range(w_start, w_end) if signals[k]["hip_y"] is not None]
                signals[i]["standing_hip_y"] = float(np.percentile(local_hips, 30)) if local_hips else global_fallback_standing
            # Also keep baseline_hip_y aliased to standing_hip_y
            signals[i]["baseline_hip_y"] = signals[i]["standing_hip_y"]

        # Ground floor plane and foot clearance tracking (distinguishes airborne jumps from grounded stance)
        for i in range(n):
            s = signals[i]
            ankles_y = [ay for ay in (s.get("ankle_l_y"), s.get("ankle_r_y")) if ay is not None]
            s["foot_lowest_y"] = max(ankles_y) if ankles_y else None

        for i in range(n):
            w_start = max(0, i - half_win)
            w_end = min(n, i + half_win + 1)
            local_feet = [signals[k]["foot_lowest_y"] for k in range(w_start, w_end) if signals[k]["foot_lowest_y"] is not None]
            if local_feet:
                local_ground_y = float(np.percentile(local_feet, 85))  # floor plane
                signals[i]["ground_y"] = local_ground_y
                if signals[i]["foot_lowest_y"] is not None:
                    # Positive value indicates feet lifted into air above court floor plane
                    signals[i]["foot_clearance"] = float(local_ground_y - signals[i]["foot_lowest_y"])
                else:
                    signals[i]["foot_clearance"] = 0.0
            else:
                signals[i]["ground_y"] = 0.90
                signals[i]["foot_clearance"] = 0.0

        # Add velocity fields (smoothed central differences)
        for i in range(n):
            if "hip_vx" not in signals[i]:
                signals[i]["hip_vx"] = 0.0
            if "hip_vy" not in signals[i]:
                signals[i]["hip_vy"] = 0.0

            if i > 0 and i < n - 1:
                dt = max(signals[i + 1]["timestamp"] - signals[i - 1]["timestamp"], 1e-4)
                if signals[i + 1]["hip_x"] is not None and signals[i - 1]["hip_x"] is not None:
                    diff_x = signals[i + 1]["hip_x"] - signals[i - 1]["hip_x"]
                    if abs(diff_x) > 1e-6 or signals[i]["hip_vx"] == 0.0:
                        signals[i]["hip_vx"] = float(diff_x / dt)
                if signals[i + 1]["hip_y"] is not None and signals[i - 1]["hip_y"] is not None:
                    diff_y = signals[i + 1]["hip_y"] - signals[i - 1]["hip_y"]
                    if abs(diff_y) > 1e-6 or signals[i]["hip_vy"] == 0.0:
                        signals[i]["hip_vy"] = float(diff_y / dt)

        return signals

    def detect_candidates_from_signals(
        self, signals: list[dict], video_name: str
    ) -> list[MovementCandidate]:
        """
        Applies candidate detection heuristics across the extracted temporal signals.
        """
        if not signals:
            return []

        # Ensure features, baselines, and velocities are computed
        signals = self._smooth_and_interpolate_signals(signals)

        raw_candidates: list[MovementCandidate] = []
        raw_candidates.extend(self._detect_lunges(signals, video_name))
        raw_candidates.extend(self._detect_smash_landings(signals, video_name))
        raw_candidates.extend(self._detect_changes_of_direction(signals, video_name))

        if getattr(self.config, "include_deferred_categories", False):
            raw_candidates.extend(self._detect_scissor_jumps(signals, video_name))
            raw_candidates.extend(self._detect_split_steps(signals, video_name))

        # Merge overlapping detections and suppress duplicates
        merged = self._merge_candidates(raw_candidates)

        # Assign unique sequence IDs
        for idx, cand in enumerate(merged, start=1):
            cand.candidate_id = f"cand_{idx:03d}"

        return merged

    def _create_candidate(
        self,
        video_id: str,
        category: str = "UNCERTAIN",
        start_time: float = 0.0,
        end_time: float = 0.0,
        score: float = 0.0,
        metrics: dict | str | None = None,
        candidate_id: str | None = None,
        status: str | None = None,
        human_decision: str = "PENDING",
        player_id: str = "",
    ) -> MovementCandidate:
        """Helper factory to construct a validated MovementCandidate instance."""
        t_start = round(float(start_time), 2)
        t_end = round(float(end_time), 2)
        dur = round(max(0.0, t_end - t_start), 2)
        if isinstance(metrics, dict):
            key_metrics_str = json.dumps(metrics)
        elif isinstance(metrics, str):
            key_metrics_str = metrics
        else:
            key_metrics_str = "{}"
        cat_str = (category or "UNCERTAIN").strip().upper()

        cand_id = candidate_id or "cand_001"
        cand_status = status or ("REVIEW_REQUIRED" if cat_str == "UNCERTAIN" else "PROPOSED")

        return MovementCandidate(
            candidate_id=cand_id,
            video_id=video_id,
            movement_type=cat_str,
            start_time=t_start,
            end_time=t_end,
            duration=dur,
            confidence_score=float(score),
            key_metrics=key_metrics_str,
            status=cand_status,
            proposed_movement=cat_str,
            heuristic_score=float(score),
            human_decision=human_decision,
            reviewed_movement="",
            reviewer_notes="",
            player_id=player_id,
        )

    def _detect_lunges(self, signals: list[dict], video_name: str) -> list[MovementCandidate]:
        candidates = []
        n = len(signals)
        in_lunge = False
        start_idx = 0
        min_knee_val = 180.0
        max_stance = 0.0
        max_stance_ratio = 0.0
        max_asym = 0.0

        for i, s in enumerate(signals):
            knee = s["min_knee"]
            if knee is not None and knee <= self.config.lunge_knee_threshold:
                if not in_lunge:
                    in_lunge = True
                    start_idx = max(0, i - 3)
                    min_knee_val = knee
                    max_stance = s["ankle_dist"] or 0.0
                    max_stance_ratio = s.get("stance_ratio", 0.0)
                    max_asym = s.get("knee_asym", 0.0)
                else:
                    min_knee_val = min(min_knee_val, knee)
                    if s["ankle_dist"]:
                        max_stance = max(max_stance, s["ankle_dist"])
                    max_stance_ratio = max(max_stance_ratio, s.get("stance_ratio", 0.0))
                    max_asym = max(max_asym, s.get("knee_asym", 0.0))
            else:
                if in_lunge:
                    end_idx = min(n - 1, i + 3)
                    t_start = signals[start_idx]["timestamp"]
                    t_end = signals[end_idx]["timestamp"]
                    dur = t_end - t_start

                    if self.config.lunge_min_duration <= dur <= self.config.lunge_max_duration:
                        # Check local hip drop
                        base_hips = [signals[k].get("baseline_hip_y", 0.5) for k in range(start_idx, end_idx)]
                        loc_base = np.median(base_hips) if base_hips else 0.5
                        lowest_hip = max(
                            [signals[k]["hip_y"] for k in range(start_idx, end_idx) if signals[k]["hip_y"] is not None]
                            or [loc_base]
                        )
                        hip_drop = lowest_hip - loc_base

                        # Condition 1: Hip drop must be present
                        if hip_drop >= self.config.lunge_hip_drop_min:
                            # Condition 2: Check for bilateral squat vs unilateral lunge
                            is_bilateral = max_asym < self.config.lunge_knee_asymmetry_min
                            is_narrow_stance = (
                                max_stance_ratio < self.config.lunge_stance_ratio_min and max_stance < 0.16
                            )

                            if is_bilateral and is_narrow_stance:
                                # Bilateral squat or standing knee bend - NOT a badminton lunge
                                pass
                            elif is_bilateral or is_narrow_stance:
                                # Ambiguous: deep knee bend but partial criteria
                                knee_score = np.clip((135.0 - min_knee_val) / 45.0, 0.4, 0.80)
                                score = float(round(knee_score * 0.6, 2))
                                candidates.append(
                                    MovementCandidate(
                                        candidate_id="",
                                        video_id=video_name,
                                        movement_type="UNCERTAIN",
                                        start_time=round(t_start, 2),
                                        end_time=round(t_end, 2),
                                        duration=round(dur, 2),
                                        confidence_score=score,
                                        key_metrics=json.dumps({
                                            "min_knee_angle": round(min_knee_val, 1),
                                            "knee_asymmetry": round(float(max_asym), 1),
                                            "stance_ratio": round(float(max_stance_ratio), 2),
                                            "hip_drop": round(float(hip_drop), 3),
                                            "uncertain_reason": "Bilateral squat or narrow stance knee bend",
                                        }),
                                        status="REVIEW_REQUIRED",
                                        proposed_movement="UNCERTAIN",
                                        heuristic_score=score,
                                        human_decision="PENDING",
                                    )
                                )
                            else:
                                # High quality unilateral lunge candidate
                                knee_score = np.clip((135.0 - min_knee_val) / 45.0, 0.4, 0.95)
                                asym_score = np.clip(max_asym / 45.0, 0.4, 0.95)
                                stance_score = np.clip(max_stance_ratio / 1.1, 0.4, 0.95)
                                score = float(round(knee_score * 0.4 + asym_score * 0.3 + stance_score * 0.3, 2))

                                candidates.append(
                                    MovementCandidate(
                                        candidate_id="",
                                        video_id=video_name,
                                        movement_type="LUNGE",
                                        start_time=round(t_start, 2),
                                        end_time=round(t_end, 2),
                                        duration=round(dur, 2),
                                        confidence_score=score,
                                        key_metrics=json.dumps({
                                            "min_knee_angle": round(min_knee_val, 1),
                                            "knee_asymmetry": round(float(max_asym), 1),
                                            "stance_ratio": round(float(max_stance_ratio), 2),
                                            "hip_drop": round(float(hip_drop), 3),
                                            "max_stance_width": round(float(max_stance), 3),
                                        }),
                                        status="PROPOSED",
                                        proposed_movement="LUNGE",
                                        heuristic_score=score,
                                        human_decision="PENDING",
                                    )
                                )

                    in_lunge = False
                    min_knee_val = 180.0
                    max_stance = 0.0
                    max_stance_ratio = 0.0
                    max_asym = 0.0

        return candidates

    def _detect_smash_landings(self, signals: list[dict], video_name: str) -> list[MovementCandidate]:
        candidates = []
        n = len(signals)
        if n < 10:
            return []

        i = 0
        while i < n - 6:
            s = signals[i]
            local_baseline = s.get("baseline_hip_y", 0.5)
            # Jump apex: hip_y significantly less than local baseline (higher off ground)
            if s["hip_y"] is not None and (local_baseline - s["hip_y"]) >= self.config.jump_elevation_min:
                apex_idx = i

                # Dynamic vertical velocity check:
                pre_vy = [signals[k]["hip_vy"] for k in range(max(0, apex_idx - 4), apex_idx)]
                post_vy = [signals[k]["hip_vy"] for k in range(apex_idx + 1, min(n, apex_idx + 5))]
                has_vertical_pulse = (
                    any(v < -self.config.jump_vertical_vel_min * 0.4 for v in pre_vy)
                    and any(v > self.config.jump_vertical_vel_min * 0.4 for v in post_vy)
                )

                # Check for overhead arm extension near apex:
                # Wrist must be elevated above shoulder level in image space
                arm_overhead = False
                for k in range(max(0, apex_idx - 4), min(n, apex_idx + 4)):
                    sh_y = signals[k]["shoulder_y"]
                    if sh_y is not None:
                        wl_y = signals[k]["wrist_l_y"]
                        wr_y = signals[k]["wrist_r_y"]
                        if (wl_y is not None and wl_y < sh_y - self.config.overhead_arm_margin) or (
                            wr_y is not None and wr_y < sh_y - self.config.overhead_arm_margin
                        ):
                            arm_overhead = True
                            break

                # Search forward for touchdown and landing impact cushioning
                landing_found = False
                landing_idx = apex_idx
                min_landing_knee = 180.0
                touchdown_reached = False
                consecutive_upright = 0
                for j in range(
                    apex_idx + 1, min(n, apex_idx + int(self.config.smash_max_duration * self.config.sample_fps))
                ):
                    curr_hip = signals[j]["hip_y"]
                    curr_knee = signals[j]["min_knee"]
                    j_base = signals[j].get("baseline_hip_y", local_baseline)
                    if curr_hip is not None and curr_hip >= j_base - 0.015:
                        touchdown_reached = True
                        # Touchdown reached
                        if curr_knee is not None and curr_knee <= self.config.landing_knee_flex_threshold:
                            landing_found = True
                            landing_idx = j
                            min_landing_knee = min(min_landing_knee, curr_knee)
                        elif landing_found:
                            landing_idx = j
                            break

                    # Upright stance reset: terminate search if player has returned to upright standing
                    if touchdown_reached and not landing_found:
                        is_upright = (
                            curr_hip is not None
                            and curr_hip <= j_base + 0.01
                            and curr_knee is not None
                            and curr_knee >= self.config.landing_upright_knee_min
                        )
                        if is_upright:
                            consecutive_upright += 1
                            if consecutive_upright >= self.config.landing_upright_consecutive_frames:
                                break
                        else:
                            consecutive_upright = 0

                if landing_found and has_vertical_pulse:
                    start_idx = max(0, apex_idx - 5)
                    end_idx = min(n - 1, landing_idx + 4)
                    t_start = signals[start_idx]["timestamp"]
                    t_end = signals[end_idx]["timestamp"]
                    dur = t_end - t_start

                    if self.config.smash_min_duration <= dur <= self.config.smash_max_duration:
                        elevation = local_baseline - signals[apex_idx]["hip_y"]
                        jump_score = np.clip(elevation / 0.10, 0.4, 0.95)
                        knee_score = np.clip((150.0 - min_landing_knee) / 40.0, 0.4, 0.95)

                        if arm_overhead:
                            score = float(round(jump_score * 0.4 + 0.35 + knee_score * 0.25, 2))
                            candidates.append(
                                MovementCandidate(
                                    candidate_id="",
                                    video_id=video_name,
                                    movement_type="SMASH_LANDING",
                                    start_time=round(t_start, 2),
                                    end_time=round(t_end, 2),
                                    duration=round(dur, 2),
                                    confidence_score=score,
                                    key_metrics=json.dumps({
                                        "jump_elevation": round(float(elevation), 3),
                                        "landing_knee_flexion": round(float(min_landing_knee), 1),
                                        "overhead_arm_detected": True,
                                    }),
                                    status="PROPOSED",
                                    proposed_movement="SMASH_LANDING",
                                    heuristic_score=score,
                                    human_decision="PENDING",
                                )
                            )
                        else:
                            # Genuine jump landing occurred, but NO overhead hitting arm detected!
                            score = float(round(jump_score * 0.5 + knee_score * 0.5, 2))
                            candidates.append(
                                MovementCandidate(
                                    candidate_id="",
                                    video_id=video_name,
                                    movement_type="UNCERTAIN",
                                    start_time=round(t_start, 2),
                                    end_time=round(t_end, 2),
                                    duration=round(dur, 2),
                                    confidence_score=score,
                                    key_metrics=json.dumps({
                                        "jump_elevation": round(float(elevation), 3),
                                        "landing_knee_flexion": round(float(min_landing_knee), 1),
                                        "overhead_arm_detected": False,
                                        "uncertain_reason": "Jump landing detected without overhead arm extension",
                                    }),
                                    status="REVIEW_REQUIRED",
                                    proposed_movement="UNCERTAIN",
                                    heuristic_score=score,
                                    human_decision="PENDING",
                                )
                            )
                        i = end_idx + 2
                        continue

            i += 1

        return candidates

    def _detect_changes_of_direction(self, signals: list[dict], video_name: str) -> list[MovementCandidate]:
        candidates = []
        n = len(signals)
        window = int(self.config.cod_reversal_window * self.config.sample_fps)

        for i in range(window, n - window):
            v_before = signals[i - window]["hip_vx"]
            v_after = signals[i + window]["hip_vx"]

            # Must have high-speed approach and exit velocity in opposite directions
            if v_before * v_after < -0.01:
                delta_v = abs(v_before - v_after)
                approach_speed = abs(v_before)
                exit_speed = abs(v_after)

                if (
                    approach_speed >= self.config.cod_velocity_threshold
                    and exit_speed >= self.config.cod_velocity_threshold * 0.65
                ):
                    start_idx = max(0, i - window)
                    end_idx = min(n - 1, i + window)
                    t_start = signals[start_idx]["timestamp"]
                    t_end = signals[end_idx]["timestamp"]
                    dur = t_end - t_start

                    if self.config.cod_min_duration <= dur <= self.config.cod_max_duration:
                        # Braking plant knee check
                        knees = [
                            signals[k]["min_knee"]
                            for k in range(start_idx, end_idx)
                            if signals[k]["min_knee"] is not None
                        ]
                        min_k = min(knees) if knees else 150.0

                        if min_k <= self.config.cod_braking_knee_max:
                            v_score = np.clip(delta_v / 0.4, 0.4, 0.90)
                            k_score = np.clip((160.0 - min_k) / 40.0, 0.4, 0.90)
                            score = float(round(v_score * 0.6 + k_score * 0.4, 2))

                            candidates.append(
                                MovementCandidate(
                                    candidate_id="",
                                    video_id=video_name,
                                    movement_type="CHANGE_OF_DIRECTION",
                                    start_time=round(t_start, 2),
                                    end_time=round(t_end, 2),
                                    duration=round(dur, 2),
                                    confidence_score=score,
                                    key_metrics=json.dumps({
                                        "velocity_reversal_delta": round(float(delta_v), 3),
                                        "approach_speed": round(float(approach_speed), 3),
                                        "braking_knee_flexion": round(float(min_k), 1),
                                    }),
                                    status="PROPOSED",
                                    proposed_movement="CHANGE_OF_DIRECTION",
                                    heuristic_score=score,
                                    human_decision="PENDING",
                                )
                            )

        return candidates

    def _detect_scissor_jumps(self, signals: list[dict], video_name: str) -> list[MovementCandidate]:
        """Detects aerial scissor leg swap during overhead rear-court movement."""
        candidates = []
        n = len(signals)
        if n < 10:
            return []

        for i in range(5, n - 8):
            s = signals[i]
            local_baseline = s.get("baseline_hip_y", 0.5)
            # Must have airborne elevation
            if s["hip_y"] is not None and (local_baseline - s["hip_y"]) >= 0.035:
                # Check lead/trail ankle swap before vs after apex
                pre_ankles = [
                    (signals[k]["ankle_l_x"] - signals[k]["ankle_r_x"])
                    for k in range(max(0, i - 6), i - 1)
                    if signals[k].get("ankle_l_x") is not None and signals[k].get("ankle_r_x") is not None
                ]
                post_ankles = [
                    (signals[k]["ankle_l_x"] - signals[k]["ankle_r_x"])
                    for k in range(i + 2, min(n, i + 8))
                    if signals[k].get("ankle_l_x") is not None and signals[k].get("ankle_r_x") is not None
                ]

                if pre_ankles and post_ankles:
                    mean_pre = float(np.mean(pre_ankles))
                    mean_post = float(np.mean(post_ankles))
                    # Clear sign reversal indicates swapping front/back legs in midair
                    if (mean_pre * mean_post < -0.015) and abs(mean_pre - mean_post) >= 0.15:
                        start_idx = max(0, i - 6)
                        end_idx = min(n - 1, i + 8)
                        t_start = signals[start_idx]["timestamp"]
                        t_end = signals[end_idx]["timestamp"]
                        dur = t_end - t_start

                        candidates.append(
                            MovementCandidate(
                                candidate_id="",
                                video_id=video_name,
                                movement_type="SCISSOR_JUMP",
                                start_time=round(t_start, 2),
                                end_time=round(t_end, 2),
                                duration=round(dur, 2),
                                confidence_score=0.85,
                                key_metrics=json.dumps({
                                    "pre_ankle_diff": round(mean_pre, 3),
                                    "post_ankle_diff": round(mean_post, 3),
                                    "swap_displacement": round(abs(mean_pre - mean_post), 3),
                                }),
                                status="PROPOSED",
                                proposed_movement="SCISSOR_JUMP",
                                heuristic_score=0.85,
                                human_decision="PENDING",
                            )
                        )
        return candidates

    def _detect_split_steps(self, signals: list[dict], video_name: str) -> list[MovementCandidate]:
        """Detects preparatory split-step bounce before shuttlecock anticipation."""
        candidates = []
        n = len(signals)
        if n < 8:
            return []

        for i in range(2, n - 5):
            s = signals[i]
            # Rapid bilateral bounce: small hop with knee flexion
            min_k = s.get("min_knee")
            asym = s.get("knee_asym", 0.0)
            stance = s.get("stance_ratio", 0.0)

            if min_k is not None and 125.0 <= min_k <= 155.0 and asym <= 15.0 and stance >= 0.65:
                # Pre-split was narrower
                pre_stances = [
                    signals[k].get("stance_ratio", 0.0)
                    for k in range(max(0, i - 4), i)
                ]
                if pre_stances and stance > np.mean(pre_stances) + 0.15:
                    start_idx = max(0, i - 3)
                    end_idx = min(n - 1, i + 4)
                    t_start = signals[start_idx]["timestamp"]
                    t_end = signals[end_idx]["timestamp"]
                    dur = t_end - t_start
                    if 0.2 <= dur <= 0.9:
                        candidates.append(
                            MovementCandidate(
                                candidate_id="",
                                video_id=video_name,
                                movement_type="FOOTWORK_SPLIT_STEP",
                                start_time=round(t_start, 2),
                                end_time=round(t_end, 2),
                                duration=round(dur, 2),
                                confidence_score=0.80,
                                key_metrics=json.dumps({
                                    "bilateral_knee_flexion": round(float(min_k), 1),
                                    "stance_expansion": round(float(stance - np.mean(pre_stances)), 2),
                                }),
                                status="PROPOSED",
                                proposed_movement="FOOTWORK_SPLIT_STEP",
                                heuristic_score=0.80,
                                human_decision="PENDING",
                            )
                        )
        return candidates

    def _merge_candidates(self, candidates: list[MovementCandidate]) -> list[MovementCandidate]:
        """
        Merges temporal overlapping candidates, resolves cross-movement conflicts,
        and marks ambiguous overlapping detections as UNCERTAIN / REVIEW_REQUIRED.
        """
        if not candidates:
            return []

        # Sort by start_time, then by end_time, then by heuristic_score descending
        candidates.sort(key=lambda c: (c.start_time, c.end_time, -c.heuristic_score))

        # Pass 1: Merge adjacent candidates of the same movement type
        pass1: list[MovementCandidate] = []
        for cand in candidates:
            if not pass1:
                pass1.append(cand)
                continue

            last = pass1[-1]
            overlap = min(last.end_time, cand.end_time) - max(last.start_time, cand.start_time)
            union = max(last.end_time, cand.end_time) - min(last.start_time, cand.start_time)
            tiou = max(0.0, overlap) / max(union, 1e-4)
            gap = cand.start_time - last.end_time

            if last.proposed_movement == cand.proposed_movement:
                if tiou >= self.config.overlap_iou_threshold or gap <= self.config.merge_gap_seconds:
                    new_start = min(last.start_time, cand.start_time)
                    new_end = max(last.end_time, cand.end_time)
                    pass1[-1] = MovementCandidate(
                        candidate_id="",
                        video_id=last.video_id,
                        movement_type=last.proposed_movement,
                        start_time=round(new_start, 2),
                        end_time=round(new_end, 2),
                        duration=round(new_end - new_start, 2),
                        confidence_score=max(last.heuristic_score, cand.heuristic_score),
                        key_metrics=last.key_metrics,
                        status=last.status,
                        proposed_movement=last.proposed_movement,
                        heuristic_score=max(last.heuristic_score, cand.heuristic_score),
                        human_decision="PENDING",
                    )
                    continue

            pass1.append(cand)

        # Pass 2: Resolve cross-type overlapping conflicts
        pass2: list[MovementCandidate] = []
        for cand in pass1:
            if not pass2:
                pass2.append(cand)
                continue

            last = pass2[-1]
            overlap = min(last.end_time, cand.end_time) - max(last.start_time, cand.start_time)
            min_dur = min(last.duration, cand.duration)
            overlap_ratio = max(0.0, overlap) / max(min_dur, 1e-4)

            if overlap_ratio >= self.config.cross_type_conflict_iou:
                # Heavy cross-type overlap conflict
                score_diff = cand.heuristic_score - last.heuristic_score

                if abs(score_diff) < 0.15:
                    # Conflicting evidence between two movement types!
                    # Mark as UNCERTAIN rather than forcing a wrong classification
                    new_start = min(last.start_time, cand.start_time)
                    new_end = max(last.end_time, cand.end_time)
                    conflict_info = {
                        "conflicting_movements": [last.proposed_movement, cand.proposed_movement],
                        "scores": [last.heuristic_score, cand.heuristic_score],
                        "reason": "Overlapping conflicting candidate detections",
                    }
                    pass2[-1] = MovementCandidate(
                        candidate_id="",
                        video_id=last.video_id,
                        movement_type="UNCERTAIN",
                        start_time=round(new_start, 2),
                        end_time=round(new_end, 2),
                        duration=round(new_end - new_start, 2),
                        confidence_score=round(max(last.heuristic_score, cand.heuristic_score) * 0.75, 2),
                        key_metrics=json.dumps(conflict_info),
                        status="REVIEW_REQUIRED",
                        proposed_movement="UNCERTAIN",
                        heuristic_score=round(max(last.heuristic_score, cand.heuristic_score) * 0.75, 2),
                        human_decision="PENDING",
                    )
                    continue
                elif score_diff > 0:
                    # Current candidate has significantly stronger evidence
                    pass2[-1] = cand
                    continue
                else:
                    # Last candidate has significantly stronger evidence; discard current candidate
                    continue

            pass2.append(cand)

        # Remove duplicate intervals (start & end within 0.25s of another candidate)
        final_list: list[MovementCandidate] = []
        for c in pass2:
            is_dup = False
            for existing in final_list:
                if abs(c.start_time - existing.start_time) < 0.25 and abs(c.end_time - existing.end_time) < 0.25:
                    is_dup = True
                    break
            if not is_dup:
                final_list.append(c)

        return final_list

    def detect_video_candidates(
        self,
        video_path: str,
        output_csv: str | None = None,
        export_clips: bool = False,
        clips_dir: str | None = None,
    ) -> list[MovementCandidate]:
        video_name = Path(video_path).name
        signals = self.extract_temporal_signals(video_path)
        candidates = self.detect_candidates_from_signals(signals, video_name)

        if output_csv:
            self.save_candidates_to_csv(candidates, output_csv)

        if export_clips and candidates:
            target_clips_dir = clips_dir or str(Path("data/clips"))
            self.export_candidate_clips(video_path, candidates, target_clips_dir)

        return candidates

    def save_candidates_to_csv(self, candidates: list[MovementCandidate], output_csv: str):
        out_path = Path(output_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CANDIDATE_CSV_FIELDS)
            writer.writeheader()
            for cand in candidates:
                writer.writerow(asdict(cand))

    def export_candidate_clips(
        self, video_path: str, candidates: list[MovementCandidate], clips_dir: str
    ):
        """
        Slices and exports proposed candidate segments as preview MP4 clips.
        """
        Path(clips_dir).mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        clean_stem = Path(video_path).stem.replace(" ", "_")

        for cand in candidates:
            clip_name = f"{cand.candidate_id}_{cand.movement_type}_{clean_stem}.mp4"
            out_clip_path = Path(clips_dir) / clip_name

            # Use ffmpeg for high-quality slice if available
            try:
                cmd = [
                    "ffmpeg", "-y",
                    "-ss", str(cand.start_time),
                    "-to", str(cand.end_time),
                    "-i", video_path,
                    "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-profile:v", "baseline",
                    str(out_clip_path)
                ]
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            except Exception:
                pass
