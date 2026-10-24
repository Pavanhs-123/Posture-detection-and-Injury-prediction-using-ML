"""
MediaPipe PoseLandmarker estimator wrapper.
Loads task bundle, detects 33 landmarks and metric 3D world landmarks,
and extracts confidence and presence scores.
"""

import os
import urllib.request
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple
import cv2
import numpy as np
import mediapipe as mp

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)
_ROOT_DIR = Path(__file__).resolve().parent.parent.parent
_ASSETS_PATH = _ROOT_DIR / "assets" / "pose_landmarker_lite.task"
DEFAULT_MODEL_PATH = _ASSETS_PATH if _ASSETS_PATH.exists() else _ROOT_DIR / "pose_landmarker_lite.task"


def ensure_model_file(model_path: Optional[Path] = None) -> Path:
    """Ensures the MediaPipe task asset exists locally, downloading if necessary."""
    target_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
    if not target_path.exists():
        target_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading pose_landmarker_lite.task to {target_path}...")
        urllib.request.urlretrieve(MODEL_URL, str(target_path))
        print("Download complete.")
    return target_path


@dataclass
class FramePoseResult:
    frame_index: int
    timestamp_sec: float
    has_pose: bool
    landmarks: Optional[List[Any]]  # 33 landmarks in normalized [0, 1]
    world_landmarks: Optional[List[Any]]  # 33 landmarks in hip-relative metric meters
    overall_confidence: float
    left_knee_conf: float
    right_knee_conf: float
    left_elbow_conf: float
    right_elbow_conf: float
    torso_conf: float


class PoseEstimator:
    """MediaPipe Tasks PoseLandmarker wrapper for video stream inference."""

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = ensure_model_file(Path(model_path) if model_path else None)
        BaseOptions = mp.tasks.BaseOptions
        PoseLandmarker = mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(self.model_path)),
            running_mode=VisionRunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.landmarker = PoseLandmarker.create_from_options(options)

    def close(self):
        if hasattr(self, "landmarker") and self.landmarker is not None:
            self.landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def process_frame(
        self,
        bgr_frame: np.ndarray,
        timestamp_ms: int,
        frame_index: int = 0,
    ) -> FramePoseResult:
        """Processes a single BGR frame and returns landmark detections and confidence."""
        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection = self.landmarker.detect_for_video(mp_image, max(0, timestamp_ms))

        if not detection.pose_landmarks:
            return FramePoseResult(
                frame_index=frame_index,
                timestamp_sec=timestamp_ms / 1000.0,
                has_pose=False,
                landmarks=None,
                world_landmarks=None,
                overall_confidence=0.0,
                left_knee_conf=0.0,
                right_knee_conf=0.0,
                left_elbow_conf=0.0,
                right_elbow_conf=0.0,
                torso_conf=0.0,
            )

        lms = detection.pose_landmarks[0]
        wlms = detection.pose_world_landmarks[0] if detection.pose_world_landmarks else None

        # Key landmark IDs (PoseLandmark indices):
        # 11: L_shoulder, 12: R_shoulder, 13: L_elbow, 14: R_elbow, 15: L_wrist, 16: R_wrist
        # 23: L_hip, 24: R_hip, 25: L_knee, 26: R_knee, 27: L_ankle, 28: R_ankle
        def get_vis(idx):
            if idx < len(lms):
                return getattr(lms[idx], "visibility", 1.0)
            return 0.0

        lk_c = min(get_vis(23), get_vis(25), get_vis(27))
        rk_c = min(get_vis(24), get_vis(26), get_vis(28))
        le_c = min(get_vis(11), get_vis(13), get_vis(15))
        re_c = min(get_vis(12), get_vis(14), get_vis(16))
        torso_c = min(get_vis(11), get_vis(12), get_vis(23), get_vis(24))
        overall_c = float(np.mean([getattr(lm, "visibility", 1.0) for lm in lms]))

        return FramePoseResult(
            frame_index=frame_index,
            timestamp_sec=timestamp_ms / 1000.0,
            has_pose=True,
            landmarks=lms,
            world_landmarks=wlms,
            overall_confidence=round(overall_c, 3),
            left_knee_conf=round(lk_c, 3),
            right_knee_conf=round(rk_c, 3),
            left_elbow_conf=round(le_c, 3),
            right_elbow_conf=round(re_c, 3),
            torso_conf=round(torso_c, 3),
        )
