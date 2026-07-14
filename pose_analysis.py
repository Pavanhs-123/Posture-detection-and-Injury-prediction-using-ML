from dataclasses import dataclass
from pathlib import Path
import urllib.request

import numpy as np

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)
MODEL_PATH = Path(__file__).with_name("pose_landmarker_lite.task")


@dataclass(frozen=True)
class PoseFeatures:
    left_knee_angle: float | None
    right_knee_angle: float | None
    knee_angle: float | None
    left_elbow_angle: float | None
    right_elbow_angle: float | None
    elbow_angle: float | None
    torso_lean: float | None


def ensure_model_file() -> None:
    if MODEL_PATH.exists():
        return

    print(f"Downloading pose model to {MODEL_PATH}...")
    with urllib.request.urlopen(MODEL_URL) as response, MODEL_PATH.open("wb") as model_file:
        model_file.write(response.read())


def calculate_angle(a: list[float], b: list[float], c: list[float]) -> float:
    a_vec = np.array(a)
    b_vec = np.array(b)
    c_vec = np.array(c)

    radians = np.arctan2(c_vec[1] - b_vec[1], c_vec[0] - b_vec[0]) - np.arctan2(
        a_vec[1] - b_vec[1], a_vec[0] - b_vec[0]
    )
    angle = abs(radians * 180.0 / np.pi)

    if angle > 180:
        angle = 360 - angle

    return float(angle)


def average_angles(values: list[float | None]) -> float | None:
    valid = [value for value in values if value is not None]
    if not valid:
        return None
    return float(sum(valid) / len(valid))


def get_point(landmarks: list, idx) -> list[float]:
    point = landmarks[idx]
    return [point.x, point.y]


def get_knee_angle(landmarks: list, pose_landmark) -> tuple[float | None, float | None, float | None]:
    left = calculate_angle(
        get_point(landmarks, pose_landmark.LEFT_HIP),
        get_point(landmarks, pose_landmark.LEFT_KNEE),
        get_point(landmarks, pose_landmark.LEFT_ANKLE),
    )
    right = calculate_angle(
        get_point(landmarks, pose_landmark.RIGHT_HIP),
        get_point(landmarks, pose_landmark.RIGHT_KNEE),
        get_point(landmarks, pose_landmark.RIGHT_ANKLE),
    )
    return left, right, average_angles([left, right])


def get_elbow_angle(landmarks: list, pose_landmark) -> tuple[float | None, float | None, float | None]:
    left = calculate_angle(
        get_point(landmarks, pose_landmark.LEFT_SHOULDER),
        get_point(landmarks, pose_landmark.LEFT_ELBOW),
        get_point(landmarks, pose_landmark.LEFT_WRIST),
    )
    right = calculate_angle(
        get_point(landmarks, pose_landmark.RIGHT_SHOULDER),
        get_point(landmarks, pose_landmark.RIGHT_ELBOW),
        get_point(landmarks, pose_landmark.RIGHT_WRIST),
    )
    return left, right, average_angles([left, right])


def get_torso_lean(landmarks: list, pose_landmark) -> float | None:
    left_shoulder = np.array(get_point(landmarks, pose_landmark.LEFT_SHOULDER))
    right_shoulder = np.array(get_point(landmarks, pose_landmark.RIGHT_SHOULDER))
    left_hip = np.array(get_point(landmarks, pose_landmark.LEFT_HIP))
    right_hip = np.array(get_point(landmarks, pose_landmark.RIGHT_HIP))

    shoulder_center = (left_shoulder + right_shoulder) / 2.0
    hip_center = (left_hip + right_hip) / 2.0
    torso_vector = shoulder_center - hip_center

    torso_length = np.linalg.norm(torso_vector)
    if torso_length < 1e-6:
        return None

    lean = np.degrees(np.arctan2(abs(torso_vector[0]), abs(torso_vector[1]) + 1e-6))
    return float(lean)


def extract_pose_features(landmarks: list, pose_landmark) -> PoseFeatures:
    left_knee, right_knee, knee_angle = get_knee_angle(landmarks, pose_landmark)
    left_elbow, right_elbow, elbow_angle = get_elbow_angle(landmarks, pose_landmark)

    return PoseFeatures(
        left_knee_angle=left_knee,
        right_knee_angle=right_knee,
        knee_angle=knee_angle,
        left_elbow_angle=left_elbow,
        right_elbow_angle=right_elbow,
        elbow_angle=elbow_angle,
        torso_lean=get_torso_lean(landmarks, pose_landmark),
    )