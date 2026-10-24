from dataclasses import dataclass
from pathlib import Path
import urllib.request

import numpy as np

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)
_ASSETS_MODEL = Path(__file__).parent / "assets" / "pose_landmarker_lite.task"
MODEL_PATH = _ASSETS_MODEL if _ASSETS_MODEL.exists() else Path(__file__).with_name("pose_landmarker_lite.task")


@dataclass(frozen=True)
class PoseFeatures:
    left_knee_angle: float | None
    right_knee_angle: float | None
    knee_angle: float | None
    left_elbow_angle: float | None
    right_elbow_angle: float | None
    elbow_angle: float | None
    torso_lean: float | None


@dataclass(frozen=True)
class PoseFeatures3D:
    left_knee_angle: float | None
    right_knee_angle: float | None
    knee_angle: float | None
    left_elbow_angle: float | None
    right_elbow_angle: float | None
    elbow_angle: float | None
    torso_lean: float | None


@dataclass(frozen=True)
class LandmarkQuality:
    left_knee_conf: float | None
    right_knee_conf: float | None
    left_elbow_conf: float | None
    right_elbow_conf: float | None
    torso_conf: float | None
    overall_conf: float | None


def ensure_model_file() -> None:
    if MODEL_PATH.exists():
        return

    print(f"Downloading pose model to {MODEL_PATH}...")
    with urllib.request.urlopen(MODEL_URL) as response, MODEL_PATH.open("wb") as model_file:
        model_file.write(response.read())


def calculate_angle(
    a: list[float],
    b: list[float],
    c: list[float],
    image_dimensions: tuple[int, int] | None = None,
) -> float:
    if a is None or b is None or c is None:
        return 0.0
    a_vec = np.array(a[:2], dtype=float)
    b_vec = np.array(b[:2], dtype=float)
    c_vec = np.array(c[:2], dtype=float)

    # Correct for non-square pixel aspect ratio if image dimensions (width, height) are provided
    if image_dimensions is not None and len(image_dimensions) == 2:
        width, height = image_dimensions
        if width > 0 and height > 0:
            scale = np.array([float(width), float(height)], dtype=float)
            a_vec = a_vec * scale
            b_vec = b_vec * scale
            c_vec = c_vec * scale

    radians = np.arctan2(c_vec[1] - b_vec[1], c_vec[0] - b_vec[0]) - np.arctan2(
        a_vec[1] - b_vec[1], a_vec[0] - b_vec[0]
    )
    angle = abs(radians * 180.0 / np.pi)

    if angle > 180:
        angle = 360 - angle

    return float(angle)


def calculate_angle_3d(a: list[float], b: list[float], c: list[float]) -> float | None:
    """
    Calculates the 3D angle in degrees formed by three points (A -> B -> C) at vertex B.
    Handles degenerate vectors (zero length) and numerical precision clamping.
    """
    if a is None or b is None or c is None:
        return None

    a_vec = np.array(a[:3], dtype=float)
    b_vec = np.array(b[:3], dtype=float)
    c_vec = np.array(c[:3], dtype=float)

    ba = a_vec - b_vec
    bc = c_vec - b_vec

    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)

    if norm_ba < 1e-6 or norm_bc < 1e-6:
        return None

    dot_product = np.dot(ba, bc)
    cosine = np.clip(dot_product / (norm_ba * norm_bc), -1.0, 1.0)
    angle = np.degrees(np.arccos(cosine))

    return float(angle)


def average_angles(values: list[float | None]) -> float | None:
    valid = [value for value in values if value is not None]
    if not valid:
        return None
    return float(sum(valid) / len(valid))


def get_landmark_confidence(point) -> float:
    """Extracts minimum of visibility and presence confidence score."""
    vis = getattr(point, "visibility", 1.0)
    pres = getattr(point, "presence", 1.0)
    v = 1.0 if vis is None else float(vis)
    p = 1.0 if pres is None else float(pres)
    return float(min(v, p))


def get_point(landmarks: list, idx, min_confidence: float | None = None) -> list[float] | None:
    """Returns [x, y] point with optional confidence check."""
    if idx >= len(landmarks):
        return None
    point = landmarks[idx]
    if min_confidence is not None and get_landmark_confidence(point) < min_confidence:
        return None
    return [float(point.x), float(point.y)]


def get_point_3d(landmarks: list, idx, min_confidence: float | None = None) -> list[float] | None:
    """Returns [x, y, z] 3D world landmark point with optional confidence check."""
    if idx >= len(landmarks):
        return None
    point = landmarks[idx]
    if min_confidence is not None and get_landmark_confidence(point) < min_confidence:
        return None
    return [float(point.x), float(point.y), float(point.z)]


def get_knee_angle(
    landmarks: list,
    pose_landmark,
    min_confidence: float | None = None,
    image_dimensions: tuple[int, int] | None = None,
) -> tuple[float | None, float | None, float | None]:
    hip_l = get_point(landmarks, pose_landmark.LEFT_HIP, min_confidence)
    knee_l = get_point(landmarks, pose_landmark.LEFT_KNEE, min_confidence)
    ankle_l = get_point(landmarks, pose_landmark.LEFT_ANKLE, min_confidence)
    left = calculate_angle(hip_l, knee_l, ankle_l, image_dimensions) if hip_l and knee_l and ankle_l else None

    hip_r = get_point(landmarks, pose_landmark.RIGHT_HIP, min_confidence)
    knee_r = get_point(landmarks, pose_landmark.RIGHT_KNEE, min_confidence)
    ankle_r = get_point(landmarks, pose_landmark.RIGHT_ANKLE, min_confidence)
    right = calculate_angle(hip_r, knee_r, ankle_r, image_dimensions) if hip_r and knee_r and ankle_r else None

    return left, right, average_angles([left, right])


def get_knee_angle_3d(landmarks_3d: list, pose_landmark, min_confidence: float | None = None) -> tuple[float | None, float | None, float | None]:
    hip_l = get_point_3d(landmarks_3d, pose_landmark.LEFT_HIP, min_confidence)
    knee_l = get_point_3d(landmarks_3d, pose_landmark.LEFT_KNEE, min_confidence)
    ankle_l = get_point_3d(landmarks_3d, pose_landmark.LEFT_ANKLE, min_confidence)
    left = calculate_angle_3d(hip_l, knee_l, ankle_l) if hip_l and knee_l and ankle_l else None

    hip_r = get_point_3d(landmarks_3d, pose_landmark.RIGHT_HIP, min_confidence)
    knee_r = get_point_3d(landmarks_3d, pose_landmark.RIGHT_KNEE, min_confidence)
    ankle_r = get_point_3d(landmarks_3d, pose_landmark.RIGHT_ANKLE, min_confidence)
    right = calculate_angle_3d(hip_r, knee_r, ankle_r) if hip_r and knee_r and ankle_r else None

    return left, right, average_angles([left, right])


def get_elbow_angle(
    landmarks: list,
    pose_landmark,
    min_confidence: float | None = None,
    image_dimensions: tuple[int, int] | None = None,
) -> tuple[float | None, float | None, float | None]:
    sh_l = get_point(landmarks, pose_landmark.LEFT_SHOULDER, min_confidence)
    el_l = get_point(landmarks, pose_landmark.LEFT_ELBOW, min_confidence)
    wr_l = get_point(landmarks, pose_landmark.LEFT_WRIST, min_confidence)
    left = calculate_angle(sh_l, el_l, wr_l, image_dimensions) if sh_l and el_l and wr_l else None

    sh_r = get_point(landmarks, pose_landmark.RIGHT_SHOULDER, min_confidence)
    el_r = get_point(landmarks, pose_landmark.RIGHT_ELBOW, min_confidence)
    wr_r = get_point(landmarks, pose_landmark.RIGHT_WRIST, min_confidence)
    right = calculate_angle(sh_r, el_r, wr_r, image_dimensions) if sh_r and el_r and wr_r else None

    return left, right, average_angles([left, right])


def get_elbow_angle_3d(landmarks_3d: list, pose_landmark, min_confidence: float | None = None) -> tuple[float | None, float | None, float | None]:
    sh_l = get_point_3d(landmarks_3d, pose_landmark.LEFT_SHOULDER, min_confidence)
    el_l = get_point_3d(landmarks_3d, pose_landmark.LEFT_ELBOW, min_confidence)
    wr_l = get_point_3d(landmarks_3d, pose_landmark.LEFT_WRIST, min_confidence)
    left = calculate_angle_3d(sh_l, el_l, wr_l) if sh_l and el_l and wr_l else None

    sh_r = get_point_3d(landmarks_3d, pose_landmark.RIGHT_SHOULDER, min_confidence)
    el_r = get_point_3d(landmarks_3d, pose_landmark.RIGHT_ELBOW, min_confidence)
    wr_r = get_point_3d(landmarks_3d, pose_landmark.RIGHT_WRIST, min_confidence)
    right = calculate_angle_3d(sh_r, el_r, wr_r) if sh_r and el_r and wr_r else None

    return left, right, average_angles([left, right])


def get_torso_lean(
    landmarks: list,
    pose_landmark,
    min_confidence: float | None = None,
    image_dimensions: tuple[int, int] | None = None,
) -> float | None:
    ls = get_point(landmarks, pose_landmark.LEFT_SHOULDER, min_confidence)
    rs = get_point(landmarks, pose_landmark.RIGHT_SHOULDER, min_confidence)
    lh = get_point(landmarks, pose_landmark.LEFT_HIP, min_confidence)
    rh = get_point(landmarks, pose_landmark.RIGHT_HIP, min_confidence)

    if not (ls and rs and lh and rh):
        return None

    left_shoulder = np.array(ls, dtype=float)
    right_shoulder = np.array(rs, dtype=float)
    left_hip = np.array(lh, dtype=float)
    right_hip = np.array(rh, dtype=float)

    # Scale to pixels if dimensions provided
    if image_dimensions is not None and len(image_dimensions) == 2:
        width, height = image_dimensions
        if width > 0 and height > 0:
            scale = np.array([float(width), float(height)], dtype=float)
            left_shoulder = left_shoulder * scale
            right_shoulder = right_shoulder * scale
            left_hip = left_hip * scale
            right_hip = right_hip * scale

    shoulder_center = (left_shoulder + right_shoulder) / 2.0
    hip_center = (left_hip + right_hip) / 2.0
    torso_vector = shoulder_center - hip_center

    torso_length = np.linalg.norm(torso_vector)
    if torso_length < 1e-6:
        return None

    lean = np.degrees(np.arctan2(abs(torso_vector[0]), abs(torso_vector[1]) + 1e-6))
    return float(lean)


def get_torso_lean_3d(landmarks_3d: list, pose_landmark, min_confidence: float | None = None) -> float | None:
    """
    Calculates 3D torso lean (inclination from the vertical gravity axis) in degrees.
    In world coordinates, Y is aligned vertically; deviation is in the X-Z horizontal plane.
    """
    ls = get_point_3d(landmarks_3d, pose_landmark.LEFT_SHOULDER, min_confidence)
    rs = get_point_3d(landmarks_3d, pose_landmark.RIGHT_SHOULDER, min_confidence)
    lh = get_point_3d(landmarks_3d, pose_landmark.LEFT_HIP, min_confidence)
    rh = get_point_3d(landmarks_3d, pose_landmark.RIGHT_HIP, min_confidence)

    if not (ls and rs and lh and rh):
        return None

    left_shoulder = np.array(ls)
    right_shoulder = np.array(rs)
    left_hip = np.array(lh)
    right_hip = np.array(rh)

    shoulder_center = (left_shoulder + right_shoulder) / 2.0
    hip_center = (left_hip + right_hip) / 2.0
    torso_vector = shoulder_center - hip_center

    torso_length = np.linalg.norm(torso_vector)
    if torso_length < 1e-6:
        return None

    horizontal_dev = np.sqrt(torso_vector[0]**2 + torso_vector[2]**2)
    vertical_dev = abs(torso_vector[1])

    lean = np.degrees(np.arctan2(horizontal_dev, vertical_dev + 1e-6))
    return float(lean)


def extract_pose_features(
    landmarks: list,
    pose_landmark,
    image_dimensions: tuple[int, int] | None = None,
) -> PoseFeatures:
    left_knee, right_knee, knee_angle = get_knee_angle(landmarks, pose_landmark, image_dimensions=image_dimensions)
    left_elbow, right_elbow, elbow_angle = get_elbow_angle(landmarks, pose_landmark, image_dimensions=image_dimensions)

    return PoseFeatures(
        left_knee_angle=left_knee,
        right_knee_angle=right_knee,
        knee_angle=knee_angle,
        left_elbow_angle=left_elbow,
        right_elbow_angle=right_elbow,
        elbow_angle=elbow_angle,
        torso_lean=get_torso_lean(landmarks, pose_landmark, image_dimensions=image_dimensions),
    )


def extract_pose_features_3d(landmarks_3d: list, pose_landmark, min_confidence: float | None = 0.5) -> PoseFeatures3D:
    left_knee, right_knee, knee_angle = get_knee_angle_3d(landmarks_3d, pose_landmark, min_confidence)
    left_elbow, right_elbow, elbow_angle = get_elbow_angle_3d(landmarks_3d, pose_landmark, min_confidence)

    return PoseFeatures3D(
        left_knee_angle=left_knee,
        right_knee_angle=right_knee,
        knee_angle=knee_angle,
        left_elbow_angle=left_elbow,
        right_elbow_angle=right_elbow,
        elbow_angle=elbow_angle,
        torso_lean=get_torso_lean_3d(landmarks_3d, pose_landmark, min_confidence),
    )


def calculate_landmark_quality(landmarks: list, pose_landmark) -> LandmarkQuality:
    """Computes joint-specific confidence indicators for landmark tracking quality."""
    def conf_triplet(idx1, idx2, idx3):
        c1 = get_landmark_confidence(landmarks[idx1]) if idx1 < len(landmarks) else 0.0
        c2 = get_landmark_confidence(landmarks[idx2]) if idx2 < len(landmarks) else 0.0
        c3 = get_landmark_confidence(landmarks[idx3]) if idx3 < len(landmarks) else 0.0
        return round(float(min(c1, c2, c3)), 3)

    lk = conf_triplet(pose_landmark.LEFT_HIP, pose_landmark.LEFT_KNEE, pose_landmark.LEFT_ANKLE)
    rk = conf_triplet(pose_landmark.RIGHT_HIP, pose_landmark.RIGHT_KNEE, pose_landmark.RIGHT_ANKLE)
    le = conf_triplet(pose_landmark.LEFT_SHOULDER, pose_landmark.LEFT_ELBOW, pose_landmark.LEFT_WRIST)
    re = conf_triplet(pose_landmark.RIGHT_SHOULDER, pose_landmark.RIGHT_ELBOW, pose_landmark.RIGHT_WRIST)

    # Torso: all 4 corners
    torso_lms = [pose_landmark.LEFT_SHOULDER, pose_landmark.RIGHT_SHOULDER, pose_landmark.LEFT_HIP, pose_landmark.RIGHT_HIP]
    torso_confs = [get_landmark_confidence(landmarks[i]) for i in torso_lms if i < len(landmarks)]
    torso_q = round(float(min(torso_confs)), 3) if torso_confs else 0.0

    all_confs = [get_landmark_confidence(lm) for lm in landmarks]
    overall = round(float(np.mean(all_confs)), 3) if all_confs else 0.0

    return LandmarkQuality(
        left_knee_conf=lk,
        right_knee_conf=rk,
        left_elbow_conf=le,
        right_elbow_conf=re,
        torso_conf=torso_q,
        overall_conf=overall
    )