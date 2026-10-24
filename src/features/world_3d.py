"""
3D metric pose world landmark feature calculations.
Calculates 3D angles in metric space from MediaPipe pose_world_landmarks.
"""

from typing import Optional, Sequence
import numpy as np


def calculate_angle_3d(a: Sequence[float], b: Sequence[float], c: Sequence[float]) -> Optional[float]:
    """
    Calculates 3D angle at vertex b between vectors (a - b) and (c - b).
    Inputs: [x, y, z] vectors.
    Returns angle in degrees [0.0, 180.0].
    """
    ba = np.array([a[0] - b[0], a[1] - b[1], a[2] - b[2]], dtype=np.float64)
    bc = np.array([c[0] - b[0], c[1] - b[1], c[2] - b[2]], dtype=np.float64)

    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)

    if not np.isfinite(norm_ba) or not np.isfinite(norm_bc) or norm_ba < 1e-6 or norm_bc < 1e-6:
        return None

    cosine = np.dot(ba, bc) / (norm_ba * norm_bc)
    cosine = np.clip(cosine, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def get_knee_angle_3d(hip: Sequence[float], knee: Sequence[float], ankle: Sequence[float]) -> Optional[float]:
    """Calculates 3D knee flexion angle (hip -> knee -> ankle)."""
    return calculate_angle_3d(hip, knee, ankle)


def get_elbow_angle_3d(shoulder: Sequence[float], elbow: Sequence[float], wrist: Sequence[float]) -> Optional[float]:
    """Calculates 3D elbow extension angle (shoulder -> elbow -> wrist)."""
    return calculate_angle_3d(shoulder, elbow, wrist)


def get_torso_lean_3d(shoulder: Sequence[float], hip: Sequence[float]) -> Optional[float]:
    """
    Calculates 3D torso lean relative to vertical gravity vector [0, -1, 0].
    In MediaPipe world coordinates: +Y is down, so upward torso is -Y.
    """
    torso_vec = np.array([
        shoulder[0] - hip[0],
        shoulder[1] - hip[1],
        shoulder[2] - hip[2],
    ], dtype=np.float64)

    norm_t = np.linalg.norm(torso_vec)
    if not np.isfinite(norm_t) or norm_t < 1e-6:
        return None

    up_vector = np.array([0.0, -1.0, 0.0], dtype=np.float64)
    cosine = np.dot(torso_vec, up_vector) / norm_t
    cosine = np.clip(cosine, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))
