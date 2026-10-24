"""
Aspect-ratio corrected 2D joint angle calculations.
Scales normalized coordinates to Euclidean pixel space before vector dot product calculation.
"""

from typing import Optional, Tuple
import numpy as np


def calculate_angle_2d(
    a: Tuple[float, float],
    b: Tuple[float, float],
    c: Tuple[float, float],
    image_dimensions: Optional[Tuple[int, int]] = None,
) -> Optional[float]:
    """
    Calculates the 2D joint angle at vertex b between vectors (a - b) and (c - b).
    If image_dimensions=(width, height) is provided, coordinates are scaled to pixel space.
    Returns angle in degrees [0.0, 180.0].
    """
    if image_dimensions is not None:
        w, h = image_dimensions
        if w > 0 and h > 0:
            a = (a[0] * w, a[1] * h)
            b = (b[0] * w, b[1] * h)
            c = (c[0] * w, c[1] * h)

    ba = np.array([a[0] - b[0], a[1] - b[1]], dtype=np.float64)
    bc = np.array([c[0] - b[0], c[1] - b[1]], dtype=np.float64)

    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)

    if not np.isfinite(norm_ba) or not np.isfinite(norm_bc) or norm_ba < 1e-6 or norm_bc < 1e-6:
        return None

    cosine = np.dot(ba, bc) / (norm_ba * norm_bc)
    cosine = np.clip(cosine, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def get_knee_angle_2d(
    hip: Tuple[float, float],
    knee: Tuple[float, float],
    ankle: Tuple[float, float],
    image_dimensions: Optional[Tuple[int, int]] = None,
) -> Optional[float]:
    """Calculates 2D knee flexion angle (hip -> knee -> ankle)."""
    return calculate_angle_2d(hip, knee, ankle, image_dimensions=image_dimensions)


def get_elbow_angle_2d(
    shoulder: Tuple[float, float],
    elbow: Tuple[float, float],
    wrist: Tuple[float, float],
    image_dimensions: Optional[Tuple[int, int]] = None,
) -> Optional[float]:
    """Calculates 2D elbow extension angle (shoulder -> elbow -> wrist)."""
    return calculate_angle_2d(shoulder, elbow, wrist, image_dimensions=image_dimensions)


def get_torso_lean_2d(
    shoulder: Tuple[float, float],
    hip: Tuple[float, float],
    image_dimensions: Optional[Tuple[int, int]] = None,
) -> Optional[float]:
    """
    Calculates 2D torso lean angle relative to vertical axis (0° = upright).
    """
    if image_dimensions is not None:
        w, h = image_dimensions
        if w > 0 and h > 0:
            shoulder = (shoulder[0] * w, shoulder[1] * h)
            hip = (hip[0] * w, hip[1] * h)

    dx = shoulder[0] - hip[0]
    dy = shoulder[1] - hip[1]

    if not np.isfinite(dx) or not np.isfinite(dy) or (abs(dx) < 1e-6 and abs(dy) < 1e-6):
        return None

    angle_rad = np.arctan2(abs(dx), abs(dy) if abs(dy) > 1e-6 else 1e-6)
    return float(np.degrees(angle_rad))
