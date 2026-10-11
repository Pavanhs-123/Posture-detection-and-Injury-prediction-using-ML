"""
Skeleton visualization and overlay rendering utilities.
Draws joints, bones, and angle annotations onto BGR image frames.
"""

from typing import Optional, Dict, Any, Sequence
import cv2
import numpy as np


# Standard MediaPipe Pose landmark indices
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28


CONNECTIONS = [
    (LEFT_SHOULDER, LEFT_ELBOW),
    (LEFT_ELBOW, LEFT_WRIST),
    (LEFT_SHOULDER, LEFT_HIP),
    (LEFT_HIP, LEFT_KNEE),
    (LEFT_KNEE, LEFT_ANKLE),
    (RIGHT_SHOULDER, RIGHT_ELBOW),
    (RIGHT_ELBOW, RIGHT_WRIST),
    (RIGHT_SHOULDER, RIGHT_HIP),
    (RIGHT_HIP, RIGHT_KNEE),
    (RIGHT_KNEE, RIGHT_ANKLE),
    (LEFT_SHOULDER, RIGHT_SHOULDER),
    (LEFT_HIP, RIGHT_HIP),
]


def draw_skeleton_overlay(
    frame: np.ndarray,
    landmarks: Sequence[Any],
    angles: Optional[Dict[str, float]] = None,
    color_bones=(0, 215, 255),  # Vibrant gold
    color_joints=(0, 255, 128),  # Vibrant green
    thickness=2,
    radius=5,
) -> np.ndarray:
    """
    Renders skeletal joints and bones on a copy of the input frame.
    Optionally annotates key joint angles if provided.
    """
    if landmarks is None or len(landmarks) < 29:
        return frame

    h, w = frame.shape[:2]
    canvas = frame.copy()

    # Convert landmarks to pixel coordinates
    pts = {}
    for idx in range(len(landmarks)):
        lm = landmarks[idx]
        px = int(lm.x * w)
        py = int(lm.y * h)
        pts[idx] = (px, py)

    # Draw bones
    for idx1, idx2 in CONNECTIONS:
        if idx1 in pts and idx2 in pts:
            cv2.line(canvas, pts[idx1], pts[idx2], color_bones, thickness, cv2.LINE_AA)

    # Draw joints
    for idx, pt in pts.items():
        if idx in (LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_ELBOW, RIGHT_ELBOW,
                   LEFT_WRIST, RIGHT_WRIST, LEFT_HIP, RIGHT_HIP,
                   LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE):
            cv2.circle(canvas, pt, radius, color_joints, -1, cv2.LINE_AA)
            cv2.circle(canvas, pt, radius + 1, (0, 0, 0), 1, cv2.LINE_AA)

    # Annotate joint angles if provided
    if angles:
        lk = angles.get("left_knee")
        if lk is not None and LEFT_KNEE in pts:
            pt = pts[LEFT_KNEE]
            cv2.putText(canvas, f"{lk:.0f}°", (pt[0] + 8, pt[1] - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(canvas, f"{lk:.0f}°", (pt[0] + 8, pt[1] - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)

        rk = angles.get("right_knee")
        if rk is not None and RIGHT_KNEE in pts:
            pt = pts[RIGHT_KNEE]
            cv2.putText(canvas, f"{rk:.0f}°", (pt[0] + 8, pt[1] - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(canvas, f"{rk:.0f}°", (pt[0] + 8, pt[1] - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)

    return canvas
