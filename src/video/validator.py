"""
Video validation and inspection utilities.
Verifies file existence, decodability, duration, frame count, FPS, and resolution.
"""

import os
from pathlib import Path
from typing import Dict, Any
import cv2

SUPPORTED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}

def validate_video(video_path: str) -> Dict[str, Any]:
    """
    Validates an input video file and extracts core metadata.

    Returns a dictionary containing:
        is_valid: bool
        error: Optional[str]
        duration: float
        fps: float
        width: int
        height: int
        frame_count: int
        codec: str
        file_size_mb: float
    """
    path = Path(video_path)
    result = {
        "is_valid": False,
        "error": None,
        "duration": 0.0,
        "fps": 0.0,
        "width": 0,
        "height": 0,
        "frame_count": 0,
        "codec": "",
        "file_size_mb": 0.0,
    }

    if not path.exists():
        result["error"] = f"File not found: {video_path}"
        return result

    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        result["error"] = f"Unsupported video extension: {path.suffix}. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        return result

    size_bytes = path.stat().st_size
    if size_bytes == 0:
        result["error"] = "Video file is empty (0 bytes)."
        return result

    result["file_size_mb"] = round(size_bytes / (1024 * 1024), 2)

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        result["error"] = "Failed to open video file. File may be corrupted or codec unsupported."
        return result

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps > 240:
        fps = 30.0  # Fallback reasonable default

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if width <= 0 or height <= 0:
        cap.release()
        result["error"] = "Invalid video dimensions (width or height is 0)."
        return result

    # Verify at least the first frame is decodable
    ret, frame = cap.read()
    if not ret or frame is None:
        cap.release()
        result["error"] = "Unable to read initial frame from video."
        return result

    # Check codec fourcc
    fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc_str = "".join([chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)])

    cap.release()

    duration = frame_count / fps if frame_count > 0 else 0.0
    if duration < 0.2:
        result["error"] = f"Video is too short ({duration:.2f}s). Minimum required duration is 0.2s."
        return result

    result["is_valid"] = True
    result["fps"] = round(fps, 2)
    result["width"] = width
    result["height"] = height
    result["frame_count"] = frame_count
    result["duration"] = round(duration, 2)
    result["codec"] = fourcc_str.strip() or "Unknown"

    return result
