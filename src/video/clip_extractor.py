"""
Clip extraction and key frame generation utilities for movement events.
Slices source videos into individual event clips and saves representative key frames.
"""

import os
import subprocess
from pathlib import Path
from typing import Optional, Callable
import cv2
import numpy as np


def extract_subclip(
    source_video_path: str,
    start_sec: float,
    end_sec: float,
    output_clip_path: str,
    overlay_fn: Optional[Callable[[np.ndarray, float], np.ndarray]] = None,
) -> bool:
    """
    Extracts a temporal sub-clip [start_sec, end_sec] from source_video_path.
    Optionally applies an overlay function frame-by-frame.
    """
    src = Path(source_video_path)
    dst = Path(output_clip_path)
    dst.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        return False

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = total_frames / fps if total_frames > 0 else 0.0

    start_sec = max(0.0, start_sec)
    if video_duration > 0:
        end_sec = min(video_duration, end_sec)
    if end_sec <= start_sec:
        end_sec = start_sec + 0.5

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    temp_dst = dst.parent / f"temp_{dst.name}"
    writer = cv2.VideoWriter(str(temp_dst), fourcc, fps, (width, height))

    frame_idx = 0
    extracted_any = False

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        cur_sec = frame_idx / fps
        if cur_sec < start_sec:
            frame_idx += 1
            continue
        if cur_sec > end_sec:
            break

        if overlay_fn is not None:
            frame = overlay_fn(frame, cur_sec)

        writer.write(frame)
        extracted_any = True
        frame_idx += 1

    cap.release()
    writer.release()

    if not extracted_any:
        if temp_dst.exists():
            temp_dst.unlink()
        return False

    # Attempt H.264 web-compatible transcode if ffmpeg is available
    transcoded = False
    try:
        cmd = [
            "ffmpeg", "-y", "-i", str(temp_dst),
            "-vcodec", "libx264", "-pix_fmt", "yuv420p",
            "-crf", "23", "-preset", "veryfast",
            "-movflags", "+faststart",
            str(dst)
        ]
        ret_code = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode
        if ret_code == 0 and dst.exists() and dst.stat().st_size > 0:
            transcoded = True
            temp_dst.unlink()
    except Exception:
        pass

    if not transcoded:
        if temp_dst.exists():
            if dst.exists():
                dst.unlink()
            temp_dst.rename(dst)

    return dst.exists() and dst.stat().st_size > 0


def extract_keyframe(
    source_video_path: str,
    timestamp_sec: float,
    output_image_path: str,
    overlay_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> bool:
    """
    Extracts a representative key frame image at timestamp_sec and saves as JPEG.
    """
    src = Path(source_video_path)
    dst = Path(output_image_path)
    dst.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        return False

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    try:
        timestamp_sec = float(timestamp_sec)
    except (TypeError, ValueError):
        timestamp_sec = 0.0
    if not np.isfinite(timestamp_sec):
        timestamp_sec = 0.0
    target_frame = max(0, int(round(timestamp_sec * fps)))
    if frame_count > 0:
        target_frame = min(target_frame, frame_count - 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    ret, frame = cap.read()

    if not ret or frame is None:
        # Fallback to frame 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap.read()

    cap.release()

    if not ret or frame is None:
        return False

    if overlay_fn is not None:
        frame = overlay_fn(frame)

    cv2.imwrite(str(dst), frame)
    return dst.exists() and dst.stat().st_size > 0
