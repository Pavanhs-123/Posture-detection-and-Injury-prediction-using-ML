import argparse
import csv
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from pose_analysis import MODEL_PATH, PoseFeatures, ensure_model_file, extract_pose_features


FIELDNAMES = [
    "video_name",
    "clip_start_sec",
    "clip_end_sec",
    "source_fps",
    "sample_fps",
    "frame_index",
    "timestamp_sec",
    "label",
    "left_knee_angle",
    "right_knee_angle",
    "knee_angle",
    "left_elbow_angle",
    "right_elbow_angle",
    "elbow_angle",
    "torso_lean",
    "pose_detected",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a labeled pose dataset from a video clip")
    parser.add_argument("--video", required=True, help="Path to the input video")
    parser.add_argument(
        "--output-csv",
        default="pose_data.csv",
        help="CSV file to store labeled frame data",
    )
    parser.add_argument("--start-time", type=float, help="Clip start time in seconds")
    parser.add_argument("--end-time", type=float, help="Clip end time in seconds")
    parser.add_argument(
        "--sample-fps",
        type=float,
        help="Number of frames per second to label from the selected clip",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to an existing CSV instead of overwriting it",
    )
    return parser.parse_args()


def prompt_float(prompt: str, minimum: float | None = None) -> float:
    while True:
        raw = input(prompt).strip()
        try:
            value = float(raw)
        except ValueError:
            print("Please enter a valid number.")
            continue

        if minimum is not None and value < minimum:
            print(f"Please enter a value greater than or equal to {minimum}.")
            continue

        return value


def format_value(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.2f}"


def normalize_label(choice: str) -> str | None:
    choice = choice.strip().lower()
    if choice == "1":
        return "low"
    if choice == "2":
        return "medium"
    if choice == "3":
        return "high"
    if choice == "4":
        custom_label = input("Enter custom label: ").strip()
        return custom_label or None
    if choice in {"q", "quit"}:
        return "__quit__"
    return None


def draw_marker(frame, point, color, radius=5):
    cv2.circle(frame, point, radius, color, -1)


def draw_pose_skeleton(frame, landmarks, pose_landmark) -> None:
    height, width = frame.shape[:2]

    points = {
        "left_hip": (int(landmarks[pose_landmark.LEFT_HIP].x * width), int(landmarks[pose_landmark.LEFT_HIP].y * height)),
        "left_knee": (int(landmarks[pose_landmark.LEFT_KNEE].x * width), int(landmarks[pose_landmark.LEFT_KNEE].y * height)),
        "left_ankle": (int(landmarks[pose_landmark.LEFT_ANKLE].x * width), int(landmarks[pose_landmark.LEFT_ANKLE].y * height)),
        "left_shoulder": (int(landmarks[pose_landmark.LEFT_SHOULDER].x * width), int(landmarks[pose_landmark.LEFT_SHOULDER].y * height)),
        "left_elbow": (int(landmarks[pose_landmark.LEFT_ELBOW].x * width), int(landmarks[pose_landmark.LEFT_ELBOW].y * height)),
        "left_wrist": (int(landmarks[pose_landmark.LEFT_WRIST].x * width), int(landmarks[pose_landmark.LEFT_WRIST].y * height)),
        "right_hip": (int(landmarks[pose_landmark.RIGHT_HIP].x * width), int(landmarks[pose_landmark.RIGHT_HIP].y * height)),
        "right_knee": (int(landmarks[pose_landmark.RIGHT_KNEE].x * width), int(landmarks[pose_landmark.RIGHT_KNEE].y * height)),
        "right_ankle": (int(landmarks[pose_landmark.RIGHT_ANKLE].x * width), int(landmarks[pose_landmark.RIGHT_ANKLE].y * height)),
        "right_shoulder": (int(landmarks[pose_landmark.RIGHT_SHOULDER].x * width), int(landmarks[pose_landmark.RIGHT_SHOULDER].y * height)),
        "right_elbow": (int(landmarks[pose_landmark.RIGHT_ELBOW].x * width), int(landmarks[pose_landmark.RIGHT_ELBOW].y * height)),
        "right_wrist": (int(landmarks[pose_landmark.RIGHT_WRIST].x * width), int(landmarks[pose_landmark.RIGHT_WRIST].y * height)),
    }

    for point in points.values():
        draw_marker(frame, point, (0, 255, 0))

    line_color = (255, 0, 0)
    cv2.line(frame, points["left_shoulder"], points["left_elbow"], line_color, 2)
    cv2.line(frame, points["left_elbow"], points["left_wrist"], line_color, 2)
    cv2.line(frame, points["left_hip"], points["left_knee"], line_color, 2)
    cv2.line(frame, points["left_knee"], points["left_ankle"], line_color, 2)
    cv2.line(frame, points["right_shoulder"], points["right_elbow"], line_color, 2)
    cv2.line(frame, points["right_elbow"], points["right_wrist"], line_color, 2)
    cv2.line(frame, points["right_hip"], points["right_knee"], line_color, 2)
    cv2.line(frame, points["right_knee"], points["right_ankle"], line_color, 2)


def fit_frame_for_display(frame, max_width: int = 920, max_height: int = 640) -> np.ndarray:
    height, width = frame.shape[:2]
    scale = min(max_width / max(1, width), max_height / max(1, height), 1.0)
    if scale == 1.0:
        return frame
    new_size = (int(width * scale), int(height * scale))
    return cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)


def compose_preview(frame: np.ndarray, features: PoseFeatures, frame_index: int, timestamp_sec: float, sample_fps: float) -> np.ndarray:
    preview_frame = fit_frame_for_display(frame)
    height = preview_frame.shape[0]
    panel_width = 420
    panel = np.zeros((height, panel_width, 3), dtype=np.uint8)

    header_lines = [
        f"Frame {frame_index}",
        f"Time: {timestamp_sec:.2f}s",
        f"Target FPS: {sample_fps:.2f}",
        "",
        "Angles",
        f"L knee: {format_value(features.left_knee_angle)}",
        f"R knee: {format_value(features.right_knee_angle)}",
        f"Avg knee: {format_value(features.knee_angle)}",
        f"L elbow: {format_value(features.left_elbow_angle)}",
        f"R elbow: {format_value(features.right_elbow_angle)}",
        f"Avg elbow: {format_value(features.elbow_angle)}",
        f"Torso lean: {format_value(features.torso_lean)}",
        "",
        "Label keys",
        "1 = Low",
        "2 = Medium",
        "3 = High",
        "4 = Custom",
    ]

    y = 40
    for line in header_lines:
        if not line:
            y += 18
            continue
        cv2.putText(panel, line, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.64, (240, 240, 240), 2)
        y += 32

    return cv2.hconcat([preview_frame, panel])


def prompt_label(frame_index: int, timestamp_sec: float, features: PoseFeatures) -> str:
    print(
        f"\nFrame {frame_index} at {timestamp_sec:.2f}s | "
        f"knee={format_value(features.knee_angle)} "
        f"elbow={format_value(features.elbow_angle)} "
        f"torso={format_value(features.torso_lean)}"
    )
    print("Choose label: 1=Low, 2=Medium, 3=High, 4=Custom")

    while True:
        choice = input("Label: ").strip()
        label = normalize_label(choice)
        if label is None:
            print("Invalid choice. Use 1, 2, 3, or 4.")
            continue
        return label


def prepare_writer(output_csv: Path, append: bool) -> tuple[csv.DictWriter, object]:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    file_exists = output_csv.exists()
    mode = "a" if append and file_exists else "w"
    handle = output_csv.open(mode, newline="")
    writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)

    if mode == "w" or output_csv.stat().st_size == 0:
        writer.writeheader()

    return writer, handle


def build_row(
    video_path: Path,
    clip_start_sec: float,
    clip_end_sec: float,
    source_fps: float,
    sample_fps: float,
    frame_index: int,
    timestamp_sec: float,
    label: str,
    features: PoseFeatures,
) -> dict[str, object]:
    pose_detected = any(
        value is not None
        for value in (
            features.left_knee_angle,
            features.right_knee_angle,
            features.knee_angle,
            features.left_elbow_angle,
            features.right_elbow_angle,
            features.elbow_angle,
            features.torso_lean,
        )
    )

    return {
        "video_name": video_path.name,
        "clip_start_sec": round(clip_start_sec, 3),
        "clip_end_sec": round(clip_end_sec, 3),
        "source_fps": round(source_fps, 3),
        "sample_fps": round(sample_fps, 3),
        "frame_index": frame_index,
        "timestamp_sec": round(timestamp_sec, 3),
        "label": label,
        "left_knee_angle": None if features.left_knee_angle is None else round(features.left_knee_angle, 3),
        "right_knee_angle": None if features.right_knee_angle is None else round(features.right_knee_angle, 3),
        "knee_angle": None if features.knee_angle is None else round(features.knee_angle, 3),
        "left_elbow_angle": None if features.left_elbow_angle is None else round(features.left_elbow_angle, 3),
        "right_elbow_angle": None if features.right_elbow_angle is None else round(features.right_elbow_angle, 3),
        "elbow_angle": None if features.elbow_angle is None else round(features.elbow_angle, 3),
        "torso_lean": None if features.torso_lean is None else round(features.torso_lean, 3),
        "pose_detected": int(pose_detected),
    }


def main() -> None:
    args = parse_args()
    video_path = Path(args.video).expanduser().resolve()
    output_csv = Path(args.output_csv).expanduser().resolve()

    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    start_time = args.start_time if args.start_time is not None else prompt_float("Start time in seconds: ", 0.0)
    end_time = args.end_time if args.end_time is not None else prompt_float("Stop time in seconds: ", 0.0)
    sample_fps = args.sample_fps if args.sample_fps is not None else prompt_float("Frames per second to label: ", 0.1)

    if end_time <= start_time:
        raise ValueError("Stop time must be greater than start time.")
    if sample_fps <= 0:
        raise ValueError("Frames per second must be greater than zero.")

    ensure_model_file()

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

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    source_fps = cap.get(cv2.CAP_PROP_FPS)
    if source_fps <= 0:
        source_fps = 30.0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / source_fps if total_frames > 0 else end_time
    clip_start_sec = max(0.0, min(start_time, duration_sec))
    clip_end_sec = max(clip_start_sec, min(end_time, duration_sec))
    max_frame_index = max(0, total_frames - 1)
    sample_interval = 1.0 / sample_fps
    sample_time = clip_start_sec
    rows: list[dict[str, object]] = []

    writer, output_handle = prepare_writer(output_csv, append=args.append)

    try:
        with PoseLandmarker.create_from_options(options) as landmarker:
            while sample_time <= clip_end_sec + 1e-9:
                frame_index = min(max_frame_index, int(round(sample_time * source_fps)))
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                success, frame = cap.read()
                if not success:
                    break

                timestamp_sec = frame_index / source_fps
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                timestamp_ms = int(timestamp_sec * 1000)
                result = landmarker.detect_for_video(mp_image, timestamp_ms)

                if result.pose_landmarks:
                    landmarks = result.pose_landmarks[0]
                    features = extract_pose_features(landmarks, PoseLandmark)
                    draw_pose_skeleton(frame, landmarks, PoseLandmark)
                else:
                    features = PoseFeatures(
                        left_knee_angle=None,
                        right_knee_angle=None,
                        knee_angle=None,
                        left_elbow_angle=None,
                        right_elbow_angle=None,
                        elbow_angle=None,
                        torso_lean=None,
                    )

                preview = compose_preview(frame, features, frame_index, timestamp_sec, sample_fps)
                cv2.imshow("Dataset Labeling Preview", preview)
                cv2.waitKey(1)

                label = prompt_label(frame_index, timestamp_sec, features)
                if label == "__quit__":
                    print("Stopped by user.")
                    break

                row = build_row(
                    video_path=video_path,
                    clip_start_sec=clip_start_sec,
                    clip_end_sec=clip_end_sec,
                    source_fps=source_fps,
                    sample_fps=sample_fps,
                    frame_index=frame_index,
                    timestamp_sec=timestamp_sec,
                    label=label,
                    features=features,
                )
                writer.writerow(row)
                rows.append(row)
                print(f"Saved frame {frame_index} with label '{label}'.")

                sample_time += sample_interval
    finally:
        cap.release()
        output_handle.close()
        cv2.destroyAllWindows()

    print(f"Saved {len(rows)} labeled frame(s) to {output_csv}")


if __name__ == "__main__":
    main()
