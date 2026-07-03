import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import urllib.request

import cv2
import mediapipe as mp
import numpy as np

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)
MODEL_PATH = Path(__file__).with_name("pose_landmarker_lite.task")


@dataclass
class FrameMetrics:
    frame_index: int
    timestamp_sec: float
    knee_angle: float | None
    elbow_angle: float | None
    torso_lean: float | None
    movement_score: float


@dataclass
class Segment:
    start_frame: int
    end_frame: int


def ensure_model_file() -> None:
    if MODEL_PATH.exists():
        return

    print(f"Downloading pose model to {MODEL_PATH}...")
    with urllib.request.urlopen(MODEL_URL) as response, MODEL_PATH.open("wb") as model_file:
        model_file.write(response.read())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a labeled badminton movement dataset from a single video"
    )
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument(
        "--output-csv",
        default="movement_dataset.csv",
        help="Path to output CSV",
    )
    parser.add_argument(
        "--manual-select",
        action="store_true",
        help="Skip auto detection and provide movement ranges manually",
    )
    parser.add_argument(
        "--playback-speed",
        type=float,
        default=1.0,
        help="Playback speed for segment preview (1.0 is real time)",
    )
    return parser.parse_args()


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


def get_knee_angle(landmarks: list, pose_landmark) -> float | None:
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
    return average_angles([left, right])


def get_elbow_angle(landmarks: list, pose_landmark) -> float | None:
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
    return average_angles([left, right])


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

    # Lean is angle from vertical axis in image coordinates.
    lean = np.degrees(np.arctan2(abs(torso_vector[0]), abs(torso_vector[1]) + 1e-6))
    return float(lean)


def extract_motion_points(landmarks: list, pose_landmark) -> np.ndarray:
    indices = [
        pose_landmark.LEFT_SHOULDER,
        pose_landmark.RIGHT_SHOULDER,
        pose_landmark.LEFT_ELBOW,
        pose_landmark.RIGHT_ELBOW,
        pose_landmark.LEFT_WRIST,
        pose_landmark.RIGHT_WRIST,
        pose_landmark.LEFT_HIP,
        pose_landmark.RIGHT_HIP,
        pose_landmark.LEFT_KNEE,
        pose_landmark.RIGHT_KNEE,
        pose_landmark.LEFT_ANKLE,
        pose_landmark.RIGHT_ANKLE,
    ]

    points = [get_point(landmarks, idx) for idx in indices]
    return np.array(points, dtype=np.float32)


def process_video(video_path: Path) -> tuple[list[FrameMetrics], float, int]:
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

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_metrics: list[FrameMetrics] = []
    prev_motion_points: np.ndarray | None = None

    with PoseLandmarker.create_from_options(options) as landmarker:
        frame_index = 0
        while True:
            success, frame = cap.read()
            if not success:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            timestamp_ms = int((frame_index / fps) * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            knee_angle = None
            elbow_angle = None
            torso_lean = None
            movement_score = 0.0

            if result.pose_landmarks:
                landmarks = result.pose_landmarks[0]
                knee_angle = get_knee_angle(landmarks, PoseLandmark)
                elbow_angle = get_elbow_angle(landmarks, PoseLandmark)
                torso_lean = get_torso_lean(landmarks, PoseLandmark)

                current_motion_points = extract_motion_points(landmarks, PoseLandmark)
                if prev_motion_points is not None:
                    delta = np.linalg.norm(current_motion_points - prev_motion_points, axis=1)
                    movement_score = float(np.mean(delta))
                prev_motion_points = current_motion_points
            else:
                prev_motion_points = None

            frame_metrics.append(
                FrameMetrics(
                    frame_index=frame_index,
                    timestamp_sec=frame_index / fps,
                    knee_angle=knee_angle,
                    elbow_angle=elbow_angle,
                    torso_lean=torso_lean,
                    movement_score=movement_score,
                )
            )
            frame_index += 1

    cap.release()
    return frame_metrics, fps, total_frames


def detect_segments_auto(frame_metrics: list[FrameMetrics], fps: float) -> list[Segment]:
    if not frame_metrics:
        return []

    movement = np.array([metric.movement_score for metric in frame_metrics], dtype=np.float32)
    window = 5
    smooth_kernel = np.ones(window, dtype=np.float32) / window
    smoothed = np.convolve(movement, smooth_kernel, mode="same")

    threshold = max(0.008, float(smoothed.mean() + 0.8 * smoothed.std()))
    active = smoothed > threshold

    min_length = max(8, int(0.25 * fps))
    pad = max(2, int(0.15 * fps))

    segments: list[Segment] = []
    run_start = None

    for idx, is_active in enumerate(active):
        if is_active and run_start is None:
            run_start = idx
        elif not is_active and run_start is not None:
            run_end = idx - 1
            if run_end - run_start + 1 >= min_length:
                start = max(0, run_start - pad)
                end = min(len(frame_metrics) - 1, run_end + pad)
                segments.append(Segment(start_frame=start, end_frame=end))
            run_start = None

    if run_start is not None:
        run_end = len(active) - 1
        if run_end - run_start + 1 >= min_length:
            start = max(0, run_start - pad)
            end = min(len(frame_metrics) - 1, run_end + pad)
            segments.append(Segment(start_frame=start, end_frame=end))

    merged: list[Segment] = []
    for segment in segments:
        if not merged:
            merged.append(segment)
            continue

        last = merged[-1]
        if segment.start_frame <= last.end_frame + 1:
            last.end_frame = max(last.end_frame, segment.end_frame)
        else:
            merged.append(segment)

    return merged


def parse_manual_segments(total_frames: int) -> list[Segment]:
    print("Manual segment selection enabled.")
    print("Enter ranges like: 120-200,260-320,400-450")
    print(f"Frame range available: 0 to {max(0, total_frames - 1)}")

    while True:
        raw = input("Movement segments: ").strip()
        if not raw:
            print("Please enter at least one range.")
            continue

        try:
            segments: list[Segment] = []
            chunks = [chunk.strip() for chunk in raw.split(",") if chunk.strip()]
            for chunk in chunks:
                start_text, end_text = chunk.split("-", maxsplit=1)
                start = int(start_text)
                end = int(end_text)
                if start < 0 or end < 0 or start >= total_frames or end >= total_frames:
                    raise ValueError("Range out of bounds")
                if start >= end:
                    raise ValueError("Start must be smaller than end")
                segments.append(Segment(start_frame=start, end_frame=end))
            return segments
        except ValueError as exc:
            print(f"Invalid format: {exc}")


def play_segment(
    cap: cv2.VideoCapture,
    segment: Segment,
    frame_metrics: list[FrameMetrics],
    fps: float,
    playback_speed: float,
) -> bool:
    cap.set(cv2.CAP_PROP_POS_FRAMES, segment.start_frame)

    delay_ms = max(1, int(1000 / max(1e-6, fps * playback_speed)))
    frame_index = segment.start_frame

    while frame_index <= segment.end_frame:
        success, frame = cap.read()
        if not success:
            break

        metric = frame_metrics[frame_index]
        knee_text = "N/A" if metric.knee_angle is None else f"{metric.knee_angle:.1f}"
        elbow_text = "N/A" if metric.elbow_angle is None else f"{metric.elbow_angle:.1f}"
        torso_text = "N/A" if metric.torso_lean is None else f"{metric.torso_lean:.1f}"

        cv2.putText(frame, f"Frame: {frame_index}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        cv2.putText(frame, f"Knee angle: {knee_text}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Elbow angle: {elbow_text}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Torso lean: {torso_text}", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Motion score: {metric.movement_score:.4f}", (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 215, 255), 2)
        cv2.putText(frame, "Press q to stop labeling", (20, 195), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

        cv2.imshow("Dataset Builder Preview", frame)
        key = cv2.waitKey(delay_ms) & 0xFF
        if key == ord("q"):
            return True

        frame_index += 1

    return False


def summarize_segment(segment: Segment, frame_metrics: list[FrameMetrics]) -> dict[str, float | None]:
    segment_rows = frame_metrics[segment.start_frame : segment.end_frame + 1]

    def avg(values: list[float | None]) -> float | None:
        valid = [value for value in values if value is not None]
        if not valid:
            return None
        return round(float(sum(valid) / len(valid)), 3)

    return {
        "avg_knee": avg([row.knee_angle for row in segment_rows]),
        "avg_elbow": avg([row.elbow_angle for row in segment_rows]),
        "avg_torso_lean": avg([row.torso_lean for row in segment_rows]),
        "avg_motion": round(float(sum(row.movement_score for row in segment_rows) / len(segment_rows)), 6),
    }


def prompt_label(segment_id: int, segment: Segment, summary: dict[str, float | None]) -> tuple[str | None, Segment | None, bool]:
    print(
        f"\nSegment {segment_id}: frames {segment.start_frame}-{segment.end_frame} | "
        f"knee={summary['avg_knee']} elbow={summary['avg_elbow']} "
        f"torso_lean={summary['avg_torso_lean']} motion={summary['avg_motion']}"
    )
    print("Label options: 1=low, 2=medium, 3=high, c=custom text, e=edit range, s=skip, q=quit")

    while True:
        choice = input("Your choice: ").strip().lower()
        if choice == "1":
            return "low", segment, False
        if choice == "2":
            return "medium", segment, False
        if choice == "3":
            return "high", segment, False
        if choice == "c":
            custom = input("Enter custom label: ").strip()
            if custom:
                return custom, segment, False
            print("Custom label cannot be empty.")
            continue
        if choice == "e":
            edited = input("Enter new range start-end: ").strip()
            try:
                start_text, end_text = edited.split("-", maxsplit=1)
                start = int(start_text)
                end = int(end_text)
                if start >= end:
                    print("Start must be smaller than end.")
                    continue
                return None, Segment(start_frame=start, end_frame=end), False
            except ValueError:
                print("Invalid range format. Use start-end.")
                continue
        if choice == "s":
            return None, segment, False
        if choice == "q":
            return None, segment, True
        print("Invalid input.")


def collect_labeled_rows(
    video_path: Path,
    frame_metrics: list[FrameMetrics],
    segments: list[Segment],
    fps: float,
    playback_speed: float,
) -> list[dict[str, object]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video for preview: {video_path}")

    rows: list[dict[str, object]] = []

    try:
        for idx, original_segment in enumerate(segments, start=1):
            segment = Segment(
                start_frame=max(0, original_segment.start_frame),
                end_frame=min(len(frame_metrics) - 1, original_segment.end_frame),
            )

            while True:
                should_quit = play_segment(cap, segment, frame_metrics, fps, playback_speed)
                if should_quit:
                    print("Stopped by user.")
                    return rows

                summary = summarize_segment(segment, frame_metrics)
                label, maybe_segment, quit_requested = prompt_label(idx, segment, summary)

                if quit_requested:
                    return rows

                if maybe_segment is not None and maybe_segment != segment and label is None:
                    segment = Segment(
                        start_frame=max(0, maybe_segment.start_frame),
                        end_frame=min(len(frame_metrics) - 1, maybe_segment.end_frame),
                    )
                    if segment.start_frame >= segment.end_frame:
                        print("Edited range is invalid after bounds check.")
                        segment = Segment(
                            start_frame=max(0, original_segment.start_frame),
                            end_frame=min(len(frame_metrics) - 1, original_segment.end_frame),
                        )
                    continue

                if label is None:
                    print("Segment skipped.")
                    break

                for frame_idx in range(segment.start_frame, segment.end_frame + 1):
                    metric = frame_metrics[frame_idx]
                    rows.append(
                        {
                            "video_name": video_path.name,
                            "segment_id": idx,
                            "segment_start_frame": segment.start_frame,
                            "segment_end_frame": segment.end_frame,
                            "frame_index": metric.frame_index,
                            "timestamp_sec": round(metric.timestamp_sec, 4),
                            "knee_angle": None if metric.knee_angle is None else round(metric.knee_angle, 3),
                            "elbow_angle": None if metric.elbow_angle is None else round(metric.elbow_angle, 3),
                            "torso_lean": None if metric.torso_lean is None else round(metric.torso_lean, 3),
                            "movement_score": round(metric.movement_score, 6),
                            "label": label,
                        }
                    )
                print(f"Saved label '{label}' for segment {idx}.")
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return rows


def write_rows(rows: list[dict[str, object]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "video_name",
        "segment_id",
        "segment_start_frame",
        "segment_end_frame",
        "frame_index",
        "timestamp_sec",
        "knee_angle",
        "elbow_angle",
        "torso_lean",
        "movement_score",
        "label",
    ]

    file_exists = output_csv.exists()

    with output_csv.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not file_exists or output_csv.stat().st_size == 0:
            writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    video_path = Path(args.video).expanduser().resolve()
    output_csv = Path(args.output_csv).expanduser().resolve()

    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    ensure_model_file()

    print("Processing video and extracting measurements...")
    frame_metrics, fps, total_frames = process_video(video_path)
    print(f"Processed {len(frame_metrics)} frames at {fps:.2f} FPS.")

    if args.manual_select:
        segments = parse_manual_segments(total_frames)
    else:
        segments = detect_segments_auto(frame_metrics, fps)
        print(f"Auto-detected {len(segments)} movement segment(s).")
        if not segments:
            print("No segments found automatically. Falling back to manual input.")
            segments = parse_manual_segments(total_frames)

    rows = collect_labeled_rows(
        video_path=video_path,
        frame_metrics=frame_metrics,
        segments=segments,
        fps=fps,
        playback_speed=max(0.1, args.playback_speed),
    )

    if not rows:
        print("No labeled rows to save.")
        return

    write_rows(rows, output_csv)
    print(f"Saved {len(rows)} labeled rows to {output_csv}")


if __name__ == "__main__":
    main()
