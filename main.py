import argparse
import csv
import os
import time
from pathlib import Path
import urllib.request

import cv2
import mediapipe as mp
import numpy as np

from pose_analysis import InjuryRiskPredictor, PoseSample

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)
MODEL_PATH = Path(__file__).with_name("pose_landmarker_lite.task")
CSV_PATH = Path(__file__).with_name("pose_data.csv")


def ensure_model_file():
    if MODEL_PATH.exists():
        return

    print(f"Downloading pose model to {MODEL_PATH}...")
    with urllib.request.urlopen(MODEL_URL) as response, MODEL_PATH.open("wb") as model_file:
        model_file.write(response.read())


def draw_marker(frame, point, color, radius=6):
    cv2.circle(frame, point, radius, color, -1)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Real-time posture detection and injury risk data capture"
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=int(os.environ.get("CAMERA_INDEX", "0")),
        help="Camera index to open",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=int(os.environ.get("FRAME_WIDTH", "640")),
        help="Requested capture width",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=int(os.environ.get("FRAME_HEIGHT", "480")),
        help="Requested capture height",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=int(os.environ.get("MAX_FRAMES", "0")),
        help="Stop after this many frames. Use 0 for unlimited when preview is enabled.",
    )
    parser.add_argument(
        "--preview",
        action=argparse.BooleanOptionalAction,
        default=os.environ.get("SHOW_PREVIEW", "1") != "0",
        help="Show or hide the preview window while running",
    )
    parser.add_argument(
        "--csv-path",
        default=str(CSV_PATH),
        help="Path to the CSV file used to append pose angles",
    )
    return parser.parse_args()


def prepare_csv_file(csv_path):
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    file_exists = csv_path.exists()
    file = csv_path.open("a", newline="")
    writer = csv.writer(file)

    if not file_exists or csv_path.stat().st_size == 0:
        writer.writerow(["knee_angle", "shoulder_angle", "risk_score", "risk_level", "feedback"])

    return file, writer


ensure_model_file()

def calculate_angle(a, b, c):

    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(
        c[1] - b[1],
        c[0] - b[0]
    ) - np.arctan2(
        a[1] - b[1],
        a[0] - b[0]
    )

    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180:
        angle = 360 - angle

    return angle

def main():
    args = parse_args()
    show_preview = args.preview
    headless_mode = not show_preview
    max_frames = args.max_frames if args.max_frames > 0 else (30 if headless_mode else 0)

    if headless_mode:
        print("Running in headless mode: use --no-preview for terminal-only runs.")
    else:
        print("Preview mode enabled: the webcam window should open now.")

    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=VisionRunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(args.camera)

    if not cap.isOpened():
        raise RuntimeError(f"Unable to open webcam {args.camera}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    file, writer = prepare_csv_file(args.csv_path)
    predictor = InjuryRiskPredictor()

    try:
        with PoseLandmarker.create_from_options(options) as landmarker:
            frame_index = 0
            started_at = time.monotonic()

            while True:
                success, frame = cap.read()

                if not success:
                    break

                frame_index += 1

                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                timestamp_ms = int((time.monotonic() - started_at) * 1000)

                results = landmarker.detect_for_video(mp_image, timestamp_ms)

                if results.pose_landmarks:
                    landmarks = results.pose_landmarks[0]

                    left_hip = landmarks[mp.tasks.vision.PoseLandmark.LEFT_HIP]
                    left_knee = landmarks[mp.tasks.vision.PoseLandmark.LEFT_KNEE]
                    left_ankle = landmarks[mp.tasks.vision.PoseLandmark.LEFT_ANKLE]
                    left_shoulder = landmarks[mp.tasks.vision.PoseLandmark.LEFT_SHOULDER]
                    left_elbow = landmarks[mp.tasks.vision.PoseLandmark.LEFT_ELBOW]
                    left_wrist = landmarks[mp.tasks.vision.PoseLandmark.LEFT_WRIST]

                    hip = [left_hip.x, left_hip.y]
                    knee = [left_knee.x, left_knee.y]
                    ankle = [left_ankle.x, left_ankle.y]
                    shoulder = [left_shoulder.x, left_shoulder.y]
                    elbow = [left_elbow.x, left_elbow.y]
                    wrist = [left_wrist.x, left_wrist.y]

                    shoulder_angle = calculate_angle(shoulder, elbow, wrist)
                    knee_angle = calculate_angle(hip, knee, ankle)

                    assessment = predictor.assess(
                        PoseSample(knee_angle=knee_angle, elbow_angle=shoulder_angle)
                    )

                    height, width = frame.shape[:2]
                    points = {
                        "hip": (int(left_hip.x * width), int(left_hip.y * height)),
                        "knee": (int(left_knee.x * width), int(left_knee.y * height)),
                        "ankle": (int(left_ankle.x * width), int(left_ankle.y * height)),
                        "shoulder": (int(left_shoulder.x * width), int(left_shoulder.y * height)),
                        "elbow": (int(left_elbow.x * width), int(left_elbow.y * height)),
                        "wrist": (int(left_wrist.x * width), int(left_wrist.y * height)),
                    }

                    for point in points.values():
                        draw_marker(frame, point, (0, 255, 0))

                    cv2.line(frame, points["shoulder"], points["elbow"], (255, 0, 0), 2)
                    cv2.line(frame, points["elbow"], points["wrist"], (255, 0, 0), 2)
                    cv2.line(frame, points["hip"], points["knee"], (255, 0, 0), 2)
                    cv2.line(frame, points["knee"], points["ankle"], (255, 0, 0), 2)

                    cv2.putText(
                        frame,
                        f"Knee Angle: {int(knee_angle)}",
                        (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2,
                    )

                    cv2.putText(
                        frame,
                        f"Shoulder Angle: {int(shoulder_angle)}",
                        (20, 100),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2,
                    )

                    risk_color = (0, 255, 0)
                    if assessment.level == "MEDIUM":
                        risk_color = (0, 215, 255)
                    elif assessment.level == "HIGH":
                        risk_color = (0, 0, 255)

                    cv2.putText(
                        frame,
                        f"Risk: {assessment.level} ({assessment.score})",
                        (20, 150),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        risk_color,
                        2,
                    )

                    feedback_lines = assessment.feedback.split(". ")
                    y_offset = 185
                    for line in feedback_lines[:2]:
                        if line:
                            cv2.putText(
                                frame,
                                line[:52],
                                (20, y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                risk_color,
                                2,
                            )
                            y_offset += 24

                    writer.writerow([
                        round(knee_angle, 2),
                        round(shoulder_angle, 2),
                        assessment.score,
                        assessment.level,
                        assessment.feedback,
                    ])

                    print(
                        f"Knee={knee_angle:.2f} Shoulder={shoulder_angle:.2f} "
                        f"Risk={assessment.level}({assessment.score})"
                    )
                    print(assessment.feedback)

                if show_preview:
                    elapsed = max(time.monotonic() - started_at, 1e-6)
                    fps = frame_index / elapsed
                    cv2.putText(
                        frame,
                        f"FPS: {fps:.1f}",
                        (20, 240),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 255),
                        2,
                    )
                    cv2.imshow("Badminton Pose Detection", frame)

                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                if max_frames and frame_index >= max_frames:
                    break
    finally:
        cap.release()
        file.close()
        if show_preview:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()