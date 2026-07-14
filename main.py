import argparse
import os
import time

import cv2
import mediapipe as mp

from pose_analysis import MODEL_PATH, ensure_model_file, extract_pose_features


def draw_marker(frame, point, color, radius=6):
    cv2.circle(frame, point, radius, color, -1)


def parse_args():
    parser = argparse.ArgumentParser(description="Real-time pose landmark preview")
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
    return parser.parse_args()


def format_value(value):
    return "N/A" if value is None else f"{value:.1f}"


def draw_pose(frame, landmarks, pose_landmark):
    height, width = frame.shape[:2]
    points = {
        "hip": (int(landmarks[pose_landmark.LEFT_HIP].x * width), int(landmarks[pose_landmark.LEFT_HIP].y * height)),
        "knee": (int(landmarks[pose_landmark.LEFT_KNEE].x * width), int(landmarks[pose_landmark.LEFT_KNEE].y * height)),
        "ankle": (int(landmarks[pose_landmark.LEFT_ANKLE].x * width), int(landmarks[pose_landmark.LEFT_ANKLE].y * height)),
        "shoulder": (int(landmarks[pose_landmark.LEFT_SHOULDER].x * width), int(landmarks[pose_landmark.LEFT_SHOULDER].y * height)),
        "elbow": (int(landmarks[pose_landmark.LEFT_ELBOW].x * width), int(landmarks[pose_landmark.LEFT_ELBOW].y * height)),
        "wrist": (int(landmarks[pose_landmark.LEFT_WRIST].x * width), int(landmarks[pose_landmark.LEFT_WRIST].y * height)),
        "right_hip": (int(landmarks[pose_landmark.RIGHT_HIP].x * width), int(landmarks[pose_landmark.RIGHT_HIP].y * height)),
        "right_knee": (int(landmarks[pose_landmark.RIGHT_KNEE].x * width), int(landmarks[pose_landmark.RIGHT_KNEE].y * height)),
        "right_ankle": (int(landmarks[pose_landmark.RIGHT_ANKLE].x * width), int(landmarks[pose_landmark.RIGHT_ANKLE].y * height)),
        "right_shoulder": (int(landmarks[pose_landmark.RIGHT_SHOULDER].x * width), int(landmarks[pose_landmark.RIGHT_SHOULDER].y * height)),
        "right_elbow": (int(landmarks[pose_landmark.RIGHT_ELBOW].x * width), int(landmarks[pose_landmark.RIGHT_ELBOW].y * height)),
        "right_wrist": (int(landmarks[pose_landmark.RIGHT_WRIST].x * width), int(landmarks[pose_landmark.RIGHT_WRIST].y * height)),
    }

    for point in points.values():
        draw_marker(frame, point, (0, 255, 0))

    cv2.line(frame, points["shoulder"], points["elbow"], (255, 0, 0), 2)
    cv2.line(frame, points["elbow"], points["wrist"], (255, 0, 0), 2)
    cv2.line(frame, points["hip"], points["knee"], (255, 0, 0), 2)
    cv2.line(frame, points["knee"], points["ankle"], (255, 0, 0), 2)
    cv2.line(frame, points["right_shoulder"], points["right_elbow"], (255, 0, 0), 2)
    cv2.line(frame, points["right_elbow"], points["right_wrist"], (255, 0, 0), 2)
    cv2.line(frame, points["right_hip"], points["right_knee"], (255, 0, 0), 2)
    cv2.line(frame, points["right_knee"], points["right_ankle"], (255, 0, 0), 2)


def annotate_frame(frame, features):
    if features is None:
        cv2.putText(frame, "No pose detected", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        return

    lines = [
        f"Left knee: {format_value(features.left_knee_angle)}",
        f"Right knee: {format_value(features.right_knee_angle)}",
        f"Avg knee: {format_value(features.knee_angle)}",
        f"Left elbow: {format_value(features.left_elbow_angle)}",
        f"Right elbow: {format_value(features.right_elbow_angle)}",
        f"Avg elbow: {format_value(features.elbow_angle)}",
        f"Torso lean: {format_value(features.torso_lean)}",
    ]

    y = 50
    for line in lines:
        cv2.putText(frame, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y += 34

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
                    features = extract_pose_features(landmarks, mp.tasks.vision.PoseLandmark)
                    draw_pose(frame, landmarks, mp.tasks.vision.PoseLandmark)
                    annotate_frame(frame, features)
                else:
                    annotate_frame(frame, None)

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
        if show_preview:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()