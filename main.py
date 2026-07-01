import csv
import os
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


def ensure_model_file():
    if MODEL_PATH.exists():
        return

    print(f"Downloading pose model to {MODEL_PATH}...")
    with urllib.request.urlopen(MODEL_URL) as response, MODEL_PATH.open("wb") as model_file:
        model_file.write(response.read())


def draw_marker(frame, point, color, radius=6):
    cv2.circle(frame, point, radius, color, -1)


ensure_model_file()

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
    running_mode=VisionRunningMode.IMAGE,
)

# ==============================
# Open Webcam
# ==============================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Unable to open webcam 0")

show_preview = os.environ.get("SHOW_PREVIEW") == "1"
headless_mode = not show_preview
max_frames = 30 if headless_mode else None
frame_count = 0

if headless_mode:
    print("Running in headless mode: set SHOW_PREVIEW=1 to open the preview window.")

# ==============================
# Create CSV File
# ==============================

file = open("pose_data.csv", mode="a", newline="")
writer = csv.writer(file)

# Uncomment this ONLY once if you want column names
# writer.writerow(["left_knee_angle"])

# ==============================
# Angle Calculation Function
# ==============================

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

# ==============================
# Main Loop
# ==============================

with PoseLandmarker.create_from_options(options) as landmarker:
    while True:
        frame_count += 1

        success, frame = cap.read()

        if not success:
            break

        # Mirror effect
        frame = cv2.flip(frame, 1)

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Process pose detection
        results = landmarker.detect(mp_image)

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

            writer.writerow([knee_angle, shoulder_angle])

            print(f"Left Knee Angle: {knee_angle:.2f}")
            print(f"Left Shoulder Angle: {shoulder_angle:.2f}")

        if not headless_mode:
            cv2.imshow("Badminton Pose Detection", frame)

            # Press Q to Quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        if max_frames is not None and frame_count >= max_frames:
            break
# ================================
# Cleanup 
# ================================

cap.release()
file.close()
if not headless_mode:
    cv2.destroyAllWindows()