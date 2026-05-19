import cv2
import mediapipe as mp
import numpy as np
import csv

# ==============================
# Initialize MediaPipe
# ==============================

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

mp_draw = mp.solutions.drawing_utils

# ==============================
# Open Webcam
# ==============================

cap = cv2.VideoCapture(0)

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

while True:

    success, frame = cap.read()

    if not success:
        break

    # Mirror effect
    frame = cv2.flip(frame, 1)

    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process Pose Detection
    results = pose.process(rgb_frame)

    # ==============================
    # If Pose Detected
    # ==============================

    if results.pose_landmarks:

        # Draw Skeleton
        mp_draw.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

        landmarks = results.pose_landmarks.landmark

        # ==============================
        # Get Coordinates
        # ==============================

        # LEFT HIP
        hip = [
            landmarks[23].x,
            landmarks[23].y
        ]

        # LEFT KNEE
        knee = [
            landmarks[25].x,
            landmarks[25].y
        ]

        # LEFT ANKLE
        ankle = [
            landmarks[27].x,
            landmarks[27].y
        ]

        #left shoulder
        shoulder = [
            landmarks[11].x,
            landmarks[11].y
        ]

        #left elbow
        elbow = [
            landmarks[13].x,
            landmarks[13].y
        ]

        #left wrist
        wrist = [
            landmarks[15].x,
            landmarks[15].y
        ]
        # ==============================
        # Calculate Knee Angle
        # ==============================
        shoulder_angle = calculate_angle(
            shoulder,
            elbow,
            wrist
        )
        knee_angle = calculate_angle(
            hip,
            knee,
            ankle
        )

        # ==============================
        # Display Angle
        # ==============================

        cv2.putText(
            frame,
            f"Knee Angle: {int(knee_angle)}",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Shoulder Angle: {int(shoulder_angle)}",
            (20, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        # ==============================
        # Save Data to CSV
        # ==============================

        writer.writerow([knee_angle, shoulder_angle])

        # ==============================
        # Print Data in Terminal
        # ==============================

        print(f"Left Knee Angle: {knee_angle:.2f}")
        print(f"Left Shoulder Angle: {shoulder_angle:.2f}")

    # ==============================
    # Show Webcam
    # ==============================

    cv2.imshow(
        "Badminton Pose Detection",
        frame
    )

    # Press Q to Quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# ================================
# Cleanup 
# ================================

cap.release()
file.close()
cv2.destroyAllWindows()