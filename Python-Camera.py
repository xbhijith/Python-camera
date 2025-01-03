import cv2
import mediapipe as mp
import numpy as np
import math

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = pose.process(frame_rgb)

    black_canvas = np.zeros_like(frame)

    if result.pose_landmarks:
        landmarks = result.pose_landmarks.landmark

        nose = landmarks[mp_pose.PoseLandmark.NOSE]
        left_eye = landmarks[mp_pose.PoseLandmark.LEFT_EYE]
        right_eye = landmarks[mp_pose.PoseLandmark.RIGHT_EYE]

        height, width, _ = frame.shape
        nose_x_pixel = int(nose.x * width)
        nose_y_pixel = int(nose.y * height)

        left_eye_x = int(left_eye.x * width)
        left_eye_y = int(left_eye.y * height)
        right_eye_x = int(right_eye.x * width)
        right_eye_y = int(right_eye.y * height)
        side_length = int(math.sqrt((left_eye_x - right_eye_x) ** 2 + (left_eye_y - right_eye_y) ** 2)) * 2

        top_left = (nose_x_pixel - side_length // 2, nose_y_pixel - side_length // 2)
        bottom_right = (nose_x_pixel + side_length // 2, nose_y_pixel + side_length // 2)

        cv2.rectangle(black_canvas, top_left, bottom_right, (255, 255, 255), 2)

        mp_drawing.draw_landmarks(black_canvas, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    cv2.imshow('MediaPipe Pose', frame)
    cv2.imshow('Black Canvas Pose', black_canvas)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
