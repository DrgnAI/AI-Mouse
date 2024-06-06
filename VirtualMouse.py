import cv2
import numpy as np
import mediapipe as mp
import autopy

from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('action.h5')

# MediaPipe Holistic model setup
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Function to process and predict the gesture
def classify_action(image, model, holistic):
    image, results = mediapipe_detection(image, holistic)
    keypoints = extract_keypoints(results)
    keypoints = np.expand_dims(keypoints, axis=0)
    action = model.predict(keypoints)
    return action, results

# MediaPipe detection utility
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

# Keypoints extraction from different landmarks
def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*3)
    return pose

# Webcam input
cap = cv2.VideoCapture(0)

# Setup MediaPipe instance
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        action, results = classify_action(frame, model, holistic)
        action_type = np.argmax(action[0])
        
        # Assume index 0 for click and 1 for not click, adjust based on your training
        if action_type == 0:
            autopy.mouse.click()

        if results.pose_landmarks:
            # Assuming the index of the pointer finger tip, change based on your keypoints setup
            pointer_finger_tip = results.pose_landmarks.landmark[8]  # Index of pointer finger tip in pose landmarks
            screen_w, screen_h = autopy.screen.size()
            x, y = pointer_finger_tip.x * screen_w, pointer_finger_tip.y * screen_h
            autopy.mouse.move(x, y)

        cv2.imshow('AI Mouse Controller', frame)
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
