#imports
import mediapipe as mp
from mediapipe.python.solutions import hands as mp_hands
import cv2

# Implementation of a class that gets the x and y coordinates of the index finger using the MediaPipe library.
class IndexFingerPosition:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def get_x(self):
        return self.x

    def get_y(self):
        return self.y

    def __str__(self):
        return f"IndexFingerPosition(x={self.x}, y={self.y})"

    # Function that gets the x and y coordinates of the index finger tip
    def get_index_finger_position(self, image):
        # Get the height and the width of the image
        h, w, _ = image.shape

        # Process the image using the hands model of the MediaPipe library
        results = self.hands.process(image)

        # If the model detects a hand in the image get the x and y coordinates of the index finger tip
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

            tip_ix, tip_iy = int(index_tip.x * w), int(index_tip.y * h)

            return (tip_ix, tip_iy)

