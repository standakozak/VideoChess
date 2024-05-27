import mediapipe as mp
import cv2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision.core import vision_task_running_mode as running_mode_module

# This file contains the HandLandmarker class which is used to get the landmarks of a hand in an image using the MediaPipe library.
# For the recogniton of the landmarks the hand_landmarker.task file is used.
class OwnHandLandmarker:
    def __init__(self, model_path):
        base_options = python.BaseOptions(model_asset_path=model_path)
        #hand_landmarker_options = vision.HandLandmarkerOptions(base_options=base_options, running_mode = running_mode_module.VisionTaskRunningMode.VIDEO)
        hand_landmarker_options = vision.HandLandmarkerOptions(base_options=base_options)
        self.hand_landmarker = vision.HandLandmarker.create_from_options(hand_landmarker_options)

    def get_landmarks(self, image):
        return self.hand_landmarker.detect(input_image=image)
    

# Example usage:
hand_landmarker = OwnHandLandmarker(model_path="Viktor/hand_landmarker.task")

# Start capturing video from the webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue

    # Convert the image from BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image using the MediaPipe representation of an image
    image = mp.Image(image_format = mp.ImageFormat.SRGB, data = image)

    # Process the image
    results = hand_landmarker.get_landmarks(image)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get the index finger tip landmark
            index_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
            h, w, c = image.shape
            ix, iy = int(index_tip.x * w), int(index_tip.y * h)

            # Print or use the pixel coordinates
            print("Index Finger Tip:", ix, iy)

    # Display the annotated image
    #cv2.imshow('Hand Tracking', image)
    if cv2.waitKey(5) & 0xFF == 27:
        break

