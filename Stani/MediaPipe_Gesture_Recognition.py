"""
Gesture Recognition using MediaPipe and a pre-trained model
described at https://developers.google.com/mediapipe/solutions/vision/gesture_recognizer
"""

### IMPORTS

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

### CONSTANTS

## Change path based on your current repository
# The path you see in Terminal + MODEL_PATH should get you to the file
MODEL_PATH = "ProjektArbeit/VideoChess/Stani/gesture_recognizer.task"
#MODEL_PATH = "gesture_recognizer.task"

EMPTY_HAND_GESTURES = ["Open_Palm", "Pointing_Up"]  # "Thumb_Up"
FULL_HAND_GESTURES = ["Closed_Fist", ] # Maybe also "None"?
RESET_GESTURES = ["Victory"]

class HandGestureState:
    HOLDING_PIECE = 1
    EMPTY = 2
    RESET = 3
    NONE = 0

GESTURE_STATES_NAMES = {
    HandGestureState.HOLDING_PIECE: "Hand holding a chess piece.",
    HandGestureState.EMPTY: "Empty hand without chess piece.",
    HandGestureState.RESET: "The hand was resetted (returning piece).",
    HandGestureState.NONE: "Please, move the hand in front of the camera."
}

GESTURE_STATES_TO_NAMES = {
    HandGestureState.HOLDING_PIECE: FULL_HAND_GESTURES,
    HandGestureState.EMPTY: EMPTY_HAND_GESTURES,
    HandGestureState.RESET: RESET_GESTURES
}


def resolve_hand_gesture_state_change(current_gesture_state, previous_gesture_state):
    if current_gesture_state["state"] != previous_gesture_state["state"]:
        print("New gesture:", current_gesture_state["gesture"])
        print(GESTURE_STATES_NAMES[current_gesture_state["state"]])
    
    return current_gesture_state["state"] != previous_gesture_state["state"]

def get_gesture_and_state(recognition_result):
    if recognition_result is None or len(recognition_result.gestures) == 0:
        return {"state": HandGestureState.NONE, "gesture": ""}

    top_gesture = recognition_result.gestures[0][0]

    for state, state_gestures in GESTURE_STATES_TO_NAMES.items():
        if top_gesture.category_name in state_gestures:
            return {"state": state, "gesture": top_gesture.category_name}
    
    return {"state": HandGestureState.NONE, "gesture": ""}


if __name__ == "__main__":    
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    
    options = vision.GestureRecognizerOptions(base_options=base_options)
    recognizer = vision.GestureRecognizer.create_from_options(options)

    # Start capturing video from the webcam
    cap = cv2.VideoCapture(0)

    last_gesture_state = get_gesture_and_state(None)

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue
        
        # Convert the image from BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Flipping the image
        image = cv2.flip(image, 1)

        # Process the image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        recognition_result = recognizer.recognize(mp_image)

        current_gesture_state = get_gesture_and_state(recognition_result)
        resolve_hand_gesture_state_change(current_gesture_state, last_gesture_state)

        # Convert the image back to BGR for displaying
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Display the annotated image
        cv2.imshow('Hand Tracking', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break
        
        last_gesture_state = current_gesture_state

    cap.release()
