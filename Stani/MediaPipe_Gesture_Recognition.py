"""
Gesture Recognition using MediaPipe and a pre-trained model
described at https://developers.google.com/mediapipe/solutions/vision/gesture_recognizer
"""

### IMPORTS

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

mp_hands = mp.solutions.hands

from cv2.typing import MatLike


### CONSTANTS

## Change path based on your current repository
# The path you see in Terminal + MODEL_PATH should get you to the file

MODEL_PATH = "Stani/gesture_recognizer.task"
#MODEL_PATH = "gesture_recognizer.task"

# Recognizes: Thumb_Up, Thumb_Down, Open_Palm, Pointing_Up, Closed_Fist, Victory, Love, None

EMPTY_HAND_GESTURES = ["Open_Palm"]
FULL_HAND_GESTURES = ["Pointing_Up"]
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

class GestureState:
    def __init__(self, state, gesture):
        self.state = state
        self.gesture = gesture



def resolve_hand_gesture_state_change(
        current_gesture_state: GestureState, previous_gesture_state: GestureState
    ) -> bool:
    if current_gesture_state.state != previous_gesture_state.state:
        print("New gesture:", current_gesture_state.gesture)
        print(GESTURE_STATES_NAMES[current_gesture_state.state])
    
    return current_gesture_state.state != previous_gesture_state.state


def get_gesture_and_state(recognition_result) -> GestureState:
    if recognition_result is None or len(recognition_result.gestures) == 0:
        return GestureState(HandGestureState.NONE, "")

    top_gesture = recognition_result.gestures[0][0]

    for state, state_gestures in GESTURE_STATES_TO_NAMES.items():
        if top_gesture.category_name in state_gestures:
            return GestureState(state, top_gesture.category_name)
    
    return GestureState(HandGestureState.NONE, "")


def get_index_finger_top_and_bottom(image: MatLike, hands: mp.solutions.hands.Hands):
    h, w, _ = image.shape

    results = hands.process(image)
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]

        tip_ix, tip_iy = int(index_tip.x * w), int(index_tip.y * h)
        mcp_ix, mcp_iy = int(index_mcp.x * w), int(index_mcp.y * h)

        return (tip_ix, tip_iy), (mcp_ix, mcp_iy)
    
def get_angle_between_vectors(v, w):
    return np.arccos(v.dot(w) / np.linalg.norm(v) * np.linalg.norm(w))

def rotate_image(image, angle, rotation_center):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    #rot_mat = cv2.getRotationMatrix2D(tuple(rotation_center.astype(int)), angle, 1.0)
    
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


def rotate_by_index_finger(image: MatLike, hands: mp.solutions.hands.Hands) -> MatLike:
    index_finger_landmarks = get_index_finger_top_and_bottom(image, hands)
    if index_finger_landmarks is not None:
        index_top = np.array((index_finger_landmarks[0][0], index_finger_landmarks[0][1]))
        index_bottom = np.array((index_finger_landmarks[1][0], index_finger_landmarks[1][1]))

        finger_vector = index_top - index_bottom
        basis_vector = np.array((0, -1))

        angle = get_angle_between_vectors(finger_vector, basis_vector)

        # -1 -> must be rotated clockwise, 1 -> must be rotated counter-clockwise
        rotation_orientation = int(np.cross(finger_vector, basis_vector) <= 0) * 2 - 1
        
        angle = np.rad2deg(angle * rotation_orientation)
        
        rotated_image = rotate_image(image, angle, index_top)
        return rotated_image


if __name__ == "__main__":
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    
    options = vision.GestureRecognizerOptions(base_options=base_options)
    recognizer = vision.GestureRecognizer.create_from_options(options)

    # Start capturing video from the webcam
    cap = cv2.VideoCapture(0)

    last_gesture_state = get_gesture_and_state(None)

    with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_hands=1, model_complexity=0) as hands:

        while cap.isOpened():
            success, image = cap.read()
            if not success:
                continue
            
            # Convert the image from BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Flipping the image
            image = cv2.flip(image, 1)
            image_to_process = image
            image_to_show = image

            rotated_image = rotate_by_index_finger(image, hands)
            
            if rotated_image is not None:
                image_to_process = rotated_image

                # Test if the image is rotated properly:
                #image_to_show = rotated_image

            # Process the image
            mp_image: mp.Image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_to_process)
            recognition_result = recognizer.recognize(mp_image)

            current_gesture_state = get_gesture_and_state(recognition_result)
            resolve_hand_gesture_state_change(current_gesture_state, last_gesture_state)
            
            # Convert the image back to BGR for displaying
            image_to_show = cv2.cvtColor(image_to_show, cv2.COLOR_RGB2BGR)

            # Display the annotated image
            cv2.imshow('Hand Tracking', image_to_show)
            if cv2.waitKey(5) & 0xFF == 27:
                break
            
            last_gesture_state = current_gesture_state

        cap.release()
