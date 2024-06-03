"""
Gesture Recognition using MediaPipe and a pre-trained model
described at https://developers.google.com/mediapipe/solutions/vision/gesture_recognizer
"""

## IMPORTANT: IF YOU WANT TO CHANGE THIS FILE, DO IT IN VirtualCamera/GestureRecognizer.py!

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
    def __init__(self, state: HandGestureState, gesture):
        self.state = state
        self.gesture = gesture


class GestureStateHandler:
    def __init__(self, state=None, gesture=None) -> None:
        self.last_gesture_state = GestureStateHandler.get_gesture_and_state(None)

        if state is None and gesture is None:
            self.current_gesture_state = GestureStateHandler.get_gesture_and_state(None)
        else:
            self.current_gesture_state = GestureState(state, gesture)

        self.unresolved_state_change = False

    def get_gesture_and_state(recognition_result) -> GestureState:
        if recognition_result is None or len(recognition_result.gestures) == 0:
            return GestureState(HandGestureState.NONE, "")

        top_gesture = recognition_result.gestures[0][0]

        for state, state_gestures in GESTURE_STATES_TO_NAMES.items():
            if top_gesture.category_name in state_gestures:
                return GestureState(state, top_gesture.category_name)
        
        return GestureState(HandGestureState.NONE, "")
    
    def update_gesture_state(self, new_gesture_state: GestureState) -> None:
        self.last_gesture_state = self.current_gesture_state
        self.current_gesture_state = new_gesture_state
    
    def update_from_recognition_result(self, recognition_result):
        self.update_gesture_state(GestureStateHandler.get_gesture_and_state(recognition_result))
        
        state_changed = self.resolve_hand_gesture_state_change()
        if state_changed:
            self.unresolved_state_change = True
        
    def get_unresolved_state_change(self) -> bool:
        return_state = self.unresolved_state_change
        self.unresolved_state_change = False
        return return_state

    def resolve_hand_gesture_state_change(self) -> bool:
        if (self.current_gesture_state.state != self.last_gesture_state.state and
             self.current_gesture_state.state != HandGestureState.NONE):
                print("New gesture:", self.current_gesture_state.gesture)
                print(GESTURE_STATES_NAMES[self.current_gesture_state.state])
            
        return self.current_gesture_state.state != self.last_gesture_state.state


def get_index_finger_tip_and_mcp(image: MatLike, hands: mp.solutions.hands.Hands):
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


def rotate_by_index_finger(image: np.ndarray, index_top: np.ndarray, index_bottom: np.ndarray) -> np.ndarray:
    finger_vector = (index_top - index_bottom).astype(float)
    basis_vector = np.array((0.0, -1.0))

    angle = get_angle_between_vectors(finger_vector, basis_vector)

    # -1 -> must be rotated clockwise, 1 -> must be rotated counter-clockwise
    rotation_orientation = int(np.cross(finger_vector, basis_vector) <= 0) * 2 - 1
    
    angle = np.rad2deg(angle * rotation_orientation)
    
    rotated_image = rotate_image(image, angle, index_top)
    return rotated_image, angle


def preprocess_image(image, uses_rgb=False, flip_image=True, 
                     finger_tip_pos: np.ndarray[int] = None, finger_mcp_pos: np.ndarray[int] = None):
    angle_deg = 0
    # Convert the image from BGR to RGB
    if not uses_rgb:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Flipping the image
    if flip_image:
        image = cv2.flip(image, 1)

    if finger_tip_pos is not None and finger_mcp_pos is not None:
        rotated_image, angle_deg = rotate_by_index_finger(image, finger_tip_pos, finger_mcp_pos)
    
        if rotated_image is not None:
            image = rotated_image
    return image, angle_deg

def rotate_vector(vector, angle_deg, center_of_rotation):
    centered_vector = vector - center_of_rotation
    theta = np.radians(angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c)))
    return ((R @ centered_vector) + center_of_rotation).astype(int)

def get_hand_landmarks(recognition_result, image_width, image_height, rotation_vector):
    if recognition_result.hand_landmarks is not None:
        for hand_landmark in recognition_result.hand_landmarks:
            # Get the index finger tip landmark
            index_tip = hand_landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]

            # Get the index finger mcp landmark
            index_mcp = hand_landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_MCP]

            # Get the x and y coordinates of the index finger tip
            index_ix, index_iy = int(index_tip.x * image_width), int(index_tip.y * image_height)

            # Get the x and y coordinates of the index finger mcp
            mcp_ix, mcp_iy = int(index_mcp.x * image_width), int(index_mcp.y * image_height)

            index_tip_np = np.array([index_ix, index_iy])
            index_mcp_np = np.array([mcp_ix, mcp_iy])

            image_center = np.array((image_width, image_height)) / 2

            return (rotate_vector(index_tip_np, rotation_vector, image_center), rotate_vector(index_mcp_np, rotation_vector, image_center))

            return (index_ix, index_iy), (mcp_ix, mcp_iy)
        return None, None
        
    else:
        return None, None
        


class GestureRecognizer:
    def __init__(self, model_path: str, uses_rgb=True, flip_image=True):
        base_options = python.BaseOptions(model_asset_path=model_path)
    
        options = vision.GestureRecognizerOptions(base_options=base_options)
        self.recognizer = vision.GestureRecognizer.create_from_options(options)
        self.uses_rgb = uses_rgb
        self.flip_image = flip_image

    def recognize_gesture(self, image, finger_tip_pos: np.ndarray[int], 
                          finger_mcp_pos: np.ndarray[int], gesture_handler: GestureStateHandler):
        preprocessed_image, rotation = preprocess_image(image, self.uses_rgb, self.flip_image, 
                                              finger_tip_pos, finger_mcp_pos)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=preprocessed_image)
        recognition_result = self.recognizer.recognize(mp_image)

        gesture_handler.update_from_recognition_result(recognition_result)
        new_finger_tip, new_finger_mcp = get_hand_landmarks(recognition_result, mp_image.width, mp_image.height, rotation)
        return preprocessed_image, (new_finger_tip, new_finger_mcp)
        

if __name__ == "__main__":
    # Start capturing video from the webcam
    cap = cv2.VideoCapture(0)
    
    gesture_state_handler = GestureStateHandler()
    gesture_recognizer = GestureRecognizer(model_path=MODEL_PATH, uses_rgb=False, flip_image=False)

    with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5, 
                        max_num_hands=1, model_complexity=0) as hands:
        index_tip = np.array((1, 0))
        index_mcp = np.array((0, 0))
        while cap.isOpened():
            success, image = cap.read()
            img_to_show = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
            if not success:
                continue
            
            #image = cv2.flip(image, 1)
            #index_finger_landmarks = get_index_finger_tip_and_mcp(image, hands)
            if index_tip is None:
                index_tip = np.array((1, 0))
            if index_mcp is None:
                index_mcp = np.array((0, 0))

            #if index_finger_landmarks is not None:
                #index_top = np.array((index_finger_landmarks[0][0], index_finger_landmarks[0][1]))
                #index_bottom = np.array((index_finger_landmarks[1][0], index_finger_landmarks[1][1]))
                
            processed_img, finger_info = gesture_recognizer.recognize_gesture(image, index_tip, index_mcp,
                                                    gesture_state_handler)

            index_tip, index_mcp = finger_info
            print(index_tip)
            #img_to_show, _ = preprocess_image(image, False, False, index_top, index_bottom)
            #img_to_show, _ = preprocess_image(image, False, False, None, None)
            if index_tip is not None and index_mcp is not None:
                img_to_show = cv2.line(img_to_show, (int(index_tip[0]), int(index_tip[1])), (int(index_mcp[0]), int(index_mcp[1])), (0, 0, 0), 5)

            # Display the annotated image
            cv2.imshow('Hand Tracking', cv2.cvtColor(cv2.flip(img_to_show, 1), cv2.COLOR_RGB2BGR))
            if cv2.waitKey(5) & 0xFF == 27:
                break
        
        cap.release()
