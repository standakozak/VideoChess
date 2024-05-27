import mediapipe as mp
import cv2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# This file contains the HandLandmarker class which is used to get the landmarks of a hand in an image using the MediaPipe library.
# For the recogniton of the landmarks the hand_landmarker.task file is used.
class OwnHandLandmarker:
    def __init__(self, model_path):
        base_options = python.BaseOptions(model_asset_path=model_path)
        hand_landmarker_options = vision.HandLandmarkerOptions(base_options=base_options)
        self.hand_landmarker = vision.HandLandmarker.create_from_options(hand_landmarker_options)

    # Function that gets the landmarks of a hand in an image
    # Returns: The landmarks of the hand in the image
    def get_landmarks(self, image):
        return self.hand_landmarker.detect(image=image)
    
    # Function that gets the x and y coordinates of the index finger tip and the index finger mcp
    # Returns: (index_ix, index_iy), (mcp_ix, mcp_iy) if the index finger tip and mcp are detected in the image, None otherwise
    # Parameters:
    # image: The image in which the index finger tip and mcp are to be detected
    def get_index_finger_info(self, image):

        # Flip the image horizontally
        image = cv2.flip(image, 1)

        # Process the image using the MediaPipe representation of an image
        image = mp.Image(image_format = mp.ImageFormat.SRGB, data = image)

        # Process the image
        results = self.get_landmarks(image=image)

        # If there are hand landmarks in the image get the x and y coordinates of the index finger tip
        if results.hand_landmarks is not None:
            for hand_landmark in results.hand_landmarks:
                # Get the index finger tip landmark
                index_tip = hand_landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]

                # Get the index finger mcp landmark
                index_mcp = hand_landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_MCP]

                # Get the x and y coordinates of the index finger tip
                index_ix, index_iy = int(index_tip.x * image.width), int(index_tip.y * image.height)

                # Get the x and y coordinates of the index finger mcp
                mcp_ix, mcp_iy = int(index_mcp.x * image.width), int(index_mcp.y * image.height)

                return (index_ix, index_iy), (mcp_ix, mcp_iy)
            
        else:
            return None
        
    

# # Example usage:
# hand_landmarker = OwnHandLandmarker(model_path="Viktor/hand_landmarker.task")

# # Start capturing video from the webcam
# cap = cv2.VideoCapture(0)

# while cap.isOpened():
#     success, image = cap.read()
#     if not success:
#         continue
    
#     image_to_show = image.copy()

#     image_to_show = cv2.flip(image_to_show, 1)

#     # Convert the image from BGR to RGB
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#     # Process the image
#     results = hand_landmarker.get_index_finger_info(image)

#     if results is not None:
#         print("Index Finger Tip:", results[0], results[1])

#     # Display the annotated image
#     cv2.imshow('Hand Tracking', image_to_show)
#     if cv2.waitKey(5) & 0xFF == 27:
#         break

