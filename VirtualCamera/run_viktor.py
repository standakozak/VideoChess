# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 11:59:19 2021

@author: droes
"""
# You can use this library for oberserving keyboard presses
import keyboard # pip install keyboard

from capturing import VirtualCamera
from overlays import initialize_hist_figure, plot_overlay_to_image, plot_strings_to_image, update_histogram
from basics import histogram_figure_numba
import cv2
import mediapipe as mp
import numpy as np
from OwnHandLandmarker import OwnHandLandmarker
from GestureRecognizer import GestureRecognizer, GestureStateHandler


# Example function
# You can use this function to process the images from opencv
# This function must be implemented as a generator function
"""
def custom_processing(img_source_generator):
    hand_landmarker = OwnHandLandmarker(model_path="Viktor/hand_landmarker.task")

    for x, sequence in enumerate(img_source_generator):
        #if x % 30 == 0:
        from OwnHandLandmarker import OwnHandLandmarker
"""

# Example function
# You can use this function to process the images from opencv
# This function must be implemented as a generator function
def custom_processing(img_source_generator):
    hand_landmarker = OwnHandLandmarker(model_path="Viktor/hand_landmarker.task")
    gesture_recognizer = GestureRecognizer(model_path="Stani/gesture_recognizer.task", uses_rgb=True, flip_image=True)
    gesture_state_handler = GestureStateHandler()

    for x, sequence in enumerate(img_source_generator):
        if x % 10 == 0:
            # Make a copy of the image to process
            image_to_process = sequence.copy()

            # Get the index finger tip and mcp coordinates
            index_info = hand_landmarker.get_index_finger_info(image_to_process)

            # If the index finger tip and mcp are detected in the image print the coordinates
            if index_info is not None:
                index_tip = np.array(index_info[0])
                index_mcp = np.array(index_info[1])
                print("Index Finger Info:", index_info[0], index_info[1])
            else:
                index_tip = index_mcp = None
            
            gesture_recognizer.recognize_gesture(image_to_process, index_tip, index_mcp,
                                                 gesture_state_handler)

        # Flip the image
        sequence = cv2.flip(sequence, 1)

        # Make sure to yield your processed image
        yield sequence


def main():
    # change according to your settings
    width = 640      
    height = 480
    fps = 30
    
    # Define your virtual camera
    vc = VirtualCamera(fps, width, height)
    
    vc.virtual_cam_interaction(
        custom_processing(
            # either camera stream
            vc.capture_cv_video(0, bgr_to_rgb=True)
            
            # or your window screen
            # vc.capture_screen()
        )
    )

if __name__ == "__main__":
    main()