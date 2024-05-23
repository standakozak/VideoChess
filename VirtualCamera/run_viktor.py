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



# Example function
# You can use this function to process the images from opencv
# This function must be implemented as a generator function
def custom_processing(img_source_generator):
    mp_hands = mp.solutions.hands

    for x, sequence in enumerate(img_source_generator):
        if x % 30 == 0:
            with mp_hands.Hands() as hands:
                    # Process the image
                    results = hands.process(sequence)

                    if results.multi_hand_landmarks:
                        for hand_landmarks in results.multi_hand_landmarks:

                            # Get the index finger tip landmark
                            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                            h, w, c = sequence.shape
                            ix, iy = int(index_tip.x * w), int(index_tip.y * h)

                            # Print or use the pixel coordinates
                            print("Index Finger Tip:", ix, iy)
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