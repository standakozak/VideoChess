# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 11:59:19 2021

@author: droes
"""
# You can use this library for oberserving keyboard presses
import keyboard # pip install keyboard

from capturing import VirtualCamera
from overlays import initialize_hist_figure, plot_overlay_to_image, plot_strings_to_image, update_histogram
from basics_own_merged import calculate_statistics, calculate_mode, calculate_entropy, equalize_histogram, plot_histogram
from basics_own_merged import apply_gabor_filter, apply_sobel_filter, apply_sobel_own_implentation, apply_linear_transformation
from basics_own_merged import apply_gabor_own_implementation, GABOR_VALUES_1, GABOR_VALUES_2, GABOR_VALUES_3
import cv2
import mediapipe as mp
import numpy as np
from OwnHandLandmarker import OwnHandLandmarker
from GestureRecognizer import GestureRecognizer, GestureStateHandler, HandGestureState
from virtualChessboard import ChessBoard
from KeyPressed import KeyPressed
import asyncio

def resolve_gesture_state(gesture_handler: GestureStateHandler, chessboard: ChessBoard):
    if not gesture_handler.resolve_hand_gesture_state_change():
        return
    
    if gesture_handler.current_gesture_state.state == HandGestureState.HOLDING_PIECE:
        chessboard.select_piece()
    elif gesture_handler.current_gesture_state.state == HandGestureState.EMPTY:
        chessboard.move_piece()
        #asyncio.run(chessboard.engine_move())
        chessboard.engine_move()
    elif gesture_handler.current_gesture_state.state == HandGestureState.RESET:
        chessboard.reset_piece()


def custom_processing(img_source_generator):
    # Initialize the objects used for the VideoChess implementation
    hand_landmarker = OwnHandLandmarker(model_path="Viktor/hand_landmarker.task")
    gesture_recognizer = GestureRecognizer(model_path="Stani/gesture_recognizer.task", uses_rgb=True, flip_image=True)
    gesture_state_handler = GestureStateHandler()
    chessBoard = ChessBoard(640, 480, square_size=50, border_size=4)
    # Initialize the chessboard position
    chessboard_pos_x, chessboard_pos_y = -50, -50

    # Initialize the key presser and set up the hook on the keyboard events
    keyPresser = KeyPressed()
    keyboard.on_press(keyPresser.on_key_event)

    # Initialize the histogram figure
    fig, ax, background, r_plot, g_plot, b_plot = initialize_hist_figure()
    

    for x, sequence in enumerate(img_source_generator):
        image_to_show = sequence.copy()
        # Flip the image
        image_to_show = cv2.flip(image_to_show, 1).copy()

        # Implementation of the VideoChess
        # Speed up performance by ignoring frames
        if x % 3 == 0 and keyPresser.get_last_key() == 'c':
            # Make a copy of the image to process during the hand gesture recognition
            image_to_process = sequence.copy()
            
            # Get the index finger tip and mcp coordinates
            index_info = hand_landmarker.get_index_finger_info(image_to_process)

            # If the index finger tip and mcp are detected in the image print the coordinates
            if index_info is not None:
                #print("Index Finger Info:", index_info[0], index_info[1])
                index_tip = np.array(index_info[0])
                index_mcp = np.array(index_info[1])

                chessboard_pos_x, chessboard_pos_y = index_info[0][0], index_info[0][1]
            else:
                index_tip, index_mcp = (None, None)
                chessboard_pos_x, chessboard_pos_y = -50, -50

            gesture_recognizer.recognize_gesture(image_to_process,
                                                        index_tip,
                                                        index_mcp, gesture_state_handler
                                                    )
        
        if keyPresser.get_last_key() == 'c':
            # Do every frame
            # Update the gesture state
            resolve_gesture_state(gesture_state_handler, chessBoard)

            # Update the chessboard position
            chessBoard.update_position(chessboard_pos_x, chessboard_pos_y)

            # Draw the chessboard on the image
            image_to_show = chessBoard.draw_board(image_to_show)

        # Display statistics of the current frame
        if keyPresser.get_last_key() == 's':
            mean, std_dev, max, min = calculate_statistics(image_to_show)
            mode = calculate_mode(image_to_show)
            entropy = calculate_entropy(image_to_show)
            strings_of_stats_to_display = ["Mean: ", str(mean), "Std Dev: ", str(std_dev), "Max: ", str(max), "Min: ", str(min), "Mode: ", str(mode), "Entropy: ", str(entropy)]
            image_to_show = plot_strings_to_image(image_to_show, strings_of_stats_to_display, (0, 0, 255), 600, 50)

        # Display the histogram of the current frame
        if keyPresser.get_last_key() == 'h':
            # Load the histogram values
            image_to_show = plot_histogram(image_to_show, ax)

        # Apply a linear transformation to the image
        if keyPresser.get_last_key() == 'l':
            image_to_show = apply_linear_transformation(image_to_show, 50, 1.5)
            strings_of_stats_to_display = ["Brightness c1: ", "50", "Contrast c2: ", "1.5"]
            image_to_show = plot_strings_to_image(image_to_show, strings_of_stats_to_display, (0, 255, 0), 600, 50)

        # Apply a gabor filter to the image
        if keyPresser.get_last_key() == 'g':
            image_to_show = apply_gabor_own_implementation(image_to_show, **GABOR_VALUES_1)
        
        # Apply a gabor filter to the image
        if keyPresser.get_last_key() == 'f':
            image_to_show = apply_gabor_own_implementation(image_to_show, **GABOR_VALUES_2)

        # Apply a sobel edge detection filter to the image
        if keyPresser.get_last_key() == 'e':
            image_to_show = apply_sobel_filter(image_to_show)

        # Apply a sobel edge detection filter both axes to the image
        if keyPresser.get_last_key() == 'b':
           image_to_show = apply_sobel_own_implentation(image_to_show, mode=0)
        
        # Apply a sobel edge detection filter on the x axis to the image
        if keyPresser.get_last_key() == 'x':
            image_to_show = apply_sobel_own_implentation(image_to_show, mode=1)
        
        # Apply a sobel edge detection filter on the y axis to the image
        if keyPresser.get_last_key() == 'y':
            image_to_show = apply_sobel_own_implentation(image_to_show, mode=2)
        
        # Apply histogram equalization to the image
        if keyPresser.get_last_key() == 'p':
            image_to_show = equalize_histogram(image_to_show)
            image_to_show = plot_histogram(image_to_show, ax)

        # Make sure to yield your processed image
        yield image_to_show


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