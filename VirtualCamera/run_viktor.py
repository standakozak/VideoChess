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
from GestureRecognizer import GestureRecognizer, GestureStateHandler, HandGestureState
from virtualChessboard import ChessBoard
from KeyPressed import KeyPressed

from timeit import timeit

def resolve_gesture_state(gesture_handler: GestureStateHandler, chessboard: ChessBoard):
    if not gesture_handler.resolve_hand_gesture_state_change():
        return
    
    if gesture_handler.current_gesture_state.state == HandGestureState.HOLDING_PIECE:
        chessboard.select_piece()
    elif gesture_handler.current_gesture_state.state == HandGestureState.EMPTY:
        chessboard.move_piece()
    elif gesture_handler.current_gesture_state.state == HandGestureState.RESET:
        chessboard.reset_piece()


def custom_processing(img_source_generator):
    #hand_landmarker = OwnHandLandmarker(model_path="Viktor/hand_landmarker.task")
    gesture_recognizer = GestureRecognizer(model_path="Stani/gesture_recognizer.task", uses_rgb=True, flip_image=False)
    gesture_state_handler = GestureStateHandler()
    chessBoard = ChessBoard(640, 480, square_size=50, border_size=4)

    # Initialize the key presser and set up the hook on the keyboard events
    keyPresser = KeyPressed()
    keyboard.on_press(keyPresser.on_key_event)

    # Initialize the chessboard position
    chessboard_pos_x, chessboard_pos_y = -50, -50
    index_tip, index_mcp = (None, None)

    for x, sequence in enumerate(img_source_generator):
        # Flip the image
        image_to_show = cv2.flip(sequence, 1).copy()

        # Speed up performance by ignoring frames
        if x % 1 == 0 and keyPresser.get_last_key() == 'h':
            # Make a copy of the image to process
            image_to_process = image_to_show.copy()
            
            # Get the index finger tip and mcp coordinates
            #index_info = hand_landmarker.get_index_finger_info(image_to_process)
            # hh
            # If the index finger tip and mcp are detected in the image print the coordinates

            _, index_info = gesture_recognizer.recognize_gesture(image_to_process,
                                                        index_tip,
                                                        index_mcp, gesture_state_handler
                                                    )
            if index_info[0] is not None and index_info[1] is not None:
                index_tip = np.array(index_info[0])
                index_mcp = np.array(index_info[1])
                chessboard_pos_x, chessboard_pos_y = index_info[0][0], index_info[0][1]
            else:
                index_tip, index_mcp = (None, None)
                chessboard_pos_x, chessboard_pos_y = -50, -50
        
        if keyPresser.get_last_key() == 'h':
            # Do every frame
            # Update the gesture state
            #print("Update gesture state:")
            #print(timeit(lambda: resolve_gesture_state(gesture_state_handler, chessBoard), number=10000))
            # Original: 0.005215299999999701 s for 10_000 cycles

            resolve_gesture_state(gesture_state_handler, chessBoard)


            # Update the chessboard position
            #print("Update Position:")
            #print(timeit(lambda: chessBoard.update_position(chessboard_pos_x, chessboard_pos_y), number=10000))
            # Original: 0.021515499999999577 s for 10_000 cycles

            chessBoard.update_position(chessboard_pos_x, chessboard_pos_y)

            # Draw the chessboard on the image
            #print("Draw chessboard:")
            
            #print(timeit(lambda: chessBoard.draw_board(image_to_show), number=1000))
            # Original: 6.842462799999998 s for 1000 cycles
            # Initial resize: 5.605389100000002 s for 1000 cycles
            # With pieces overlay: 4.201044999999997 s for 1 000 cycles
            image_to_show = chessBoard.draw_board(image_to_show)

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