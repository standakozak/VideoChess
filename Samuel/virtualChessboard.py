import cv2
import numpy as np

def draw_virtual_chessboard(frame, width, height, square_size=50, board_size=(8, 8)):
    start_x = width - int((width/2)+((board_size[0]/2)*square_size))
    start_y = height - int((height/2)+((board_size[0]/2)*square_size))
    rows, cols = board_size
    
    for i in range(rows):
        for j in range(cols):
            top_left = (start_x + j * square_size, start_y + i * square_size)
            bottom_right = (start_x + (j + 1) * square_size, start_y + (i + 1) * square_size)
            color = (0, 0, 0) if (i + j) % 2 == 0 else (255, 255, 255)
            cv2.rectangle(frame, top_left, bottom_right, color, -1)

def main():
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        draw_virtual_chessboard(frame, 640, 480)

        cv2.imshow('Camera Feed with Virtual Checkerboard', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
