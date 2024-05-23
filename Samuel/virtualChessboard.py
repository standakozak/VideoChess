import cv2
import numpy as np

def draw_virtual_chessboard(frame, width, height, square_size=50, board_size=(8, 8)):
    start_x = width - int((width/2)+((board_size[0]/2)*square_size))
    start_y = height - int((height/2)+((board_size[0]/2)*square_size))
    rows, cols = board_size
    
    coordinates = np.zeros((rows, cols, 2, 2), dtype=int)

    for r in range(rows):
        for c in range(cols):
            top_left = (start_x + c * square_size, start_y + r * square_size)
            bottom_right = (start_x + (c + 1) * square_size, start_y + (r + 1) * square_size)
            color = (0, 0, 0) if (c + r) % 2 == 0 else (255, 255, 255)
            cv2.rectangle(frame, top_left, bottom_right, color, -1)

            coordinates[r, c] = [top_left, bottom_right]
    # print(coordinates)
    start_position(frame, coordinates)
    #quit()

def start_position(frame, cords):
    piece_img = {
        'B': cv2.imread(r'Samuel\img\white\bishop.png', cv2.IMREAD_COLOR), 
        'K': cv2.imread(r'Samuel\img\white\king.png', cv2.IMREAD_COLOR), 
        'N': cv2.imread(r'Samuel\img\white\knight.png', cv2.IMREAD_COLOR), 
        'P': cv2.imread(r'Samuel\img\white\pawn.png', cv2.IMREAD_COLOR),
        'Q': cv2.imread(r'Samuel\img\white\queen.png', cv2.IMREAD_COLOR), 
        'R': cv2.imread(r'Samuel\img\white\rook.png', cv2.IMREAD_COLOR),

        'b': cv2.imread(r'Samuel\img\black\bishop.png', cv2.IMREAD_COLOR), 
        'k': cv2.imread(r'Samuel\img\black\king.png', cv2.IMREAD_COLOR), 
        'n': cv2.imread(r'Samuel\img\black\knight.png', cv2.IMREAD_COLOR), 
        'p': cv2.imread(r'Samuel\img\black\pawn.png', cv2.IMREAD_COLOR),
        'q': cv2.imread(r'Samuel\img\black\queen.png', cv2.IMREAD_COLOR), 
        'r': cv2.imread(r'Samuel\img\black\rook.png', cv2.IMREAD_COLOR)
    }

    piece_positions = {
        # White pieces
        (0, 0): 'R',  
        (0, 1): 'N',  
        (0, 2): 'B',  
        (0, 3): 'Q',  
        (0, 4): 'K',  
        (0, 5): 'B',  
        (0, 6): 'N',  
        (0, 7): 'R',  
        (1, 0): 'P',  
        (1, 1): 'P',  
        (1, 2): 'P',  
        (1, 3): 'P',  
        (1, 4): 'P',  
        (1, 5): 'P',  
        (1, 6): 'P',  
        (1, 7): 'P',  

        # Black pieces
        (7, 0): 'r',  
        (7, 1): 'n',  
        (7, 2): 'b',  
        (7, 3): 'q',  
        (7, 4): 'k',  
        (7, 5): 'b',  
        (7, 6): 'n',  
        (7, 7): 'r',  
        (6, 0): 'p',  
        (6, 1): 'p',  
        (6, 2): 'p',  
        (6, 3): 'p',  
        (6, 4): 'p',  
        (6, 5): 'p',  
        (6, 6): 'p',  
        (6, 7): 'p',  
    }

    for (r, c), piece in piece_positions.items():
        top_left, bottom_right = cords[r, c]
        piece_img_np = piece_img[piece]
        place_piece(frame, piece_img_np, top_left, bottom_right)


def place_piece(frame, piece_img, top_left, bottom_right):
    piece_img_resized = cv2.resize(piece_img, (50,50))#
    print(frame)
    #quit()
    frame[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]] = piece_img_resized
    # frame[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]] = piece_img_resized


def main():
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        draw_virtual_chessboard(frame, 640, 480)

        cv2.imshow('Camera Feed with Virtual Checkerboard', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
