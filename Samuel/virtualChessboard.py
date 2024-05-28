import cv2
import numpy as np

class ChessBoard:
    def __init__(self, width=640, height=480, square_size=50, board_size=(8, 8), boarder_size=4):
        self.width = width
        self.height = height
        self.square_size = square_size
        self.board_size = board_size
        self.boarder_size = boarder_size
        self.selected_piece = None
        self.selected_cords = [0,0]
        self.coordinate_array = np.zeros((board_size[0], board_size[1], 2, 2), dtype=int)
    
        self.piece_img = {
            'B': cv2.imread(r'Samuel\img\white\bishop.png', cv2.IMREAD_UNCHANGED), 
            'K': cv2.imread(r'Samuel\img\white\king.png', cv2.IMREAD_UNCHANGED), 
            'N': cv2.imread(r'Samuel\img\white\knight.png', cv2.IMREAD_UNCHANGED), 
            'P': cv2.imread(r'Samuel\img\white\pawn.png', cv2.IMREAD_UNCHANGED),
            'Q': cv2.imread(r'Samuel\img\white\queen.png', cv2.IMREAD_UNCHANGED), 
            'R': cv2.imread(r'Samuel\img\white\rook.png', cv2.IMREAD_UNCHANGED),

            'b': cv2.imread(r'Samuel\img\black\bishop.png', cv2.IMREAD_UNCHANGED), 
            'k': cv2.imread(r'Samuel\img\black\king.png', cv2.IMREAD_UNCHANGED), 
            'n': cv2.imread(r'Samuel\img\black\knight.png', cv2.IMREAD_UNCHANGED), 
            'p': cv2.imread(r'Samuel\img\black\pawn.png', cv2.IMREAD_UNCHANGED),
            'q': cv2.imread(r'Samuel\img\black\queen.png', cv2.IMREAD_UNCHANGED), 
            'r': cv2.imread(r'Samuel\img\black\rook.png', cv2.IMREAD_UNCHANGED)
        }

        self.piece_positions = {
            # White pieces
            (0, 0): 'r',  
            (0, 1): 'n',  
            (0, 2): 'b',  
            (0, 3): 'q',  
            (0, 4): 'k',  
            (0, 5): 'b',  
            (0, 6): 'n',  
            (0, 7): 'r',  
            (1, 0): 'p',  
            (1, 1): 'p',  
            (1, 2): 'p',  
            (1, 3): 'p',  
            (1, 4): 'p',  
            (1, 5): 'p',  
            (1, 6): 'p',  
            (1, 7): 'p',  

            # Black pieces
            (7, 0): 'R',  
            (7, 1): 'N',  
            (7, 2): 'B',  
            (7, 3): 'Q',  
            (7, 4): 'K',  
            (7, 5): 'B',  
            (7, 6): 'N',  
            (7, 7): 'R',  
            (6, 0): 'P',  
            (6, 1): 'P',  
            (6, 2): 'P',  
            (6, 3): 'P',  
            (6, 4): 'P',  
            (6, 5): 'P',  
            (6, 6): 'P',  
            (6, 7): 'P',  
        }


    def draw_virtual_chessboard(self, frame, width, height):
        board_size, square_size = self.board_size, self.square_size
        start_x = width - int((width / 2) + ((board_size[0] / 2) * self.square_size))
        start_y = height - int((height / 2) + ((board_size[0] / 2) * square_size))
        rows, cols = board_size

        coordinates = np.zeros((rows, cols, 2, 2), dtype=int)

        for r in range(rows):
            for c in range(cols):
                top_left = (start_x + c * square_size, start_y + r * square_size)
                bottom_right = (start_x + (c + 1) * square_size, start_y + (r + 1) * square_size)
                color = (0, 0, 0) if (c + r) % 2 == 0 else (255, 255, 255)
                rec_top_left = [top_left[0]+self.boarder_size//2, top_left[1]+self.boarder_size//2]
                rec_bottom_right = [bottom_right[0]-self.boarder_size//2, bottom_right[1]-self.boarder_size//2]
                cv2.rectangle(frame, rec_top_left, rec_bottom_right, color, self.boarder_size)

                coordinates[r, c] = [top_left, bottom_right]

        frame = self.start_position(frame, coordinates)
        self.coordinate_array = coordinates
        return frame, coordinates
    

    def start_position(self, frame, cords):
        for (r, c), piece in self.piece_positions.items():
            top_left, bottom_right = cords[r, c]
            piece_img_np = self.piece_img[piece]
            frame = self._place_piece(frame, piece_img_np, top_left, bottom_right)

        return frame
    

    def _place_piece(self, frame, piece_img, top_left, bottom_right):
        piece_img_resized = cv2.resize(piece_img, (50, 50))
        alpha_piece = piece_img_resized[:, :, 3] / 255.0

        roi = frame[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

        for color in range(0, 3):
            roi[:, :, color] = alpha_piece * piece_img_resized[:, :, color] + (1 - alpha_piece) * roi[:, :, color]

        frame[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]] = roi

        return frame
    

    def select_area(self, frame, coordinates, selected_cords):
        for c in coordinates:
            for coords in c:
                x_min, y_min = coords[0]
                x_max, y_max = coords[1]

                if x_min <= selected_cords[0] <= x_max and y_min <= selected_cords[1] <= y_max:
                    cv2.rectangle(frame, coords[0], coords[1], (0, 0, 255), self.boarder_size)
                    return frame
        return frame
    

    def set_piece(self, coordinates, selected_cords):
        for r in range(len(coordinates)):
            for c in range(len(coordinates[r])):
                x_min, y_min = coordinates[r][c][0]
                x_max, y_max = coordinates[r][c][1]

                if x_min <= selected_cords[0] <= x_max and y_min <= selected_cords[1] <= y_max:
                    if (r, c) in self.piece_positions:
                        self.selected_piece = (r, c)
                        return (r, c), self.piece_positions[(r, c)]
        return None
    

    def move_piece(self, coordinates, selected_cords):
        for r in range(len(coordinates)):
            for c in range(len(coordinates[r])):
                x_min, y_min = coordinates[r][c][0]
                x_max, y_max = coordinates[r][c][1]

                if x_min <= selected_cords[0] <= x_max and y_min <= selected_cords[1] <= y_max:
                    if self.selected_piece:
                        self.piece_positions[(r, c)] = self.piece_positions.pop(self.selected_piece)
                        self.selected_piece = None


    def update_position(self, frame, x, y):
            self.selected_cords[0] = x
            self.selected_cords[1] = y
            frame = self.select_area(frame, self.coordinate_array, self.selected_cords)
            return frame
    

    def reset_piece(self):
        self.selected_piece = None


    def select_piece(self):
        self.set_piece(self.coordinate_array, self.selected_cords)

    
    def place_piece(self):
        self.move_piece(self.coordinate_array, self.selected_cords)


    def main(self):
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        cap = cv2.VideoCapture(0)

        selected_cords = [0, 0]

        def on_mouse(event, x, y, flags, params):
            if event == cv2.EVENT_MOUSEMOVE:
                selected_cords[0] = x
                selected_cords[1] = y
                # frame = self.select_area(frame, coordinate_array, selected_cords)

            if event == cv2.EVENT_LBUTTONDOWN:
                if self.selected_piece is None:
                    pos, piece = self.set_piece(coordinate_array, selected_cords)
                else:
                    self.move_piece(coordinate_array, selected_cords)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            frame, coordinate_array = self.draw_virtual_chessboard(frame, self.width, self.height)
            frame = self.select_area(frame, coordinate_array, selected_cords)

            cv2.imshow('Camera Feed with Virtual Chessboard', frame)
            cv2.setMouseCallback("Camera Feed with Virtual Chessboard", on_mouse)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    ChessBoard().main()
