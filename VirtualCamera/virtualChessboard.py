import cv2
import numpy as np

import chess  # pip install chess
import chess.engine
import asyncio


def get_bgr_from_rgb(rgb_color):
    r, g, b = rgb_color
    return (b, g, r)


HOVER_SQUARE_COLOR = get_bgr_from_rgb((255, 0, 0))
SELECTED_SQUARE_COLOR = get_bgr_from_rgb((26, 36, 130))
BLACK_SQUARE_COLOR = get_bgr_from_rgb((0, 0, 0))
WHITE_SQUARE_COLOR = get_bgr_from_rgb((255, 255, 255))
LEGAL_MOVE_SQUARE_COLOR = get_bgr_from_rgb((50, 168, 66))

class ChessBoard:
    def __init__(self, width=640, height=480, square_size=50, board_size=(8, 8), border_size=4, use_stockfish=True):
        self.width = width
        self.height = height
        self.square_size = square_size
        self.board_size = board_size
        self.border_size = border_size
        self.selected_piece = None
        self.hovered_chess_coords = (0, 0)
        self.coordinate_array = self.init_coordinate_array()
        self.board = chess.Board()
        self.current_piece_legal_moves = []
        self.white_move = True

        self.engine = None
        if use_stockfish:
            #asyncio.run(self.init_engine())
            self.init_engine()

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
        self.piece_img = {
            name: cv2.resize(img, (50, 50)) for name, img in self.piece_img.items()
        }

        self.piece_positions = {
            # Black pieces
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

            # White pieces
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

    def init_engine(self):
        self.engine = chess.engine.SimpleEngine.popen_uci(r"stockfish\stockfish-windows-x86-64-avx2.exe")
        
    def is_piece_white(self, coords: tuple[int]):
        if coords not in self.piece_positions:
            return None

        piece_code = self.piece_positions[coords]
        return piece_code == piece_code.upper()

    def draw_rectangle(self, frame, chess_coords: tuple[int], color: tuple[int], half_border=False):
        edited_frame = frame.copy()
        top_left, bottom_right = self.coordinate_array[chess_coords]
        
        if half_border:
            top_left = [pos + self.border_size // 2 for pos in top_left]
            bottom_right = [pos - self.border_size // 2 for pos in bottom_right]

        cv2.rectangle(edited_frame, top_left, bottom_right, color, self.border_size)
        return edited_frame
    

    def get_board_top_left_corner(self):
        rows, cols = self.board_size
        
        start_pos_x = self.width - int((self.width / 2) + ((rows / 2) * self.square_size))
        start_pos_y = self.height - int((self.height / 2) + ((rows / 2) * self.square_size))

        #start_pos_y = 5
        return start_pos_x, start_pos_y
    

    def get_chess_coords_from_pos(self, x, y):
        """
        Returns tuple[int] | None
        """
        start_x, start_y = self.get_board_top_left_corner()

        selected_row = (y - start_y) // self.square_size
        selected_col = (x - start_x) // self.square_size

        if (0 <= selected_row < self.board_size[0]) and (0 <= selected_col < self.board_size[1]):
            return (selected_row, selected_col)    
        return None
        


    def init_coordinate_array(self):
        """
        Creates a (rows, cols, 2, 2) array with the (x, y) coordinates of top left 
        and bottom right corners for each square.
        """
        
        start_pos_x, start_pos_y = self.get_board_top_left_corner()
        rows, cols = self.board_size
        
        coordinates = np.zeros((rows, cols, 2, 2), dtype=int)

        for r in range(rows):
            for c in range(cols):
                top_left = (start_pos_x + c * self.square_size, start_pos_y + r * self.square_size)
                bottom_right = (start_pos_x + (c + 1) * self.square_size, start_pos_y + (r + 1) * self.square_size)

                coordinates[r, c] = [top_left, bottom_right]
        return coordinates
    
        """
        # Implementation with numpy, approximately the same speed
        index = np.ndindex((rows, cols))
        index_arr = np.array(list(index)).reshape(rows, cols, 2)

        top_lefts = index_arr * self.square_size + np.array((start_pos_x, start_pos_y))
        bottom_rights = (index_arr + 1) * self.square_size + np.array((start_pos_x, start_pos_y))

        final_arr = np.transpose(np.array([top_lefts, bottom_rights]), (1, 2, 0, 3))
        return final_arr
        """

    def draw_virtual_chessboard(self, frame):
        rows, cols = self.board_size

        for r in range(rows):
            for c in range(cols):
                color = WHITE_SQUARE_COLOR if (c + r) % 2 == 0 else BLACK_SQUARE_COLOR
                frame = self.draw_rectangle(frame, (r, c), color, half_border=True)

        frame = self.start_position(frame)
        return frame
    

    def start_position(self, frame):
        for (r, c), piece in self.piece_positions.items():
            top_left, bottom_right = self.coordinate_array[r, c]
            piece_img_np = self.piece_img[piece]
            frame = self._place_piece(frame, piece_img_np, top_left, bottom_right)

        return frame
    

    def _place_piece(self, frame, piece_img, top_left, bottom_right):
        alpha_piece = piece_img[:, :, 3] / 255.0

        roi = frame[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

        for color in range(0, 3):
            roi[:, :, color] = alpha_piece * piece_img[:, :, color] + (1 - alpha_piece) * roi[:, :, color]

        frame[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]] = roi

        return frame
    

    def highlight_hovered_area(self, frame):
        if self.hovered_chess_coords is not None:
            return self.draw_rectangle(frame, self.hovered_chess_coords, HOVER_SQUARE_COLOR, half_border=False)
        
        return frame
    
    def highlight_selected_piece(self, frame):
        if self.selected_piece is not None:
            frame = self.draw_rectangle(frame, self.selected_piece, SELECTED_SQUARE_COLOR, half_border=False)
        return frame

    def highlight_legal_moves(self, frame):
        for move_target in self.current_piece_legal_moves:
            frame = self.draw_rectangle(frame, move_target, LEGAL_MOVE_SQUARE_COLOR, half_border=True)
        return frame


    def update_position(self, x, y):
        """
        Method accessible outside the class
        """
        self.hovered_chess_coords = self.get_chess_coords_from_pos(x, y)
    

    def reset_piece(self):
        self.selected_piece = None
        self.current_piece_legal_moves = []

    def get_uci_from_chess_coord(self, coords):
        return str("abcdefgh"[coords[1]] + str(8 - coords[0]))

    def get_chess_coords_from_uci(self, uci):
        second_coord = "abcdefgh".find(uci[0])
        return (8 - int(uci[1]), second_coord)

    def get_possible_moves_from_coords(self, coords):
        coords_uci = self.get_uci_from_chess_coord(coords)
        filtered_moves = filter(lambda x: (chess.square_name(x.from_square) == coords_uci), self.board.legal_moves)
        return list(map(lambda move: self.get_chess_coords_from_uci(chess.square_name(move.to_square)), filtered_moves))

    def select_piece(self):
        """
        Selects the piece the mouse / finger is pointing at

        Method accessible outside the class
        """
        if self.hovered_chess_coords is None:
            return
        if tuple(self.hovered_chess_coords) in self.piece_positions:
            if self.is_piece_white(self.hovered_chess_coords) == self.white_move:
                self.selected_piece = self.hovered_chess_coords
                self.current_piece_legal_moves = self.get_possible_moves_from_coords(self.selected_piece)

    def move_piece(self, ignore_legal_moves=False):
        """
        Method accessible outside the class
        """
        source = self.selected_piece
        target = self.hovered_chess_coords
        if source is not None and target is not None:
            if ignore_legal_moves or target in self.current_piece_legal_moves:
                self.piece_positions[target] = self.piece_positions.pop(source)

                # Update chess.Board
                self.board.push_uci(self.get_uci_from_chess_coord(source) + self.get_uci_from_chess_coord(target))
                self.reset_piece()

                # Switch moves
                self.white_move = not self.white_move
    
    def engine_move(self):
        if self.engine is not None and not self.white_move:
            engine_result = self.engine.play(self.board, chess.engine.Limit(time=0.1))
            engine_move = engine_result.move
            source_square = self.get_chess_coords_from_uci(chess.square_name(engine_move.from_square))
            target_square = self.get_chess_coords_from_uci(chess.square_name(engine_move.to_square))

            self.selected_piece = source_square
            self.hovered_chess_coords = target_square
            self.move_piece(ignore_legal_moves=True)

    def draw_board(self, frame):
        """
        Method accessible outside the class
        """
        frame = self.draw_virtual_chessboard(frame)
        frame = self.highlight_legal_moves(frame)
        frame = self.highlight_hovered_area(frame)

        frame = self.highlight_selected_piece(frame)
        return frame


    def main(self):
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        cap = cv2.VideoCapture(0)

        def on_mouse(event, x, y, flags, params):
            if event == cv2.EVENT_MOUSEMOVE:
                self.hovered_chess_coords = self.get_chess_coords_from_pos(x, y)

            if event == cv2.EVENT_LBUTTONDOWN:
                if self.selected_piece is None:
                    self.select_piece()
                else:
                    self.move_piece()
                    #asyncio.run(self.engine_move())
                    self.engine_move()
            if event == cv2.EVENT_RBUTTONDOWN:
                self.reset_piece()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            frame = self.draw_board(frame)

            cv2.imshow('Camera Feed with Virtual Chessboard', frame)
            cv2.setMouseCallback("Camera Feed with Virtual Chessboard", on_mouse)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    ChessBoard().main()
