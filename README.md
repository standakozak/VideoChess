# VideoChess
A computer vision project for playing chess with a webcam. It uses two MediaPipe models for gesture recognition and hand position recognition.

## Initialize
1. Install all dependencies (pip install mediapipe, ...) and **OBS** (only for webcam).
1. Run **VirtualCamera/run.py**.
1. As soon as the webcam activates (Output "Camera Properties"... in Terminal), **open OBS**
1. in OBS -> **Sources** -> plus icon (Add Source) -> **Video Capture Device** -> Create new -> Ok -> Device - choose **OBS Virtual Camera**.
1. You should see the image produced by the code.

Functionalities are listed in the **VirtualCamera/keybinds.txt** file.

The file ``virtualChessboard.py`` contains the chess logic, you can run the file directly to play chess with your mouse.
If you want to play against stockfish, clone [this branch from GitHub](https://github.com/standakozak/VideoChess/tree/chess_engine). Then, you can also change the skill level (between 0 and 20) of the AI opponent when initializing the `ChessBoard()` class. 

``GestureRecognizer.py`` and ``OwnHandLandmarker.py`` contain the recognition models.
