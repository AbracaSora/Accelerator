import cv2 as cv
import dlib
import numpy as np

from GazeTracking.Eye import Eye
from GazeTracking.Calibration import Calibration
from GazeTracking.Model import Model


def Mark(frame, point, color=(0, 255, 0)):
    x, y = map(int, point)
    dx = [-5, 5, 0, 0]
    dy = [0, 0, -5, 5]
    for i in range(4):
        cv.line(frame, (x + dx[i], y + dy[i]), (x - dx[i], y - dy[i]), color)
    return frame


class GazeTracker:
    def __init__(self):
        self.frame = None
        self.FaceDetector = dlib.get_frontal_face_detector()
        self.EyeDetector = dlib.shape_predictor("GazeTracking/model/shape_predictor_68_face_landmarks.dat")
        self.eyeRight = None
        self.eyeLeft = None
        self.calibration = Calibration()

    @property
    def Located(self):
        try:
            int(self.eyeLeft.pupil.x)
            int(self.eyeLeft.pupil.y)
            int(self.eyeRight.pupil.x)
            int(self.eyeRight.pupil.y)
            return True
        except TypeError:
            return False

    @property
    def pupilLeftCoords(self):
        if self.Located:
            x = self.eyeLeft.origin[0] + self.eyeLeft.pupil.x
            y = self.eyeLeft.origin[1] + self.eyeLeft.pupil.y
            return x, y

    @property
    def pupilRightCoords(self):
        if self.Located:
            x = self.eyeRight.origin[0] + self.eyeRight.pupil.x
            y = self.eyeRight.origin[1] + self.eyeRight.pupil.y
            return x, y

    @property
    def centerLeft(self):
        if self.Located:
            x = self.eyeLeft.origin[0] + self.eyeLeft.center[0]
            y = self.eyeLeft.origin[1] + self.eyeLeft.center[1]
            return x, y

    @property
    def centerRight(self):
        if self.Located:
            x = self.eyeRight.origin[0] + self.eyeRight.center[0]
            y = self.eyeRight.origin[1] + self.eyeRight.center[1]
            return x, y

    def _update(self):
        frame = cv.cvtColor(self.frame, cv.COLOR_BGR2GRAY)
        Face = self.FaceDetector(frame)
        try:
            Landmarks = self.EyeDetector(frame, Face[0])
        except IndexError:
            return

        try:
            self.eyeLeft = Eye(frame, Landmarks, 0, self.calibration)
            self.eyeRight = Eye(frame, Landmarks, 1, self.calibration)
        except IndexError:
            self.eyeLeft = None
            self.eyeRight = None

    def Track(self, frame):
        self.frame = frame
        self._update()

    def Annotated(self):
        frame = self.frame.copy()

        if self.Located:
            color = (0, 255, 0)
            frame = Mark(frame, self.pupilLeftCoords, color)
            frame = Mark(frame, self.pupilRightCoords, color)

            frame = Mark(frame, self.centerLeft, color)
            frame = Mark(frame, self.centerRight, color)
        return frame

    def Distance(self):
        if self.Located:
            x_left, _ = self.pupilLeftCoords
            x_right, _ = self.pupilRightCoords
            width = x_right - x_left
            std = 6.3
            focus = 300
            distance = (std * focus) / width
            return distance
