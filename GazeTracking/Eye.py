import cv2 as cv
import dlib
import numpy as np
import math

from GazeTracking.Calibration import Calibration
from GazeTracking.Pupil import Pupil


def Midpoint(p1, p2):
    return (p1.x + p2.x) // 2, (p1.y + p2.y) // 2


class Eye:
    def __init__(self, original_frame, landmarks, side, calibration: Calibration):
        self.frame = None
        self.center = None
        self.pupil = None
        self.landmark_points = None
        self.origin = None

        if side == 0:
            points = [i for i in range(36, 42)]
        else:
            points = [i for i in range(42, 48)]

        self._isolate(original_frame, landmarks, points)

        if not calibration.is_complete():
            calibration.evaluate(self.frame, side)

        threshold = calibration.threshold(side)
        self.pupil = Pupil(self.frame, threshold)

    def _isolate(self, frame, landmarks, points):
        region = np.array(
            [
                (
                    landmarks.part(point).x, landmarks.part(point).y
                ) for point in points
            ]
        )
        region = region.astype(np.int32)
        self.landmark_points = region

        height, width = frame.shape[:2]
        black_frame = np.zeros((height, width), np.uint8)
        mask = np.full((height, width), 255, np.uint8)
        cv.fillPoly(mask, [region], (0, 0, 0))
        eye = cv.bitwise_not(black_frame, frame.copy(), mask=mask)

        margin = 5
        min_x = np.min(region[:, 0]) - margin
        max_x = np.max(region[:, 0]) + margin
        min_y = np.min(region[:, 1]) - margin
        max_y = np.max(region[:, 1]) + margin

        self.frame = eye[min_y:max_y, min_x:max_x]
        self.origin = (min_x, min_y)

        height, width = self.frame.shape[:2]
        self.center = (width / 2, height / 2)

    def _blinking_ratio(self, landmarks, points):
        left = (landmarks.part(points[0]).x, landmarks.part(points[0]).y)
        right = (landmarks.part(points[3]).x, landmarks.part(points[3]).y)
        top = Midpoint(landmarks.part(points[1]), landmarks.part(points[2]))
        bottom = Midpoint(landmarks.part(points[5]), landmarks.part(points[4]))

        eye_width = math.hypot((left[0] - right[0]), (left[1] - right[1]))
        eye_height = math.hypot((top[0] - bottom[0]), (top[1] - bottom[1]))

        try:
            ratio = eye_width / eye_height
        except ZeroDivisionError:
            ratio = None

        return ratio
