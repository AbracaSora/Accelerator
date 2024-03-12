import numpy as np
import cv2 as cv


class Pupil(object):
    def __init__(self, eye_frame, threshold):
        self.iris_frame = None
        self.threshold = threshold
        self.x = None
        self.y = None

        self.detect_iris(eye_frame)

    @staticmethod
    def image_processing(eye_frame, threshold):
        kernel = np.ones((3, 3), np.uint8)
        new_frame = cv.bilateralFilter(eye_frame, 10, 15, 15)
        new_frame = cv.erode(new_frame, kernel, iterations=3)
        new_frame = cv.threshold(new_frame, threshold, 255, cv.THRESH_BINARY)[1]

        return new_frame

    def detect_iris(self, eye_frame):
        self.iris_frame = self.image_processing(eye_frame, self.threshold)

        contours, _ = cv.findContours(self.iris_frame, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)[-2:]
        contours = sorted(contours, key=cv.contourArea)

        try:
            moments = cv.moments(contours[-2])
            self.x = int(moments['m10'] / moments['m00'])
            self.y = int(moments['m01'] / moments['m00'])
        except (IndexError, ZeroDivisionError):
            pass
