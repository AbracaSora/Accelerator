import math

import numpy as np
import pyautogui as pa
import cv2 as cv
import dlib as dl
from PIL import Image, ImageOps
from keras.preprocessing.image import img_to_array
from GazeTrackingAlter.Model import Model
from GazeTrackingAlter.Dataset import Dataset

FaceDetector = dl.get_frontal_face_detector()
LandmarkDetector = dl.shape_predictor('GazeTrackingAlter/model/shape_predictor_68_face_landmarks.dat')


class GazeTracker:
    def __init__(self):
        self.frame = None
        self.model = Model
        self.Dataset = Dataset()

    @staticmethod
    def preprocess(frame):
        Landmarks = FaceDetector(frame)
        if len(Landmarks) == 0:
            return None
        Landmarks = LandmarkDetector(frame, Landmarks[0])
        minx = Landmarks.part(37).x - 8
        miny = Landmarks.part(38).y - 8
        maxx = Landmarks.part(46).x + 8
        maxy = Landmarks.part(47).y + 8
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        frame = frame[miny:maxy, minx:maxx]
        frame = cv.resize(frame, (50, 25))
        cv.imshow("Eye Tracking", cv.cvtColor(frame, cv.COLOR_RGB2BGR))
        return frame

    def insert(self, frame, mouse_pos: tuple[int, int]):
        frame = self.preprocess(frame)
        if frame is None:
            return
        mouse_pos = np.array([
            mouse_pos[0] / pa.size()[0] * 2 - 1,
            mouse_pos[1] / pa.size()[1] * 2 - 1
        ]
        )
        print(mouse_pos)
        frame = frame.astype(np.float32) / 255.
        self.Dataset.insert((np.expand_dims(frame, axis=0), mouse_pos))

    def train(self):
        tr_x = np.concatenate(self.Dataset.train[0], axis=0)
        tr_y = np.array(self.Dataset.train[1])
        val_x = np.concatenate(self.Dataset.validation[0])
        val_y = np.array(self.Dataset.validation[1])
        bs = math.floor(self.Dataset.size / 10)
        if bs < 4:
            bs = 4
        elif bs > 64:
            bs = 64
        self.model.fit(tr_x, tr_y, epochs=20, batch_size=bs, validation_data=[val_x, val_y])

    def predict(self, frame):
        frame = self.preprocess(frame)
        if frame is None:
            return 0, 0
        frame = frame.astype(np.float32) / 255.
        x,y = self.model.predict(np.expand_dims(frame, axis=0))[0]
        # print(prediction)
        print(x, y)
        return x, y
