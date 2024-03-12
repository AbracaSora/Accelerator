import numpy as np
import pyautogui as pa
import cv2 as cv
import dlib as dl
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
        minx = Landmarks.part(37).x - 5;
        miny = Landmarks.part(38).y - 5;
        maxx = Landmarks.part(46).x + 5;
        maxy = Landmarks.part(47).y + 5;
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        frame = frame[miny:maxy, minx:maxx]
        frame = cv.copyMakeBorder(frame, 5 + (20 - frame.shape[0]) // 2, 5 + (21 - frame.shape[0]) // 2,5 + (90 - frame.shape[1]) // 2,
                          5 + (91 - frame.shape[1]) // 2, cv.BORDER_CONSTANT, value=[0, 0, 0])
        cv.imshow("Eye Tracking", frame)
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
        frame = img_to_array(frame)
        frame /= 255.
        self.Dataset.insert((frame, mouse_pos))

    def train(self):
        tr_x = np.array(self.Dataset.train[0])
        tr_y = np.array(self.Dataset.train[1])
        val_x = np.array(self.Dataset.validation[0])
        val_y = np.array(self.Dataset.validation[1])
        self.model.fit(tr_x, tr_y, epochs=5, batch_size=32, validation_data=[val_x, val_y])

    def predict(self, frame):
        frame = self.preprocess(frame)
        if frame is None:
            return 0, 0
        frame = img_to_array(frame)
        frame /= 255.
        frame = np.array([frame])
        x, y = self.model.predict([frame])[0]
        return x, y
