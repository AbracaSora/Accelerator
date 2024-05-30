import math

import numpy as np
import pyautogui as pa
import cv2 as \
    cv
import dlib as dl
from GazeTrackingAlter.Model import *
from GazeTrackingAlter.Dataset import Dataset

width, height = pa.size()
FaceDetector = dl.get_frontal_face_detector()
LandmarkDetector = dl.shape_predictor('GazeTrackingAlter/model/shape_predictor_68_face_landmarks.dat')


class GazeTracker:
    def __init__(self):
        self.frame = None
        self.model = NewModel()
        self.Dataset = Dataset()

    # 加载模型
    def load(self):
        """
        :return: None

        加载模型
        """
        self.model = load_model('GazeTrackingAlter/model/Accelerator.h5')

    # 保存模型 TODO: 预训练较为准确的模型,作为初始模型
    def save(self):
        """
        :return: None

        保存模型
        """
        self.model.save('GazeTrackingAlter/model/Accelerator.h5')

    # 预处理图像
    @staticmethod
    def preprocess(frame):
        """
        :param frame: 图像
        :return: 预处理后的图像

        人脸检测, 关键点检测, 裁剪眼部, 缩放图像
        """
        Landmarks = FaceDetector(frame) # 人脸检测
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
        # cv.imshow("Eye Tracking", cv.cvtColor(frame, cv.COLOR_RGB2BGR))
        return frame

    # 插入数据
    def insert(self, frame, mouse_pos: tuple[int, int]):
        """
        :param frame: 图像
        :param mouse_pos: 鼠标位置
        :return: 数据集大小

        预处理图像, 插入数据, 返回数据集大小
        """
        frame = self.preprocess(frame)
        if frame is None:
            return
        mouse_pos = np.array(
            [
                mouse_pos[0] / pa.size()[0] * 2 - 1,
                mouse_pos[1] / pa.size()[1] * 2 - 1
            ]
        )
        print(mouse_pos)
        frame = frame.astype(np.float32) / 255.
        self.Dataset.insert((np.expand_dims(frame, axis=0), mouse_pos))
        return self.Dataset.size

    # 训练模型
    def train(self):
        """
        :return: None

        训练模型, 保存模型
        """
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
        self.save()

    # 预测注视点
    def predict(self, frame):
        """
        :param frame: 图像
        :return: 预测的注视点

        预处理图像, 预测注视点
        """
        frame = self.preprocess(frame)
        if frame is None:
            return 0, 0
        frame = frame.astype(np.float32) / 255.
        x, y = self.model.predict(np.expand_dims(frame, axis=0))[0]
        print(x, y)
        x = (x + 1) / 2 * width
        y = (y + 1) / 2 * height
        x, y = map(int, (x, y))
        return x, y
