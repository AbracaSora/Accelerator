import math
import sys
import time

import numpy as np
import cv2
import mediapipe as mp
from HandTracking.utils import Utils


class HandProcess():
    def __init__(self, static_image_mode=False, max_num_hands=2):
        # 参数
        self.mp_drawing = mp.solutions.drawing_utils  # 初始化medialpipe的画图函数
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_hands = mp.solutions.hands  # 初始化手掌检测对象
        # 调用mediapipe的Hands函数，输入手指关节检测的置信度和上一帧跟踪的置信度，输入最多检测手的数目，进行关节点检测
        self.hands = self.mp_hands.Hands(static_image_mode=static_image_mode,
                                         min_detection_confidence=0.7,
                                         min_tracking_confidence=0.5,
                                         max_num_hands=max_num_hands)
        # 初始化一个列表来存储
        self.landmark_list = []
        # 定义所有的手势动作对应的鼠标操作
        self.action_labels = {
            'none': '无',
            'scroll_up': '向前翻页',
            'scroll_down': '向后翻页',
            'control_up': '放大',
            'control_down': '缩小',
            'identify':'识别'
        }
        self.action_deteted = ''
        # 上一帧的拇指和中指距离距离
        self.l0 = 100
        self.last_action_time = {'scroll_up': 0, 'scroll_down': 0, 'identify': 0}  # 添加一个字典来记录每个操作的上次执行时间戳
        self.action_cooldown = {'scroll_up': 0.5, 'scroll_down': 0.5, 'identify': 5}  # 定义动作间隔时间

    def checkHandsIndex(self, handedness):
        # 判断数量
        if len(handedness) == 1:
            handedness_list = [handedness[0].classification[0].label]
        else:
            handedness_list = [handedness[0].classification[0].label, handedness[1].classification[0].label]
        return
        # 计算两点点的距离

    def getDistance(self, pointA, pointB):
        # math.hypot为勾股定理计算两点长度的函数，得到食指和拇指的距离
        return math.hypot((pointA[0] - pointB[0]), (pointA[1] - pointB[1]))

        # 获取手指在图像中的坐标

    def getFingerXY(self, index):
        return self.landmark_list[index][1], self.landmark_list[index][2]

        # 将手势识别的结果绘制到图像上，根据不同的识别结果展示不同的绘制方式

    def drawInfo(self, img, action, flag):
        thumbXY, indexXY = map(self.getFingerXY, [4, 8])

        if action == 'control_up':
            img = cv2.circle(img, thumbXY, 20, (255, 0, 255), -1)
            img = cv2.circle(img, indexXY, 20, (255, 0, 255), -1)
            img = cv2.line(img, indexXY, thumbXY, (255, 0, 255), 2)
        elif action == 'control_down':
            img = cv2.circle(img, thumbXY, 20, (0, 0, 255), -1)
            img = cv2.circle(img, indexXY, 20, (0, 0, 255), -1)
            img = cv2.line(img, indexXY, thumbXY, (0, 0, 255), 2)
        return img

        # 返回手掌各种动作

    def checkHandAction(self, img, drawKeyFinger=True):
        current_time = time.time()  # 获取当前时间
        upList = self.checkFingersUp()
        action = 'none'
        flag = 0
        keyPointData = []  # 存储手部关键点数据的列表

        if len(upList) == 0:
            return img, action, keyPointData

        # 在检测到动作时，构建手部关键点数据
        for landmark in self.landmark_list:
            landmark_id, p_x, p_y, _ = landmark
            keyPointData.append((p_x, p_y))  # 添加手部关键点的坐标信息到列表中

        # 侦测距离
        dete_dist = 100

        # 中指
        key_point = self.getFingerXY(8)

        # 向上滑：五指向上
        if upList == [1, 1, 1, 1, 1]:
            action = 'scroll_up'
            flag =1

        # 向下滑：除拇指外四指向上
        if upList == [0, 1, 1, 1, 1]:
            action = 'scroll_down'
            flag = 1

        # 识别：食指和中指向上
        if upList == [0, 1, 1, 0, 0]:
            action = 'identify'
            flag = 1

        # 放大或者缩小：食指与拇指出现暂停移动，如果两指相互靠近，触发缩小，如果相互远离，触发放大
        if upList == [1, 1, 0, 0, 0]:
            l1 = self.getDistance(self.getFingerXY(4), self.getFingerXY(8))
            if l1 > 120:
                action = 'control_up'
                flag = 1
            elif l1 < 50:
                action = 'control_down'
                flag = 1
            #self.l0 = l1
            # 获取展示放大缩小效果
            min_volume = 100
            max_volume = 1

        if action in self.action_cooldown and current_time - self.last_action_time[action] < self.action_cooldown[
            action]:
            flag = 0

        if flag != 0:
            self.last_action_time[action] = current_time  # 更新该操作的上次执行时间戳

            # 根据动作绘制相关点
        img = self.drawInfo(img, action, flag) if drawKeyFinger else img
        self.action_deteted = self.action_labels[action]

        if flag == 0:
            action = 'none'

        return img, action, keyPointData

        # 返回向上手指的数组

    def checkFingersUp(self):

        fingerTipIndexs = [4, 8, 12, 16, 20]
        upList = []
        if len(self.landmark_list) == 0:
            return upList
        # 拇指，比较x坐标
        if self.landmark_list[fingerTipIndexs[0]][1] < self.landmark_list[fingerTipIndexs[0] - 1][1]:
            upList.append(1)
        else:
            upList.append(0)

        # 其他指头，比较Y坐标
        for i in range(1, 5):
            if self.landmark_list[fingerTipIndexs[i]][2] < self.landmark_list[fingerTipIndexs[i] - 2][2]:
                upList.append(1)
            else:
                upList.append(0)

        return upList

    # 分析手指的位置，得到手势动作
    def processOneHand(self, img, drawBox=True, drawLandmarks=True):
        utils = Utils()
        results = self.hands.process(img)
        self.landmark_list = []

        if results.multi_hand_landmarks:

            for hand_index, hand_landmarks in enumerate(results.multi_hand_landmarks):

                if drawLandmarks:
                    self.mp_drawing.draw_landmarks(
                        img,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style())

                # 遍历landmark

                for landmark_id, finger_axis in enumerate(hand_landmarks.landmark):
                    h, w, c = img.shape
                    p_x, p_y = math.ceil(finger_axis.x * w), math.ceil(finger_axis.y * h)

                    self.landmark_list.append([
                        landmark_id, p_x, p_y,
                        finger_axis.z
                    ])

                # 框框和label
                if drawBox:
                    x_min, x_max = min(self.landmark_list, key=lambda i: i[1])[1], \
                        max(self.landmark_list, key=lambda i: i[1])[1]
                    y_min, y_max = min(self.landmark_list, key=lambda i: i[2])[2], \
                        max(self.landmark_list, key=lambda i: i[2])[2]
                    img = cv2.rectangle(img, (x_min - 30, y_min - 30), (x_max + 30, y_max + 30), (0, 255, 0), 2)
                    img = utils.cv2AddChineseText(img, self.action_deteted, (x_min - 20, y_min - 120),
                                                  textColor=(255, 0, 255), textSize=60)

        return img

    def process(self, img, draw=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.hands_data = self.hand_detector.process(img_rgb)
        if draw:
            if self.hands_data.multi_hand_landmarks:
                for handlms in self.hands_data.multi_hand_landmarks:
                    self.drawer.draw_landmarks(img, handlms, mp.solutions.hands.HAND_CONNECTIONS)

    def find_position(self, img):
        h, w, c = img.shape
        self.position = {'Left': {}, 'Right': {}}
        if self.hands_data.multi_hand_landmarks:
            i = 0
            for point in self.hands_data.multi_handedness:
                score = point.classification[0].score
                if score >= 0.8:
                    label = point.classification[0].label
                    hand_lms = self.hands_data.multi_hand_landmarks[i].landmark
                    for id, lm in enumerate(hand_lms):
                        x, y = int(lm.x * w), int(lm.y * h)
                        self.position[label][id] = (x, y)
                i = i + 1
        return self.position

    def fingers_count(self, hand='Left'):
        tips = [4, 8, 12, 16, 20]
        tip_data = {4: 0, 8: 0, 12: 0, 16: 0, 20: 0}
        for tip in tips:
            ltp1 = self.position[hand].get(tip, None)
            ltp2 = self.position[hand].get(tip - 2, None)
            if ltp1 and ltp2:
                if tip == 4:
                    if ltp1[0] > ltp2[0]:
                        if hand == 'Left':
                            tip_data[tip] = 1
                        else:
                            tip_data[tip] = 0
                    else:
                        if hand == 'Left':
                            tip_data[tip] = 0
                        else:
                            tip_data[tip] = 1
                else:
                    if ltp1[1] > ltp2[1]:
                        tip_data[tip] = 0
                    else:
                        tip_data[tip] = 1
        return list(tip_data.values()).count(1)
