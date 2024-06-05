import cv2
from HandTracking.handUtils import HandProcess
import time
from HandTracking.utils import Utils

# 手势识别对象，追踪最开始识别出来的第一只手
handprocess = HandProcess(False, 1)
utils = Utils()
# 计算刷新率
fpsTime = time.time()
camera = cv2.VideoCapture(0)
c = 1
frameRate = 10  # 帧数截取间隔（每隔100帧截取一帧）


# TODO: 更新手势识别,优化识别逻辑,提高识别效率和操作效率
def recognize(image):
    global c
    global fpsTime
    global utils
    global frameRate
    global camera
    img = image
    action_zh = ''
    # if c % frameRate == 0:
    img = cv2.flip(img, 1)
    # 镜像，需要根据镜头位置来调整
    h, w, c = img.shape
    img = handprocess.processOneHand(img)
    # 调用手势识别文件获取手势动作
    img, action, keyPointData = handprocess.checkHandAction(img, drawKeyFinger=True)
    # 通过手势识别得到手势动作，将其画在图像上显示
    action_zh = handprocess.action_labels[action]  # 获取识别出来的手势 例使用if action_zh == '动作1'：进行使用识别结果
    # 显示刷新率FPS
    cTime = time.time()
    fps_text = 1 / (cTime - fpsTime)
    fpsTime = cTime
    c += 1
    return action_zh, img
