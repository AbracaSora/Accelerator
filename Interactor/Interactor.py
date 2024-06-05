from flask import Flask, send_file, request
from flask_cors import CORS

import GemeniAI
from GazeTrackingAlter import GazeTracker
from HandTracking import *
from GemeniAI import *

import base64
import pyautogui as pa
import numpy as np
import cv2 as cv
import json
import os
import signal
import atexit
import shutil

mem = (0,0)
width, height = pa.size()  # 获取屏幕的宽高
GT = GazeTracker()  # 创建GazeTracker对象
isTrained = False  # 是否训练
TimeWait = 20  # 等待时间
Cam = cv.VideoCapture(0)  # 打开摄像头

Interactor = Flask(__name__)  # 创建Flask对象
CORS(Interactor)  # 允许跨域
# 临时保存图片的文件夹
UPLOAD_FOLDER = 'temp'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


# 定义释放摄像头资源的函数
def release_camera():
    global Cam
    if Cam.isOpened():
        Cam.release()
    cv.destroyAllWindows()


# 捕捉终止信号并释放摄像头
def signal_handler(signal, frame):
    release_camera()
    shutil.rmtree("temp")
    print('Application closed, camera released.')


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# 注册atexit以在正常退出时释放摄像头
atexit.register(release_camera)


# 上传图片的路由
@Interactor.route('/upload', methods=['POST'])
def upload():
    """
    :return: 上传图片的结果

    上传图片的路由, 上传成功返回200, 上传失败返回400, 上传的图片保存在temp文件夹下, 文件名为上传的文件名
    """
    file = request.files['file']
    if file:
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)
        return 'Image uploaded successfully.', 200
    else:
        return 'No Image uploaded.', 400


# 提供图片的路由
@Interactor.route('/image/<filename>', methods=['GET'])
def get_image(filename):
    """
    :param filename: 图片的文件名
    :return: 图片

    提供图片的路由, 返回图片
    """
    image_path = os.path.join(UPLOAD_FOLDER, filename)
    if os.path.exists(image_path):
        image_path = os.path.abspath(image_path)
        return send_file(image_path, mimetype='image/jpeg')
    else:
        return 'Image not found.', 404


@Interactor.route('/imageList', methods=['GET'])
def get_image_list():
    """
    :return: 图片列表

    返回temp文件夹下的图片列表
    """
    image_list = os.listdir(UPLOAD_FOLDER)
    response = {
        'length': len(image_list),
        'image_list': image_list
    }
    return json.dumps(response)


@Interactor.route('/LoadModel', methods=['GET'])
def LoadModel():
    """
    :return: 加载模型的结果

    加载模型, 加载成功返回200, 加载失败返回500
    """
    global isTrained
    try:
        GT.load()
        isTrained = True
    except Exception as e:
        return str(e), 500
    return 'Model loaded successfully.', 200


@Interactor.route('/Camera', methods=['POST'])
def Camera():
    """
    :return: 摄像头的结果

    返回摄像头的结果, 包括是否训练, 数据集大小, 鼠标位置, 动作
    """
    global isTrained, mem
    ret, frame = Cam.read()
    x, y = GT.predict(frame)
    action,frame = recognize(frame)
    print(action)
    if action == '识别':
        mem = (x / pa.size()[0], y / pa.size()[1])
    _, buffer = cv.imencode('.jpg', frame)
    frame_bytes = buffer.tobytes()
    encoded = base64.b64encode(frame_bytes).decode('utf-8')
    response = {
        'isTrained': isTrained,
        'size': GT.Dataset.size,
        'position': {
            'x': x / pa.size()[0] * 100,
            'y': y / pa.size()[1] * 100
        },
        'action': action,
        'frame': encoded
    }
    return json.dumps(response), 200


@Interactor.route('/Train', methods=['POST'])
def Train():
    """
    :return: 训练的结果

    训练模型, 训练成功返回200, 训练失败返回400
    """
    global isTrained
    if isTrained:
        return 'Model has been trained.', 400
    ret, frame = Cam.read()
    GT.insert(frame, pa.position())
    if GT.Dataset.size == 20:
        GT.train()
        isTrained = True
    response = {
        'isTrained': isTrained,
        'size': GT.Dataset.size,
        'position': {
            'x': 0,
            'y': 0
        },
        'action': ''
    }
    return json.dumps(response), 200


@Interactor.route('/Identify', methods=['POST'])
def Identify():
    """
    :return: 保存图片的结果

    保存图片, 保存成功返回200, 保存失败返回500
    """
    imgBlob = request.get_data()
    imgBlob = json.loads(imgBlob)
    imgB64 = base64.b64decode(imgBlob['img'][23:])
    Position = {
        'x': mem[0],
        'y': mem[1]
    }
    img = cv.imdecode(np.frombuffer(imgB64, np.uint8), cv.IMREAD_COLOR)
    Position = (int(Position['x'] * img.shape[0])), int(Position['y'] * img.shape[1])

    # 截取一个矩形
    x, y = Position
    h, w = 200, 200
    u, d, l, r = max(0, x - h // 2), min(img.shape[0], x + h // 2), max(0, y - w // 2), min(img.shape[1], y + w // 2)
    img = img[u:d, l:r]
    try:
        cv.imwrite('temp/img.jpg', img)
        text = Gemini().generate(PIL.Image.fromarray(cv.cvtColor(img,cv.COLOR_BGR2RGB)), '这是什么?请用中文回答。')
        return text, 200
    except Exception as e:
        return str(e), 500
