from flask import Flask, send_file, request
from flask_cors import CORS
from GazeTrackingAlter import GazeTracker
from HandTracking import *

import base64
import pyautogui as pa
import numpy as np
import cv2 as cv
import json
import os

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
    global isTrained
    ret, frame = Cam.read()
    x, y = GT.predict(frame)
    action = recognize(frame)
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


@Interactor.route('/SaveImage', methods=['POST'])
def SaveImage():
    """
    :return: 保存图片的结果

    保存图片, 保存成功返回200, 保存失败返回500
    """
    imgBlob = request.get_data()
    imgBlob = json.loads(imgBlob)
    imgB64 = base64.b64decode(imgBlob['img'][23:])
    img = cv.imdecode(np.frombuffer(imgB64, np.uint8), cv.IMREAD_COLOR)
    print(img)
    try:
        cv.imwrite('temp/img.jpg', img)
    except Exception as e:
        print(e)
        return str(e), 500
    return 'Image saved successfully.', 200
