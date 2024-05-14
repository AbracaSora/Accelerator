from flask import Flask, send_file, request
from flask_cors import CORS
from GazeTrackingAlter import GazeTracker
from HandTracking import *
import pyautogui as pa
import cv2 as cv
import json
import os

width, height = pa.size()
GT = GazeTracker()
isTrained = False
TimeWait = 20
Cam = cv.VideoCapture(0)

Interactor = Flask(__name__)
CORS(Interactor)
# 临时保存图片的文件夹
UPLOAD_FOLDER = 'temp'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


# 上传图片的路由
@Interactor.route('/upload', methods=['POST'])
def upload():
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
    image_path = os.path.join(UPLOAD_FOLDER, filename)
    if os.path.exists(image_path):
        image_path = os.path.abspath(image_path)
        return send_file(image_path, mimetype='image/jpeg')
    else:
        return 'Image not found.', 404


@Interactor.route('/imageList', methods=['GET'])
def get_image_list():
    image_list = os.listdir(UPLOAD_FOLDER)
    response = {
        'length': len(image_list),
        'image_list': image_list
    }
    return json.dumps(response)


@Interactor.route('/LoadModel', methods=['GET'])
def LoadModel():
    global isTrained
    try:
        GT.load()
        isTrained = True
    except Exception as e:
        return str(e), 500
    return 'Model loaded successfully.', 200


@Interactor.route('/Camera', methods=['POST'])
def Camera():
    global isTrained
    ret, frame = Cam.read()
    x, y = GT.predict(frame)
    action = recognize(frame)
    response = {
        'isTrained': isTrained,
        'size': GT.Dataset.size,
        'position': {
            'x': x,
            'y': y
        },
        'action': action
    }
    return json.dumps(response), 200


@Interactor.route('/Train', methods=['POST'])
def Train():
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
