from flask import Flask, send_file, request
from flask_cors import CORS
from flask_socketio import SocketIO
from GazeTrackingAlter import GazeTracker
import base64
import pyautogui as pa
import pygame as pg
import numpy as np
import cv2 as cv
import json
import os

width, height = pa.size()
GT = GazeTracker()
isTrained = False
# screen = pg.display.set_mode((width, height))
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
        return 'Image uploaded successfully.'


# 提供图片的路由
@Interactor.route('/image/<filename>', methods=['GET'])
def get_image(filename):
    image_path = os.path.join(UPLOAD_FOLDER, filename)
    if os.path.exists(image_path):
        image_path = os.path.abspath(image_path)
        return send_file(image_path, mimetype='image/jpeg')
    else:
        return 'Image not found.'


@Interactor.route('/imageList', methods=['GET'])
def get_image_list():
    image_list = os.listdir(UPLOAD_FOLDER)
    response = {
        'length': len(image_list),
        'image_list': image_list
    }
    return json.dumps(response)


@Interactor.route('/Camera', methods=['POST'])
def Camera():
    global isTrained
    flag = request.json['isDataset']
    ret, frame = Cam.read()
    response = {
        'isTrained': isTrained,
        'size': 0,
        'position': {
            'x': 0,
            'y': 0
        },
        'action': ''
    }
    if flag and not isTrained:
        x, y = pa.position()
        GT.insert(frame, (x, y))
        response['size'] = GT.Dataset.size
        if GT.Dataset.size == 20:
            GT.train()
            isTrained = True
    elif isTrained:
        x, y = GT.predict(frame)
        x = (x + 1) / 2 * width
        y = (y + 1) / 2 * height
        x, y = map(int, (x, y))
        response['size'] = GT.Dataset.size
        response['position']['x'] = x
        response['position']['y'] = y
    return json.dumps(response), 200
