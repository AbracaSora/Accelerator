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
trained = False
# screen = pg.display.set_mode((width, height))
TimeWait = 20

Interactor = Flask(__name__)
CORS(Interactor)
socketio = SocketIO(Interactor, cors_allowed_origins="*", protocol_version=3)
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
    file = request.json['image']
    if not file:
        return 'No Image Uploaded.', 400
    file = base64.b64decode(file[22:])
    file_array = np.frombuffer(file, np.uint8)
    frame = cv.imdecode(file_array, cv.IMREAD_COLOR)
    cv.imshow("Camera", frame)
    cv.imwrite("temp/Camera.jpg", frame)
    return 'Image Received.', 200
