from flask import Flask, send_file, request
from flask_cors import CORS
from flask_socketio import SocketIO
import multiprocessing
import cv2 as cv
import json
import os

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
    # print(request.files)
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
