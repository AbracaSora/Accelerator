import cv2 as cv
import dlib

# Load the detector
FaceDetector = dlib.get_frontal_face_detector()
EyeDetector = dlib.cnn_face_detection_model_v1("shape_predictor_68_face_landmarks.dat")