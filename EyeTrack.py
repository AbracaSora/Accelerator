import cv2 as cv
import dlib
import matplotlib.pyplot as plt

# Load the detector
FaceDetector = dlib.get_frontal_face_detector()
EyeDetector = dlib.shape_predictor("model/shape_predictor_68_face_landmarks.dat")

img = cv.imread("local/img_1.png")

# Convert image into grayscale
imgGray = cv.cvtColor(src=img, code=cv.COLOR_BGR2GRAY)

Face = FaceDetector(imgGray)
Landmarks = EyeDetector(imgGray, Face[0])

eyeLeft = [Landmarks.parts()[i] for i in range(36, 42)]
eyeRight = [Landmarks.parts()[i] for i in range(42, 48)]

centerLeft = (
    sum([point.x for point in eyeLeft]) // 6,
    sum([point.y for point in eyeLeft]) // 6
)
centerRight = (
    sum([point.x for point in eyeRight]) // 6,
    sum([point.y for point in eyeRight]) // 6
)

pupilLeft = cv.minMaxLoc(
    imgGray[
        centerLeft[1] - 20:centerLeft[1] + 20,
        centerLeft[0] - 20:centerLeft[0] + 20
    ]
)[3]
pupilRight = cv.minMaxLoc(
    imgGray[
        centerRight[1] - 20:centerRight[1] + 20,
        centerRight[0] - 20:centerRight[0] + 20
    ]
)[3]

cv.circle(img, centerLeft, 5, (0, 255, 0), -1)
cv.circle(img, centerRight, 5, (0, 255, 0), -1)

cv.imshow(winname="Image", mat=img)
cv.waitKey(delay=0)
