from Interactor import socketio, Interactor
# import cv2 as cv
# import pyautogui as pa
# import keyboard as kb
# import pygame as pg
#
# from GazeTrackingAlter.GazeTracking import GazeTracker
#
# pg.init()
#
#
# def Midpoint(p1, p2):
#     return (p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2
#
#
# def GazeTracking():
#     width, height = pa.size()
#     Camera = cv.VideoCapture(0)
#     GT = GazeTracker()
#     trained = False
#     screen = pg.display.set_mode((width, height))
#     TimeWait = 20
#     while True:
#         ret, frame = Camera.read()
#         x, y = pa.position()
#         if kb.is_pressed('space') and TimeWait >= 20:
#             GT.insert(frame, (x, y))
#             TimeWait = 0
#         if GT.Dataset.size == 20 and not trained:
#             GT.train()
#             trained = True
#         if trained:
#             x, y = GT.predict(frame)
#             print()
#             x = (x + 1) / 2 * width
#             y = (y + 1) / 2 * height
#             x, y = map(int, (x, y))
#             socketio.send({'x': x, 'y': y})
#             pg.draw.circle(screen, (0, 255, 0), (x, y), 10)
#             pg.display.update()
#         TimeWait += 1
#         if cv.waitKey(1) & 0xFF == ord('q'):
#             break
#
#
# First_Connect = True
#
#
# @socketio.on('connect')
# def connect():
#     global First_Connect
#     if First_Connect:
#         print('Connected.')
#         First_Connect = False
#         socketio.start_background_task(GazeTracking)


if __name__ == '__main__':
    socketio.run(Interactor, debug=True, allow_unsafe_werkzeug=True)
