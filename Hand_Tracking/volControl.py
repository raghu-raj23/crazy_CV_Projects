from subprocess import call

import cv2
import math
import numpy as np
import time

import handTrackModule as htm


# Function to adjust Master volume in Linux
def setVol(volume):
    valid = False
    while not valid:
        try:
            if (volume <= 100) and (volume >= 0):
                call(["amixer", "-D", "pulse", "sset", "Master", str(volume) + "%"])
                valid = True
        except ValueError:
            pass


# Resolution of my camera
wCam, hCam = 640, 480

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
currTime, prevTime = 0, 0
detector = htm.HandDetector(detConf=0.7)

# Volume range for my pc is Limits: Capture 0 - 65536(Linux)
# hand detecting area range 20 - 130

vmin, vmax = 0, 65536
hmin, hmax = 20, 130
vol, barVol, perVol = 0, 402, 0

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPositions(img, draw=False)

    if len(lmList):
        # location for thumb and index finger
        x1, y1, x2, y2 = lmList[4][1], lmList[4][2], lmList[8][1], lmList[8][2]
        mx, my = (x1 + x2) // 2, (y1 + y2) // 2

        cv2.circle(img, (x1, y1), 7, (6, 128, 255), cv2.FILLED)
        cv2.circle(img, (x2, y2), 7, (6, 128, 255), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (255, 125, 0), 3)
        cv2.circle(img, (mx, my), 7, (126, 141, 47), cv2.FILLED)

        length = math.hypot(x2 - x1, y2 - y1)
        vol = np.interp(length, [hmin, hmax], [vmin, vmax])
        vol = math.floor((vol / vmax) * 100)
        barVol = np.interp(length, [hmin, hmax], [400, 150])
        perVol = np.interp(length, [hmin, hmax], [0, 100])
        setVol(vol)

        if length < 20:
            cv2.circle(img, (mx, my), 7, (85, 85, 252), cv2.FILLED)

    #  Creating a volume bar
    cv2.rectangle(img, (50, 150), (85, 400), (37, 25, 22), 3)
    cv2.rectangle(img, (50, int(barVol)), (85, 400), (112, 108, 103), cv2.FILLED)
    cv2.putText(img, f'{int(perVol)}%', (45, 140), cv2.FONT_HERSHEY_SIMPLEX, 1, (51, 99, 255), 2)

    # For fps
    currTime = time.time()
    fps = 1 / (currTime - prevTime)
    prevTime = currTime

    cv2.putText(img, f'FPS:{int(fps)}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (227, 189, 73), 2)
    cv2.imshow("Volume Gesture Controller", img)
    cv2.waitKey(1)

"""
For windows users:
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
print(volume.GetVolumeRange())
volume.SetMasterVolumeLevel(-20.0, None)
"""
