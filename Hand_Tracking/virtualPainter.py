import cv2
import numpy as np
import os
import handTrackModule as htm

imgFPath = 'header'
imgList = os.listdir(imgFPath)
overlayList = []

for impath in imgList:
    im = cv2.imread(f'{imgFPath}/{impath}')
    overlayList.append(im)

detector = htm.HandDetector(detConf=0.7)
header = overlayList[0]

# Customising the colour and brush size
drawColor = (89, 75, 233)
brush = 15
eraser = 50

xp, yp = 0, 0
imgCanvas = np.zeros((480, 640, 3), np.uint8)

cap = cv2.VideoCapture(0)

# my camera max is 640 x 480
cap.set(3, 640)
cap.set(4, 480)

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img = detector.findHands(img, draw=True)
    lmList = detector.findPositions(img, draw=False)

    if len(lmList):
        xi, yi = lmList[8][1:]  # Location of index finger
        xm, ym = lmList[12][1:]  # Location of middle finger
        fingers = detector.fingersUp()

        if fingers[1] and fingers[2]:
            xp, yp = 0, 0  # Previous location
            print('Selection Mode')

            """
            range values for the paint locations for my case
            change values accordingly
            
            1 90 - 195
            2 196 - 310
            3 311 - 410
            4 411 - 515
            E 516 - 639
            """

            if yi < 80:  # The range is for the height of the header image
                if 90 < xi < 195:
                    header = overlayList[1]
                    drawColor = (89, 75, 233)
                elif 196 < xi < 310:
                    header = overlayList[4]
                    drawColor = (77, 145, 255)
                elif 311 < xi < 410:
                    header = overlayList[0]
                    drawColor = (173, 74, 0)
                elif 411 < xi < 515:
                    header = overlayList[2]
                    drawColor = (55, 128, 0)
                elif 516 < xi < 639:
                    header = overlayList[3]
                    drawColor = (0, 0, 0)

            cv2.rectangle(img, (xi, yi - 25), (xm, ym + 25), drawColor, cv2.FILLED)

        elif fingers[1] and not fingers[2]:
            cv2.circle(img, (xi, yi), 15, (255, 0, 0), cv2.FILLED)
            print('Drawing Mode')

            if xp == yp == 0:
                xp, yp = xi, yi

            if drawColor == (0, 0, 0):
                cv2.line(img, (xp, yp), (xi, yi), drawColor, eraser)
                cv2.line(imgCanvas, (xp, yp), (xi, yi), drawColor, eraser)
            else:
                cv2.line(img, (xp, yp), (xi, yi), drawColor, brush)
                cv2.line(imgCanvas, (xp, yp), (xi, yi), drawColor, brush)

            xp, yp = xi, yi

    # Merging both the canvas and camera image into one output
    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)  # Converted the canvas to gray
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    # created a threshold to get the inverted image of the binary converted canvas output
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)  # converting the inverted canvas image back to BGR
    img = cv2.bitwise_and(img, imgInv)  # this will perform a bitwise and over both the image and canvas
    # Now the output will have a black sketch according to our drawing
    img = cv2.bitwise_or(img, imgCanvas)
    # This will fill the black spaces(of the sketch) with the original color selected to draw

    img[0:80][0:640] = header  # We are slicing the image and adding the header
    # img = cv2.addWeighted(img, 0.5, imgCanvas, 0.5, 0)

    cv2.imshow("Image", img)
    # cv2.imshow("ImageCanvas", imgCanvas)
    cv2.waitKey(1)
