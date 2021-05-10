"""
This is a testing file to test the mediapipe library.
"""

import cv2
import mediapipe as mp
import time

# capturing video from webcam
cap = cv2.VideoCapture(0)

"""
Mediapipe detects the hand landmarks and assign them a number.
We are tracking this hand landmarks.
"""
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils  # Drawing utilities provided by mediapipe.

prevTime, currTime = 0, 0

while True:
    success, img = cap.read()
    # Image format must be converted to RGB for mediapipe to process
    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_RGB)

    # print(results.multi_hand_landmarks) # if you want to view the landmarks locations

    if results.multi_hand_landmarks:
        for handLM in results.multi_hand_landmarks:
            for id, lm in enumerate(handLM.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                # print(id, cx, cy)
                if id == 8:
                    cv2.circle(img, (cx, cy), 15, (6, 128, 255), cv2.FILLED)
            mpDraw.draw_landmarks(img, handLM, mpHands.HAND_CONNECTIONS)

    # Calculates the fps
    currTime = time.time()
    fps = 1 / (currTime - prevTime)
    prevTime = currTime

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (227, 189, 73), 3)

    cv2.imshow("Hand Tracker", img)
    cv2.waitKey(1)
