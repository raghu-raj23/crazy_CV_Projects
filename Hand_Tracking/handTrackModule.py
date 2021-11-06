import cv2
import mediapipe as mp
import time


class HandDetector:

    def __init__(self, mode=False, maxHands=2, detConf=0.5, trackConf=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detConf = detConf
        self.trackConf = trackConf
        self.complexity = 1

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,self.complexity, self.detConf, self.trackConf)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIDs = [4, 8, 12, 16, 20]

    def findHands(self, img, draw=True):
        # this function detects and return the hand landmarks
        img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_RGB)

        if self.results.multi_hand_landmarks:
            for handLM in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLM, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPositions(self, img, handNum=0, draw=True):
        # This function is used to locate the required hand landmark on the canvas
        self.lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNum]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 10, (6, 128, 255), cv2.FILLED)
        return self.lmList

    def fingersUp(self):
        #  This function is used to identify which fingers are raised
        fingers = []
        if self.lmList[self.tipIDs[0]][1] < self.lmList[self.tipIDs[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        for id in range(1, 5):
            if self.lmList[self.tipIDs[id]][2] < self.lmList[self.tipIDs[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers


def main():
    prevTime, currTime = 0, 0
    cap = cv2.VideoCapture(0)
    detector = HandDetector()

    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lms = detector.findPositions(img)
        if len(lms):
            print(lms[4])
        #  Following is done for finding the fps
        currTime = time.time()
        fps = 1 / (currTime - prevTime)
        prevTime = currTime
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (227, 189, 73), 3)
        cv2.imshow("Hand Tracke", img)
        cv2.waitKey(1)


if __name__ == '__main__':
    main()


"""

Sample format to use HT module

import handTrackModule as htm
import cv2, time

prevTime, currTime = 0, 0
cap = cv2.VideoCapture(0)
detector = htm.HandDetector()

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lms = detector.findPositions(img)
    if len(lms):
        print(lms[8])
    currTime = time.time()
    fps = 1 / (currTime - prevTime)
    prevTime = currTime
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (125, 125, 125), 3)
    cv2.imshow("Image", img)
    cv2.waitKey(1)
    
"""
