import cv2
import mediapipe as mp
import time


class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, modelComplexity=1, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.modelComplex = modelComplexity
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelComplex, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findpos(self, image, handNO=0, draw=True):
        lmlist = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNO]
            for lm in myHand.landmark:
                # h, w, c = image.shape
                # cx, cy, = lm.x * w, lm.y * h
                # print(lm.x, lm.y)
                lmlist.append(lm.x)
                lmlist.append(lm.y)

        return lmlist

    def box_around_hand(self, image, handNO=0, draw=True):
        x = []
        y = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNO]
            for lm in myHand.landmark:
                x.append(lm.x)
                y.append(lm.y)
        return x, y