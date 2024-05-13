import pickle
import cv2
import mediapipe as mp
import HandTrackingModule as htm
import numpy as np

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

######################
wCam, hCam = 640, 480
######################


vid = cv2.VideoCapture(0)
vid.set(3, wCam)
vid.set(4, hCam)
detector = htm.handDetector(detectionCon=0.7)


labels_dict= {
    'a': 'A', 'b': 'B', 'c': 'C', 'd': 'D', 'e': 'E', 'f': 'F', 'g': 'G', 'h': 'H', 'i': 'I', 'j': 'J',
    'k': 'K', 'l': 'L', 'm': 'M', 'n': 'N', 'o': 'O', 'p': 'P', 'q': 'Q', 'r': 'R', 's': 'S', 't': 'T',
    'u': 'U', 'v': 'V', 'w': 'W', 'x': 'X', 'y': 'Y', 'z': 'Z',
}

while True:
    ret, img = vid.read()
    H, W, _ = img.shape
    img = detector.findHands(img)

    cordinates= detector.findpos(img)
    x,y = detector.box_around_hand(img)

    if cordinates != []:
        prediction = model.predict([np.asarray(cordinates)])
        predicted_character = labels_dict[prediction[0]]

        x1 = int(min(x) * W) - 10
        y1 = int(min(y) * H) - 10

        x2 = int(max(x) * W) - 10
        y2 = int(max(y) * H) - 10
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(img, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                    cv2.LINE_AA)

    cv2.imshow('Image', img)
    k = cv2.waitKeyEx(1)
    if k == 27:
        break


vid.release()
cv2.destroyAllWindows()