import cv2
import HandTrackingModule as htm
import pickle
import json

######################
wCam, hCam = 640, 480
######################

vid = cv2.VideoCapture(0)
vid.set(3, wCam)
vid.set(4, hCam)
detector = htm.handDetector(detectionCon=0.7)
counter = 0

try:
    with open('savedata.json', 'r') as f:
        existing_data = json.load(f)
except FileNotFoundError:
    existing_data = {'data': [], 'label': []}

length_dataset = len(existing_data['data'])

while True:
    ret, img = vid.read()
    img = detector.findHands(img)
    cv2.imshow('Image', img)

    key = cv2.waitKeyEx(1)

    if key != -1:
        if key == 13: #enter key
            with open('savedata.json', 'w', encoding='utf-8') as f:
                json.dump(existing_data, f)
                print(len(existing_data['data']), len(existing_data['label']))
                break
        elif key == 8:      # removing last element from dataset #backspacekey
            existing_data['data'].pop()
            existing_data['label'].pop()
            print(length_dataset)
            length_dataset -= 1

        elif key == 27:  #esc key
            break

        key_char = chr(key)
        lmlist  = detector.findpos(img)
        if lmlist != []:
            existing_data['data'].append(lmlist)
            existing_data['label'].append(key_char)

            print(counter)
            counter += 1

vid.release()
cv2.destroyAllWindows()