import cv2
import numpy as np
import HandTrackingModule as htm
import time
import pyautogui as ui

#######################################
wCam, hCam = 1280, 720
frameR = 100       # 暂定
smoothening = 3    # 暂定
#######################################

pTime = 0
plocX, plocY = 0, 0
clocX, clocY = 0, 0

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

detector = htm.handDetector(maxHands=1)
wScr, hScr = ui.size()
# print(wScr, hScr)

while True:
    # 找手
    success, img = cap.read()
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img)

    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]
        # print(x1, y1, x2, y2)

        fingers = detector.fingersUp()            # 检测手指向上与否
        # print(fingers)
        cv2.rectangle(img, (frameR, 0), (wCam - frameR, hCam - 2 * frameR),
                      (255, 0, 255), 2)

        if fingers[1] and fingers[2] == 0:
            x3 = np.interp(x1, (frameR, wCam-frameR), (0, wScr))
            y3 = np.interp(y1, (0, hCam-2*frameR), (0, hScr))
            # 平滑处理
            clocX = plocX + (x3-plocX)/smoothening
            clocY = plocY + (y3 - plocY) / smoothening

            # 连接鼠标
            ui.moveTo(wScr-clocX, clocY)
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            plocX, plocY = clocX, clocY

        if fingers[1] and fingers[2] == 1:            # 点击模式
            length, img, lineInfo = detector.findDistance(8, 12, img)
            # print(length)
            if length < 40:
                cv2.circle(img, (lineInfo[4], lineInfo[5]), 15, (0, 255, 0), cv2.FILLED)
                # time.sleep(0.5)
                ui.leftClick()


    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, f'FPS:{int(fps)}', (20, 50), cv2.FONT_HERSHEY_COMPLEX,
                1, (255, 0, 255), 2)

    cv2.imshow("Image", img)
    cv2.waitKey(1)