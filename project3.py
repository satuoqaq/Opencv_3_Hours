import os

import cv2 as cv

##################
frameWidth = 640
frameHeight = 480
nPlateCascade = cv.CascadeClassifier('Resources/haarcascade_russian_plate_number.xml')
###################

FileList = os.listdir('Resources/Car')
cnt = 0
for name in FileList:
    img = cv.imread('Resources/Car/' + name)
    imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    numberPlate = nPlateCascade.detectMultiScale(imgGray, 1.1, 4)
    for (x, y, w, h) in numberPlate:
        area = w * h
        if area > 500:
            cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv.putText(img, "Number Plate", (x, y - 5),
                       cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
            imgRoi = img[y:y + h, x:x + w]
            cv.imshow("ROI", imgRoi)
    cv.imshow("Result", img)
    cv.waitKey(0)
    # 存起来
    cv.imwrite("Resources/Scanned/NoPlate_" + str(cnt) + ".jpg", imgRoi)
    cnt = cnt + 1
