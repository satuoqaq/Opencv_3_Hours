import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
cap.set(10, 150)

widthImg = 480
heightImg = 640


def stackImages(scale, imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                    imgArray[x][y] = cv.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None,
                                               scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y] = cv.cvtColor(imgArray[x][y], cv.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank] * rows
        hor_con = [imageBlank] * rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None, scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv.cvtColor(imgArray[x], cv.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        ver = hor
    return ver


def preProcessing(img):
    imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    imgBlur = cv.GaussianBlur(imgGray, (5, 5), 1)
    imgCanny = cv.Canny(imgBlur, 100, 100)
    kernel = np.ones((5, 5))
    # 膨胀  腐蚀
    imgDial = cv.dilate(imgCanny, kernel, iterations=2)
    imgThres = cv.erode(imgDial, kernel, iterations=1)
    return imgThres


def getContours(img):
    contours, hierarchy = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    biggest = np.array([])
    maxArea = 0
    for cnt in contours:
        area = cv.contourArea(cnt)
        # print(area)
        if area > 500:
            # print(area)
            cv.drawContours(imgContours, cnt, -1, (0, 255, 0), 3)
            peri = cv.arcLength(cnt, True)  # 轮廓边长
            approx = cv.approxPolyDP(cnt, 0.02 * peri, True)  # 轮廓的角点
            if area > maxArea and len(approx) == 4:
                biggest = approx
                maxArea = max(area, maxArea)
            x, y, w, h = cv.boundingRect(approx)
    cv.drawContours(imgContours, biggest, -1, (255, 0, 0), 10)
    return biggest


def reorder(myPoints):
    # 四个角点排序,整的挺麻烦
    myPoints = myPoints.reshape((4, 2))
    myPointsNew = np.zeros((4, 1, 2), np.int32)
    add = myPoints.sum(1)
    # print("add", add)
    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] = myPoints[np.argmax(add)]
    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] = myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]
    # print("newPoints", myPointsNew)
    return myPointsNew


def getWarp(img, biggest):
    # 2d旋转平移矩阵算一个,然后移动过去剪切
    biggest = reorder(biggest)
    pts1 = np.float32(biggest)
    pts2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])
    matrix = cv.getPerspectiveTransform(pts1, pts2)
    imgOutput = cv.warpPerspective(img, matrix, (widthImg, heightImg))
    return imgOutput


while True:
    success, img = cap.read()
    img = cv.resize(img, (widthImg, heightImg))
    imgContours = img.copy()
    imgThres = preProcessing(img)
    biggest = getContours(imgThres)
    if biggest.size != 0:
        imgWarp = getWarp(img, biggest)
        StackImages = stackImages(0.6, [[img, imgThres], [imgContours, imgWarp]])
        cv.imshow("stackImage", StackImages)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
