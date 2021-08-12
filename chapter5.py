import cv2 as cv
import numpy  as np

img = cv.imread("Resources/cards.jpg")
width, height = 250, 350
# 图像中牌的位置
pts1 = np.float32([[111, 218], [287, 188], [154, 482], [352, 440]])
# 要移动到的位置
pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
# 得到转换矩阵
matrix = cv.getPerspectiveTransform(pts1, pts2)
# 把使用转换矩阵把矩阵转换,后边俩参数是剪切的宽高
# 旋转完不剪切,原始大小
imgOutput = cv.warpPerspective(img, matrix, img.shape[0:2])
# 把黑桃❤剪切出来
imgOutput_black_heart = cv.warpPerspective(img, matrix, [width, height])

cv.imshow("Image", img)
cv.imshow("Output", imgOutput)
cv.imshow("Output_black_heart", imgOutput_black_heart)
cv.waitKey(0)
