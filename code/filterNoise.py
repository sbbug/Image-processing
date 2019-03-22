'''
实现图像的增强，并且消除噪声点
'''

import cv2 as cv
import sharpen as f
import numpy as np
import math

#实现图像的卷积操作
def imgConvoluting(image,filter):

    w, h = filter.shape
    con_img = image.copy()
    filter_w, filter_h = filter.shape
    for i in range(1, len(image) + 1 - w):
        for j in range(1, len(image[i]) + 1 - h):
            con_img[i][j] = (image[i:i + filter_w:, j:j + filter_h:] * filter).sum()

    return con_img

def imgGradient(image):
    '''
    :param image: 灰度图
    :return:
    '''

    #高斯降噪
    gass = np.array([[1/16,1/8,1/16],[1/8,1/4,1/8],[1/16,1/8,1/16]])

    img_after_gass = imgConvoluting(image,gass)

    #梯度算子
    g_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    g_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    con_img = image.copy()

    img_g_x = imgConvoluting(img_after_gass, g_x)
    img_g_y = imgConvoluting(img_after_gass, g_y)

    for i in range(len(con_img)):
        for j in range(len(con_img[i])):
            con_img[i][j] = math.fabs(img_g_x[i][j])+math.fabs(img_g_y[i][j])

    return con_img

def getMatrix(img_path):

    raw_img = cv.imread(img_path)
    gray_img = cv.cvtColor(raw_img,cv.COLOR_BGR2GRAY)

    return raw_img,gray_img

def show(m):
    for i in range(len(m)):
        for j in range(len(m[i])):
            print(m[i][j],end="|")
        print("")

if __name__ =="__main__":

    cv.namedWindow("gray", 0);
    raw_img,gray_img = getMatrix("../images/W.png")

    cv.imwrite("gray.png", gray_img)

    # b,g,r = cv.split(raw_img)
    #
    # print(b)
    # print(g)
    # print(r)
    # show(b)
    # show(g)
    # show(r)
    # img_grad = imgGradient(gray_img)
    #
    for i in range(len(gray_img)):
        for j in range(len(gray_img[i])):
            print(gray_img[i][j],end="|")
            if gray_img[i][j]<53:
               print("--------")
               cv.circle(raw_img,(j,i),1,(255,0,0),-1)
            else:
               gray_img[i][j] = 0
        print("")
    # # img = BrightnessContrastFilter(img)
    # # img = BilateralBlur(img)
    # # img = AutoBW(img)
    #
    #
    #
    # # cv.imshow("gray", img)
    cv.imwrite("raw.png",raw_img)
    # cv.imwrite("binary.png", gray_img)
    # img = imgGradient(img)
    #
    # print(img)
    # #cv.resizeWindow("enhanced", 640, 480);
    # cv.namedWindow("binary", 0);
    # cv.imshow("binary", img)
    # cv.waitKey(0)


