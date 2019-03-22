'''
求取图像的梯度，检测图像轮廓
'''

import numpy as np
import cv2 as cv
import math
#获取图像的灰度矩阵
def getMatrix(img_path):

    raw_img = cv.imread(img_path)
    gray_img = cv.cvtColor(raw_img,cv.COLOR_BGR2GRAY)
    return gray_img

#实现图像的卷积操作
def imgConvoluting(image,filter):

    w, h = filter.shape
    con_img = image.copy()
    filter_w, filter_h = filter.shape
    for i in range(1, len(image) + 1 - w):
        for j in range(1, len(image[i]) + 1 - h):
            con_img[i][j] = (image[i:i + filter_w:, j:j + filter_h:] * filter).sum()

    return con_img

#使用二阶微分实现检测，实现图像的锐化
def twoDifference(image,c):

   raw_img = image.copy()
   res_img = image.copy()

   filter =  np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])

   filter_img = imgConvoluting(image,filter)

   for i in range(len(image)):
       for j in range(len(image[i])):
           res_img[i][j] = raw_img[i][j]+c*filter_img[i][j]

   return res_img
#sobel算子检测
def imgSobel(image):
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
            #con_img[i][j] = np.sqrt(np.power(img_g_x[i][j],2)+np.power(img_g_y[i][j],2))

            #将检测到的轮廓点阈值化
            if con_img[i][j]>100:
                image[i][j] = 255
            else:
                image[i][j] = 0

    return image



if __name__=="__main__":

    gray_img = getMatrix("../images/D.jpg")
    cv.imwrite("../images/gray_img.png", gray_img)
    res_img = imgSobel(gray_img)
    cv.imwrite("../images/res.png",res_img)
    print(res_img)
    np.savetxt("../images/im.txt",np.round(res_img,1))



