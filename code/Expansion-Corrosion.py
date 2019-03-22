'''
使用检测模板实现图像的膨胀与腐蚀
'''
import cv2 as cv
import numpy as np


#获取图像的二值图像
def getBinaryImg(img_path):
    '''
    :param img_path:
    :return:
    '''
    raw_img = cv.imread(img_path)
    gray_img = cv.cvtColor(raw_img, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(gray_img, 127, 255, cv.THRESH_BINARY_INV)

    return thresh

#图像腐蚀算法
def imgCorrode(img):

    '''
    :param img: numpy数组 二值图图像
    :return: 二值图像
    '''
    #腐蚀算子,大小为奇数维度
    #filter = np.array([[0,255,0],[255,255,255],[0,255,0]])
    filter = np.array([[255, 255, 255], [255, 255, 255], [255, 255, 255]])
    filter_w,filter_h = filter.shape
    new_img = img.copy()

    for i in range(int(filter_w/2),len(img)+1-filter_w):
        for j in range(int(filter_h/2),len(img[i])+1-filter_h):
            if((img[i:i + filter_w:, j:j + filter_h:] == filter).all()):
                new_img[i][j] = 255
            else:
                new_img[i][j] = 0

    return new_img



#图像膨胀算法
def imgExpand(img):
    '''
    :param img:
    :return:
    '''
    # 膨胀算子,大小为奇数维度
    # filter = np.array([[0,255,0],[255,255,255],[0,255,0]])
    filter = np.array([[255, 255, 255], [255, 255, 255], [255, 255, 255]])
    filter_w, filter_h = filter.shape
    new_img = img.copy()

    for i in range(int(filter_w / 2), len(img) + 1 - filter_w):
        for j in range(int(filter_h / 2), len(img[i]) + 1 - filter_h):

            flag = False
            for n  in range(len(filter)):
                for m in range(len(filter[n])):
                    if(filter[n][m]==img[i-int(filter_w/2+n)][j-int(filter_h/2+m)]):
                        flag = True
                        break

            if flag==True:
                new_img[i][j] = 255
            else:
                new_img[i][j] = 0
    return new_img



if __name__ =="__main__":
    raw_img = cv.imread("../images/EE.jpg")
    gray_img = cv.cvtColor(raw_img, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(gray_img, 100, 255, cv.THRESH_BINARY)



    print(thresh)
    cv.imshow("ray", thresh)
    thresh = imgCorrode(thresh)
    print(thresh)
    cv.imshow("expand", thresh)
    thresh = imgExpand(thresh)
    cv.imshow("Corrode", thresh)
    cv.waitKey(0)