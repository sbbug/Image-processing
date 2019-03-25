import cv2 as cv
import numpy as np
import ExpansionCorrosion as EC
import matplotlib.pyplot as plt

#获取图像的二值图
def getBinaryImg(file_path):
    raw_img = cv.imread(file_path)
    gray_img = cv.cvtColor(raw_img, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(gray_img, 100, 255, cv.THRESH_BINARY)
    res = EC.imgCorrode(thresh,6,6)
    res = EC.imgExpand(res,6,6)
    return res

#将图像进行投影压缩
def getSplitLocation(img_bin):

    img_h,img_w = img_bin.shape
    img_vec = np.zeros(img_w,np.int32)

    for i in range(img_w):
        col = img_bin[:,i]
        for j in range(img_h):
            if col[j]==255:
                img_vec[i]=img_vec[i]+1

    print(img_vec)
    x = range(0,img_w)
    #绘制每一列的白色像素统计值
    plt.plot(x, img_vec, linewidth=2)
    plt.show()


if __name__ =="__main__":

    img_bin = getBinaryImg("../images/numbers.png")
    getSplitLocation(img_bin)
    cv.imshow("bin",img_bin)
    # for i in range(len(img_bin)):
    #     for j in range(len(img_bin[i])):
    #         print(img_bin[i][j],end=" ")
    #     print("")
    cv.waitKey(0)