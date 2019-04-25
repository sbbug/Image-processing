import cv2 as cv
import numpy as np
import ExpansionCorrosion as EC
import matplotlib.pyplot as plt

#获取图像的二值图
def getBinaryImg(file_path):
    '''
    :param file_path: 图像路径
    :return: 返回处理后二值图
    '''
    raw_img = cv.imread(file_path)
    gray_img = cv.cvtColor(raw_img, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(gray_img, 100, 255, cv.THRESH_BINARY)

    res = EC.imgCorrode(thresh,6,6)
    res = EC.imgExpand(res,6,6)
    return res,thresh

#将图像进行投影压缩
def getSplitLocation(img_bin,thre):
    '''
    :param img_bin: 图像二值图
    :param thre: 阈值，大于该阈值则为数字区域
    :return:
    '''
    img_h,img_w = img_bin.shape
    img_vec = np.zeros(img_w,np.int32)
    split_loc = []

    for i in range(img_w):
        col = img_bin[:,i]
        for j in range(img_h):
            if col[j]==255:
                img_vec[i]=img_vec[i]+1

    #寻找分割点
    flag = True
    start = -1
    end = -1
    for i in range(len(img_vec)):
        if img_vec[i]>=thre and start==-1:
            start=i
            flag = False
        elif img_vec[i]<thre and flag==False:
            end = i
            split_loc.append([start,end])
            flag = True
            start = -1
            end = -1
        else:
            continue


    x = range(0,img_w)
    #绘制每一列的白色像素统计值
    plt.plot(x, img_vec, linewidth=2)
    plt.show()

    return split_loc


if __name__ =="__main__":

    raw = cv.imread("../images/numbers.png")
    img_bin,r = getBinaryImg("../images/numbers.png")
    img_h,img_w = img_bin.shape
    split_locations = getSplitLocation(img_bin,20)
    print(split_locations)
    for i in range(len(split_locations)):
        point = split_locations[i]
        print(point)
        # cv.circle(raw,(point[0],0),3,(0,0,255),0)
        # cv.circle(raw,(point[1],0), 3, (0, 0, 255),0)
        # #img_bin[0:point[1]-point[0],point[0]:img_h]
        data = img_bin[0:img_h,point[0]:point[1]]
        # print(data.shape)
       # data = cv.resize(data,(40,40))
        cv.imwrite(str(i)+".jpg",data)
    cv.imshow("raw", r)
    cv.imshow("bin",img_bin)
    cv.imshow("res", raw)
    # for i in range(len(img_bin)):
    #     for j in range(len(img_bin[i])):
    #         print(img_bin[i][j],end=" ")
    #     print("")
    cv.waitKey(0)