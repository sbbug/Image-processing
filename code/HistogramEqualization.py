import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
font = FontProperties(fname=r"C:\Windows\Fonts\simhei.ttf", size=14)

debug = True
# 获取图像中灰度图出现的频率向量
def getVector(img):
    '''
    :param img:
    :return:
    '''
    vector = np.zeros((1,256))
    h,w = img.shape
    for i in range(h):
        for j in range(w):
            vector[0][img[i][j]]+=1
    return np.round(vector[0]/sum(vector[0]),4) #归一化

# 调用numpy实现直方图均衡化
def equalImg(img):
    '''
    :param img:
    :return:
    '''
    # 获取直方图信息 hist是灰度值出现的频数 bins是[0-256]
    hist, bins = np.histogram(img.flatten(), 256, [0, 256])
    # 返回沿给定轴的元素的累积和
    '''
    a = np.array([1,2,3,4])
    b = np.cumsum(a)
    [1 2 3 4]
    [ 1  3  6 10]
    '''
    accu = hist.cumsum()
    # 对数组中指定的元素值进行屏蔽
    accu_m = np.ma.masked_equal(accu, 0)
    # 做归一化
    accu_m = (accu_m - accu_m.min()) * 255 / (accu_m.max() - accu_m.min())
    # 将屏蔽掉的元素值进行填充,获取了旧的灰度值对应的新的灰度值
    accu = np.ma.filled(accu_m, 0).astype('uint8')
    # 根据accu里的数据，将img里的数据进行替换

    if debug:
       plt.plot(hist)
       plt.show()

    # 方式1：
    img2 = img.copy()
    for i in range(len(img)):
        for j in  range(len(img[i])):
            img2[i][j] = accu[img[i][j]]

    # 方式2:
    #img2 = accu[img]
    hist, bins = np.histogram(img2.flatten(), 256, [0, 256])
    if debug:
       plt.plot(hist)
       plt.show()
    return img2

if __name__=="__main__":

    raw_img = cv2.imread("../images/HE/test.png")
    raw_img = cv2.cvtColor(raw_img,cv2.COLOR_BGR2GRAY)
    img = equalImg(raw_img)
    # hist, bins = np.histogram(img.flatten(), 256, [0, 256])
    # print(hist)
    # cdf = hist.cumsum()
    # print(cdf)
    # cdf_normalized = cdf * hist.max() / cdf.max()
    # cdf_m = np.ma.masked_equal(cdf, 0)
    # cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    # cdf = np.ma.filled(cdf_m, 0).astype('uint8')
    # # vector = getVector(img)
    # # print(vector)
    # img2 = cdf[img]
    # plt.plot(cdf_normalized)
    # plt.show()
    cv2.imshow("raw",raw_img)
    cv2.imshow("gray",img)
    cv2.waitKey(-1)