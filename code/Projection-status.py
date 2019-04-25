import cv2 as cv
import numpy as np

#type transform
def getMatInt(Mat):

    d = Mat.shape
    for i in range(d[2]):
        for n in range(d[0]):
            for m in range(d[1]):
                Mat[n,m,i] = int(Mat[n,m,i])
                # print(Mat[n,m,i])
    Mat = Mat.astype(np.uint8)
    return Mat
#将LAB空间分割
def imgSplitLAB(img):

    l,a,b =  cv.split(img)
    return l,a,b
def gamma(image,thre):
    """
    :param image: numpy type
    :param thre:float
    :return: image numpy
    """
    f = image / 255.0
    # we can change thre accoding  to real condition
    # thre = 0.3
    out = np.power(f, thre)
    out = getMatInt(out * 255)
    return out
#将图像转换为LAB空间
def imgToLAB(img):
    # if img==None:
    #     return None
    lab = cv.cvtColor(img,cv.COLOR_BGR2LAB) #参数选择cv2.COLOR_BGR2Lab,cv2.COLOR_BGR2LAB

    return lab
#计算矩阵的平均值
def calMatAvg(img_single):
    print(img_single.shape)
    w,h = img_single.shape
    sum = 0

    for i in range(w):
        for j in range(h):
            sum = sum+img_single[i][j]

    avg = sum/(w*h)

    return avg

#计算矩阵标准差
def calMatSD(mat):

    mat = np.array(mat)
    sd = np.std(mat)

    return sd
def getPointSetByL(img_lab,l,b):

    L, A, B = imgSplitLAB(img_lab)
    avg_l = calMatAvg(L)
    avg_a = calMatAvg(A)
    avg_b = calMatAvg(B)
    sd = calMatSD(L)
    points = []
    w,h = L.shape
    thre = avg_l-(sd/3)
    print(avg_a+avg_b)

    for i in range(w):
        for j in range(h):
            print(B[i][j],end=" ")
            if (avg_a+avg_b)<256:
                if L[i][j]<thre:
                    points.append((i,j))
            elif (L[i][j]>l[0] and L[i][j]<l[1]) and B[i][j]>b[0] and B[i][j]<b[1]:
                points.append((i,j))
        print("")
    return points

if __name__ =="__main__":

    img = cv.imread("../images/meterReader/contact5.png")
    img = cv.GaussianBlur(img, (5, 5), 0)

    #LAB

    #GAMMA矫正
    #img = gamma(img,0.8)

    #RGB均衡化
    # (b, g, r) = cv.split(img)
    # bH = cv.equalizeHist(b)
    # gH = cv.equalizeHist(g)
    # rH = cv.equalizeHist(r)
    # # 合并每一个通道
    # img = cv.merge((bH, gH, rH))

    # #YUV均衡化
    # channelsYUV = cv.split(img)
    # channelsYUV[0] = cv.equalizeHist(channelsYUV[0])
    #
    # channels = cv.merge(channelsYUV)
    # img = cv.cvtColor(channels, cv.COLOR_YCrCb2BGR)


    # a = 2
    # img = img * float(a)
    # img[img > 255] = 255
    # img = np.round(img)
    # img = img.astype(np.uint8)


    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray_img = cv.equalizeHist(gray_img)
    ret, thresh = cv.threshold(gray_img, 100, 255, cv.THRESH_BINARY)
    cv.namedWindow("enhanced", 0)
    cv.resizeWindow("enhanced", 400, 200)
    cv.imshow("enhanced", img)

    cv.waitKey(0)
    cv.destroyWindow()



