import numpy as np
import cv2 as cv

#实现图像的卷积操作
def imgConvoluting(image,filter):

    w,h= filter.shape
    con_img = image.copy()
    filter_w,filter_h = filter.shape
    for i in range(1,len(image)+1-w):
        for j in range(1,len(image[i])+1-h):
              print(image[i:i+filter_w:,j:j+filter_h:])
              con_img[i][j] = (image[i:i+filter_w:,j:j+filter_h:]*filter).sum()

    return con_img

a = cv.imread("../images/5.png")
b = cv.imread("../images/Y.jpg")
a = a.astype(int)
b = b.astype(int)
#c= cv.subtract(a,b)
c = np.abs(np.subtract(a,b))
r,_,_ = cv.split(a)
for i in range(len(r)):
    for j in range(len(r[i])):
        print(r[i][j],end=" ")
    print("")
print("=====================")
r,_,_ = cv.split(b)
for i in range(len(r)):
    for j in range(len(r[i])):
        print(r[i][j],end=" ")
    print("")

print("=====================")
r,_,_ = cv.split(c)
for i in range(len(r)):
    for j in range(len(r[i])):
        print(r[i][j],end=" ")
    print("")



# b = np.array([[1,-1,1],[1,-1,1]])
#
# # for i in range(1,len(a)-1):
# #     for j in range(1,len(a[i])-1):
# print(imgConvoluting(a,b))
