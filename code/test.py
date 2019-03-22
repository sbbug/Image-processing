import numpy as np


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

a=np.array([[1,4],[3,4]])
b=np.array([[1,2],[3,4]])
print((a==b).all())
# b = np.array([[1,-1,1],[1,-1,1]])
#
# # for i in range(1,len(a)-1):
# #     for j in range(1,len(a[i])-1):
# print(imgConvoluting(a,b))
