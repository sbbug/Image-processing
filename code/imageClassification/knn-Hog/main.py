import cv2
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import data, color, exposure
from sklearn.decomposition import PCA
import os
import numpy as np

train_dir = "../images/train/"
test_dir = "../images/test/"


# 加载数据集
def loadData(data_path):
    data_label = []
    data = []
    file_list = os.listdir(data_path)
    n = len(file_list)
    for i in range(n):

        # 获取标签数据
        data_label.append( [int(file_list[i].split("_")[0])] )

        # 获取图片特征
        image = cv2.imread(os.path.join(data_path, file_list[i]))
        image = cv2.resize(image, (256, 128))
        fd = hog(image, orientations=9, pixels_per_cell=(4, 4),
                                cells_per_block=(1, 1), visualize=False)


        data.append(fd)

    data = np.array(data)
    data_labels = np.array(data_label)

    return data.astype(np.float32),data_labels.astype(np.float32)
# 开始进行数据降维处理
def reduceDim(data_mat):
    pass


if __name__ =="__main__":

    train,train_label = loadData(train_dir)
    test,test_label = loadData(test_dir)

    print(train.shape)
    print(train_label.shape)

    knn = cv2.ml.KNearest_create()
    knn.train(train, cv2.ml.ROW_SAMPLE, train_label)
    ret,results,neighbours,dist = knn.findNearest(test,k=1)

    print("result: ", results)
    # print("neighbours: ", neighbours)
    # print("distance: ", dist)

    # 统计正确率
    right = 0
    all = len(test_label)
    for i in range(all):
        if results[i][0]==test_label[i][0]:
            right = right+1

    per = (right/all)
    print("准确率: ",per)
