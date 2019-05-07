'''
KNN类定义
'''
import cv2
from skimage.feature import hog
import numpy as np
import os
# 定义knn类
class Knn(object):

    train = None
    trainLabel = None

    # 构造方法
    def __init__(self):
        '''
        '''
        super(Knn, self).__init__()
        # 加载训练集到内存
        Knn.loadData()

    # 静态方法操作静态属性
    @classmethod
    def loadData(self):
        '''
        :return:
        '''
        if self.train is None or self.trainLabel is None:
            self.train, self.trainLabel = self.loadTrainData(self,"训练数据的目录")
            print("KNN训练数据加载完成")

    # 识别算法
    def getKnnResult(self,image):
        '''
        :param image: 目标图像 numpy矩阵
        :return:
        '''
        # 获取当前检测数据
        test = self.loadSingle(image)

        # 调用KNN分类
        knn = cv2.ml.KNearest_create()
        knn.train(self.train, cv2.ml.ROW_SAMPLE, self.trainLabel)
        ret, results, neighbours, dist = knn.findNearest(test, k=1)

        return int(results[0][0])

    # 获取图像的hog特征
    def getImageHog(self,image):
        '''
        :param image: 输入目标图像
        :return: 目标图像的hog特征向量
        '''
        resize_w = 128
        resize_h = 64

        image = cv2.resize(image, (resize_w, resize_h))

        # 获取hog特征
        fd = hog(image, orientations=9, pixels_per_cell=(4, 4),
                 cells_per_block=(1, 1), visualize=False)

        return fd

    # 获取单张图像的hog特征矩阵
    def loadSingle(self,image):
        '''
        :param image: RGB图像
        :return: 单张图像矩阵
        '''

        return np.array([self.getImageHog(image)]).astype(np.float32)

    def loadTrainData(self,dataPath):
        '''
        :param data_path: 图像数据所在目录
        :return: 返回图像hog特征矩阵 图像标签
        '''
        dataLabel = []
        data = []
        file_list = os.listdir(dataPath)
        n = len(file_list)
        for i in range(n):
            # 获取标签数据
            dataLabel.append([int(file_list[i].split("_")[0])])

            # 获取图片特征
            image = cv2.imread(os.path.join(dataPath, file_list[i]))

            # 提取hog特征
            fd = self.getImageHog(self,image)
            # 添加到data列表
            data.append(fd)

        # 数据类型转换
        data = np.array(data)
        data_labels = np.array(dataLabel)

        return data.astype(np.float32), data_labels.astype(np.float32)


if __name__ =="__main__":

    for i in range(10):
        knn = Knn()
        image = cv2.imread("./dataset/test/1_76.jpg")
        print(knn.getKnnResult(image))
