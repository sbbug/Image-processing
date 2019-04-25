import cv2
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import data, color, exposure
import os
import numpy as np

train_dir = "../images/train/"
test_dir = "../images/test/"

# 加载数据集
def loadData(data_path):
    train_label = []
    train = []
    file_list = os.listdir(data_path)
    n = len(file_list)
    for i in range(n):

        # 获取标签数据
        train_label.append( [int(file_list[i].split("_")[0])] )

        # 获取图片特征
        image = cv2.imread(os.path.join(data_path, file_list[i]))
        image = cv2.resize(image, (256, 128))
        fd = hog(image, orientations=9, pixels_per_cell=(4, 4),
                                cells_per_block=(1, 1), visualize=False)

        train.append(fd)


    train = np.array(train)
    train_labels = np.array(train_label)
    print(train)
    print(train_label)
    np.savetxt("train.txt",train)
    np.savetxt("train_labels.txt",train_label)
    # train = train.reshape((train.size, 2))
    # train_labels = train_labels.reshape((train_labels.size, 1))
    return train.astype(np.float32),train_labels.astype(np.float32)
if __name__ =="__main__":
    #train = np.random.randint(1, 10, (73, 18432)).astype(np.float32)
    train,train_label = loadData(train_dir)
    #test,_ = loadData(test_dir)

    print(train.shape)
    print(train_label.shape)
    #test,_ = loadData(test_dir)
    # Initiate kNN, train the data, then test it with test data for k=1

    knn = cv2.ml.KNearest_create()
    knn.train(train, cv2.ml.ROW_SAMPLE, train_label)
    #ret,results,neighbours,dist = knn.findNearest(test,k=3)

    # print("result: ", results)
    # print("neighbours: ", neighbours)
    # print("distance: ", dist)
#
#
# #image = cv2.imread(filename)
#
# file_list = os.listdir(data_dir)
#
# for i in range(len(file_list)):
#     if i==1:
#         break;
#     image = cv2.imread(os.path.join(data_dir,file_list[i]))
#     image = cv2.resize(image, (256, 128))
#     fd, hog_image = hog(image, orientations=9, pixels_per_cell=(4, 4),
#                         cells_per_block=(1, 1), visualize=True)
#
#     print(image.shape)
#     print(fd.shape)
#     print(hog_image.shape)
#     fig, ax = plt.subplots(1, 2, figsize=(20, 10), sharex=True, sharey=True)
#
#     ax[0].axis('off')
#     ax[0].imshow(image, cmap=plt.cm.gray)
#     ax[0].set_title('Input image')
#     ax[0].set_adjustable('box-forced')
#
#     # Rescale histogram for better display
#     hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 0.02))
#
#     ax[1].axis('off')
#     ax[1].imshow(hog_image, cmap=plt.cm.gray)
#     ax[1].set_title('Histogram of Oriented Gradients')
#     ax[1].set_adjustable('box-forced')
#
#     plt.show()
