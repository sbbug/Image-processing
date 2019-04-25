import cv2
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances

#根据模板图像与与图像进行特征点匹配
def meterFinderBySIFT(image, template):


    templateBlurred = cv2.GaussianBlur(template, (3, 3), 0)
    imageBlurred = cv2.GaussianBlur(image, (3, 3), 0)

    sift = cv2.xfeatures2d.SIFT_create()

    # shape of descriptor n * 128, n is the num of key points.
    # a row of descriptor is the feature of related key point.
    templateKeyPoint, templateDescriptor = sift.detectAndCompute(templateBlurred, None)
    imageKeyPoint, imageDescriptor = sift.detectAndCompute(imageBlurred, None)
    print("templateKeyPoint")
    print(len(templateKeyPoint))
    print("templateDescriptor")
    print(templateDescriptor.shape)

    print("imageKeyPoint")
    print(len(imageKeyPoint))
    print("imageDescriptor")
    print(imageDescriptor.shape)

    # for debug
    templateBlurred = cv2.drawKeypoints(templateBlurred, templateKeyPoint, templateBlurred)
    imageBlurred = cv2.drawKeypoints(imageBlurred, imageKeyPoint, imageBlurred)
    cv2.imshow("template", templateBlurred)
    cv2.imshow("image", imageBlurred)
    cv2.waitKey(0)
    cv2.destroyWindow()
    '''
    # match
    bf = cv2.BFMatcher()
    # k = 2, so each match has 2 points. 2 points are sorted by distance.
    matches = bf.knnMatch(templateDescriptor, imageDescriptor, k=2)

    # The first one is better than the second one
    good = [[m] for m, n in matches if m.distance < 0.8 * n.distance]

    # distance matrix
    templatePointMatrix = np.array([list(templateKeyPoint[p[0].queryIdx].pt) for p in good])
    imagePointMatrix = np.array([list(imageKeyPoint[p[0].trainIdx].pt) for p in good])
    templatePointDistanceMatrix = pairwise_distances(templatePointMatrix, metric="euclidean")
    imagePointDistanceMatrix = pairwise_distances(imagePointMatrix, metric="euclidean")

    # del bad match
    distances = []
    maxAbnormalNum = 15
    for i in range(len(good)):
        diff = abs(templatePointDistanceMatrix[i] - imagePointDistanceMatrix[i])
        # distance between distance features
        diff.sort()
        distances.append(np.sqrt(np.sum(np.square(diff[:-maxAbnormalNum]))))

    averageDistance = np.average(distances)
    good2 = [good[i] for i in range(len(good)) if distances[i] < 2 * averageDistance]

    # for debug
    # matchImage = cv2.drawMatchesKnn(template, templateKeyPoint, image, imageKeyPoint, good2, None, flags=2)
    # cv2.imshow("matchImage", matchImage)
    # cv2.waitKey(0)

    # not match
    if len(good2) < 3:
        print("not found!")
        return template
    '''
    '''
    # 寻找转换矩阵 M
    src_pts = np.float32([templateKeyPoint[m[0].queryIdx].pt for m in good2]).reshape(-1, 1, 2)
    dst_pts = np.float32([imageKeyPoint[m[0].trainIdx].pt for m in good2]).reshape(-1, 1, 2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()
    h, w, _ = template.shape

    # 找出匹配到的图形的四个点和标定信息里的所有点
    pts = np.float32(
        [[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0], [startPoint[0], startPoint[1]], [endPoint[0], endPoint[1]],
         [centerPoint[0], centerPoint[1]],
         # [startPointUp[0], startPointUp[1]],
         # [endPointUp[0], endPointUp[1]],
         # [centerPointUp[0], centerPointUp[1]]
         ]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, M)

    # 校正图像
    angle = 0.0
    vector = (dst[3][0][0] - dst[0][0][0], dst[3][0][1] - dst[0][0][1])
    cos = (vector[0] * (200.0)) / (200.0 * math.sqrt(vector[0] ** 2 + vector[1] ** 2))
    if (vector[1] > 0):
        angle = math.acos(cos) * 180.0 / math.pi
    else:
        angle = -math.acos(cos) * 180.0 / math.pi
    # print(angle)

    change = cv2.getRotationMatrix2D((dst[0][0][0], dst[0][0][1]), angle, 1)
    src_correct = cv2.warpAffine(image, change, (image.shape[1], image.shape[0]))
    array = np.array([[0, 0, 1]])
    newchange = np.vstack((change, array))
    # 获得校正后的所需要的点
    newpoints = []
    for i in range(len(pts)):
        point = newchange.dot(np.array([dst[i][0][0], dst[i][0][1], 1]))
        point = list(point)
        point.pop()
        newpoints.append(point)
    src_correct = src_correct[int(round(newpoints[0][1])):int(round(newpoints[1][1])),
                  int(round(newpoints[0][0])):int(round(newpoints[3][0]))]

    width = src_correct.shape[1]
    height = src_correct.shape[0]
    if width == 0 or height == 0:
        return template

    startPoint = (int(round(newpoints[4][0]) - newpoints[0][0]), int(round(newpoints[4][1]) - newpoints[0][1]))
    endPoint = (int(round(newpoints[5][0]) - newpoints[0][0]), int(round(newpoints[5][1]) - newpoints[0][1]))
    centerPoint = (int(round(newpoints[6][0]) - newpoints[0][0]), int(round(newpoints[6][1]) - newpoints[0][1]))

    def isOverflow(point, width, height):
        if point[0] < 0 or point[1] < 0 or point[0] > width - 1 or point[1] > height - 1:
            return True
        return False

    if isOverflow(startPoint, width, height) or isOverflow(endPoint, width, height) or isOverflow(centerPoint, width,
                                                                                                  height):
        print("overflow!")
        return template

    # startPointUp = (int(round(newpoints[7][0]) - newpoints[0][0]), int(round(newpoints[7][1]) - newpoints[0][1]))
    # endPointUp = (int(round(newpoints[8][0]) - newpoints[0][0]), int(round(newpoints[8][1]) - newpoints[0][1]))
    # centerPointUp = (int(round(newpoints[9][0]) - newpoints[0][0]), int(round(newpoints[9][1]) - newpoints[0][1]))
    info["startPoint"]["x"] = startPoint[0]
    info["startPoint"]["y"] = startPoint[1]
    info["centerPoint"]["x"] = centerPoint[0]
    info["centerPoint"]["y"] = centerPoint[1]
    info["endPoint"]["x"] = endPoint[0]
    info["endPoint"]["y"] = endPoint[1]
    '''
    #return src_correct


if __name__ =="__main__":

   image = cv2.imread("../images/sift/image.jpg")
   tpl = cv2.imread("../images/sift/tpl.png")

   meterFinderBySIFT(image,tpl)