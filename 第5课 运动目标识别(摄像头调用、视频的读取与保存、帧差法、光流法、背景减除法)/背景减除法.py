# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   yykzhjh@163.com
@DateTime :   2022/10/18 17:27
@Version  :   1.0
@License  :   (C)Copyright 2022
"""
import cv2
import numpy as np


"""
1. 背景消除：OpenCV中常用的两种背景消除方法，一种是基于高斯混合模型GMM实现的背景提取，另外一种是基于最近邻KNN实现的。
2. GMM模型：
    - MOG2算法，高斯混合模型分离算法，它为每个像素选择适当数量的高斯分布，它可以更好地适应不同场景的照明变化等
    - 函数：cv2.createBackgroundSubtractorMOG2(int history = 500, double varThreshold = 16, bool detectShadows = true)
3. KNN模型：cv2.createBackgroundSubtractorKNN()
4. 方法：主要通过视频中的背景进行建模，实现背景消除，生成mask图像，通过对mask二值图像分析实现对前景活动对象的区域的提取，整个步骤如下：
    1）初始化背景建模对象GMM
    2）读取视频一帧
    3）使用背景建模消除生成mask
    4）对mask进行轮廓分析提取ROI
    5）绘制ROI对象
"""
# 打开视频
cap = cv2.VideoCapture(r"./videos/move_detect.flv")

# 创建减除器
fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=100, detectShadows=False)




# 识别行人的函数
def get_person(image, opt=1):
    # 获得前景
    mask = fgbg.apply(image)

    # 获得卷积核
    line = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5), (-1, -1))
    # 开运算
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, line)
    cv2.imshow("mask", mask)

    # 搜索轮廓
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 遍历连通的轮廓
    for c in range(len(contours)):
        # 计算连通区域面积
        area = cv2.contourArea(contours[c])
        if area < 50:
            continue
        # 获得最小外接矩形
        rect = cv2.minAreaRect(contours[c])
        # 目标画椭圆
        cv2.ellipse(image, rect, (0, 255, 0), 2, 8)
        # 中心点画圆
        cv2.circle(image, (np.int32(rect[0][0]), np.int32(rect[0][1])), 2, (255, 0, 0), 2, 8, 0)
    return image, mask





# 主函数，遍历视频的每一帧
while cap.isOpened():
    # 读取一帧
    ret, frame = cap.read()
    # 获取减除后的结果
    result, m_ = get_person(frame)
    cv2.imshow("result", result)
    # 按键控制视频结束
    if cv2.waitKey(20) & 0xff == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


















