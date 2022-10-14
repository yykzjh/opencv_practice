# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   yykzhjh@163.com
@DateTime :   2022/10/14 17:33
@Version  :   1.0
@License  :   (C)Copyright 2022
"""
import cv2
import numpy as np

"""
1. 目的：直方图均衡化是将原图像通过某种变换，得到一幅灰度直方图为均匀分布的新图像的方法。
2. 直方图均衡化方法的基本思想：是对在图像中像素个数多的灰度级进行展宽，而对像素个数少的灰度级进行缩减。从而达到清晰图像的目的。
3. 直方图均衡化函数：
cv2.equalizeHist(img)
    - 参数1：待均衡化图像
4. 步骤:
    1) 统计直方图中每个灰度级出现的次数；
    2) 计算累计归一化直方图；
    3) 重新计算像素点的像素值
"""
# # 直接读取为灰度图像
# img = cv2.imread(r"./images/dark.jpg", 0)
# cv2.imshow("dark", img)
# # 对图像进行直方图均衡化
# img_equal = cv2.equalizeHist(img)
# # 展示图像
# cv2.imshow("img_equal", img_equal)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

"""
彩色直方图均衡化，将三个通道分离后分别进行灰度直方图均衡化
"""
# # 直接读取为灰度图像
# img = cv2.imread(r"./images/dark1.jpg", 1)
# cv2.imshow("src", img)
# # 对图像进行分离
# (b, g, r) = cv2.split(img)
# # 分别对三个通道进行直方图均衡化
# bH = cv2.equalizeHist(b)
# gH = cv2.equalizeHist(g)
# rH = cv2.equalizeHist(r)
# # 合并三个通道
# res = cv2.merge((bH, gH, rH))
# # 展示图像
# cv2.imshow("dst", res)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

"""
1. Gamma变化：Gamma变换是对输入图像灰度值进行的非线性操作，使输出图像灰度值与输入图像灰度值呈指数关系：
2. 目的：Gamma变换就是用来图像增强，其提升了暗部细节，通过非线性变换，让图像从暴光强度的线性响应变得更接近人眼感受的响应，
即将漂白（相机曝光）或过暗（曝光不足）的图片，进行矫正。
"""
# 读取一张图像
img = cv2.imread(r"./images/dark1.jpg")
# 该函数返回一个0~255的灰度值经过gamma变换后的灰度值映射表
def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = []
    for i in range(256):
        table.append(((i / 255.0) ** invGamma) * 255)
    table = np.array(table).astype("uint8")
    # 根据表对图像中的值进行映射
    return cv2.LUT(image, table)

img_gamma = adjust_gamma(img, 0.8)
# 展示图像
cv2.imshow("img", img)
cv2.imshow("img_gamma", img_gamma)
cv2.waitKey(0)
cv2.destroyAllWindows()







