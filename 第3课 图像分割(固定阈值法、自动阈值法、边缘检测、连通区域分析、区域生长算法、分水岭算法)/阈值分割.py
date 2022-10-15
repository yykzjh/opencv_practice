# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   yykzhjh@163.com
@DateTime :   2022/10/15 16:51
@Version  :   1.0
@License  :   (C)Copyright 2022
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt



"""
1. 直方图双峰法：六十年代中期提出的直方图双峰法(也称 mode 法)是典型的全局单阈值分割方法。
2. 基本思想：假设图像中有明显的目标和背景，则其灰度直方图呈双峰分布，当灰度级直方图具有双峰特性时，
选取两峰之间的谷对应的灰度级作为阈值。
"""

"""
固定阈值分割函数：
cv2.threshold(src, thresh, maxval, type)
    - 参数1：原图像
    - 参数2：对像素值进行分类的阈值
    - 参数3：当像素值高于(小于)阈值时，应该被赋予的新的像素值，
    - 参数4：第四个参数是阈值方法。
        - THRESH_BINARY：       maxval if src(x, y) > thresh else 0
        - THRESH_BINARY_INV：   0 if src(x, y) > thresh else maxval
        - THRESH_TRUNC：        thresh if src(x, y) > thresh else src(x, y)
        - THRESH_TOZERO：       src(x, y) if src(x, y) > thresh else 0
        - THRESH_TOZERO_INV：   0 if src(x, y) > thresh else src(x, y)
    - 返回值：函数有两个返回值，一个为retVal, 一个阈值化处理之后的图像。
"""
# # 灰度图读取
# img = cv2.imread(r"./images/thresh.png", 0)
# # 阈值分割,二值化方法
# ret, th = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
# print(ret)
# # 显示分割后的图像
# cv2.imshow("thresh", th)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

"""
五种常用固定阈值分割的方法
- THRESH_BINARY：       maxval if src(x, y) > thresh else 0
- THRESH_BINARY_INV：   0 if src(x, y) > thresh else maxval
- THRESH_TRUNC：        thresh if src(x, y) > thresh else src(x, y)
- THRESH_TOZERO：       src(x, y) if src(x, y) > thresh else 0
- THRESH_TOZERO_INV：   0 if src(x, y) > thresh else src(x, y)
"""
# # 读取图像
# img = cv2.imread(r"./images/person.png", 0)
# # 5种阈值法分割图像
# ret1, thresh1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
# ret2, thresh2 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
# ret3, thresh3 = cv2.threshold(img, 127, 255, cv2.THRESH_TRUNC)
# ret4, thresh4 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO)
# ret5, thresh5 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO_INV)
#
# # 将分割结果放在一起
# images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]
# # 循环遍历，进行显示
# for i in range(6):
#     plt.subplot(2, 3, i+1)
#     plt.imshow(images[i], cmap="gray")
# plt.suptitle("fixed threshold")
# plt.show()

"""
自动阈值分割法——自适应阈值法：自适应阈值会每次取图片的一小部分计算阈值，这样图片不同区域的阈值就不尽相同，适用于明暗分布不均的图片。
自适应阈值分割函数：
cv2.adaptiveThreshold()
    - 参数1：要处理的原图
    - 参数2：高于阈值时赋予的值，一般为255
    - 参数3：小区域阈值的计算方式
        - ADAPTIVE_THRESH_MEAN_C：小区域内取均值
        - ADAPTIVE_THRESH_GAUSSIAN_C：小区域内加权求和，权重是个高斯核
    - 参数4：阈值方式（跟前面讲的那5种相同）
        - THRESH_BINARY：       maxval if src(x, y) > thresh else 0
        - THRESH_BINARY_INV：   0 if src(x, y) > thresh else maxval
        - THRESH_TRUNC：        thresh if src(x, y) > thresh else src(x, y)
        - THRESH_TOZERO：       src(x, y) if src(x, y) > thresh else 0
        - THRESH_TOZERO_INV：   0 if src(x, y) > thresh else src(x, y)
    - 参数5：小区域的边长，如11就是11*11的小块
    - 参数6：最终阈值等于小区域计算出的阈值再减去此值,白色比较多取正值,黑色比较多取负值
"""
# # 读取图像
# img = cv2.imread(r"./images/paper2.png", 0)
#
# # 固定阈值
# _, th1 = cv2.threshold(img, 75, 255, cv2.THRESH_BINARY)
# # 自适应阈值,均值
# th2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 4)
# # 自适应阈值,高斯核
# th3 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 17, 6)
#
# # 全局固定阈值、均值自适应、高斯加权自适应对比
# titles = ["Origin", "Global(v = 127)", "Adaptive Mean", "Adaptive Gaussian"]
# images = [img, th1, th2, th3]
# for i in range(4):
#     plt.subplot(2, 2, i+1),
#     plt.imshow(images[i], cmap="gray")
#     plt.title(titles[i], fontsize=8)
#     plt.xticks([])
#     plt.yticks([])
# plt.show()

"""
自动阈值分割法——迭代法阈值分割：
步骤：
1． 求出图像的最大灰度值和最小灰度值，分别记为ZMAX和ZMIN，令初始阈值T0=(ZMAX+ZMIN)/2；
2． 根据阈值TK将图象分割为前景和背景，分别求出两者的平均灰度值ZO和ZB ；
3． 求出新阈值TK+1=(ZO+ZB)/2；
4． 若TK==TK+1，则所得即为阈值；否则转2，迭代计算；
5 ． 使用计算后的阈值进行固定阈值分割。
"""

"""
自动阈值分割法——Otsu大津法：最大类间方差法，1979年日本学者大津提出，是一种基于全局阈值的自适应方法。
1. 灰度特性：图像分为前景和背景。当取最佳阈值时，两部分之间的差别应该是最大的，衡量差别的标准为最大类间方差。
2. 直方图有两个峰值的图像，大津法求得的阈值T近似等于两个峰值之间的低谷。
函数参数说明：
1. thresh在这里是迭代的初始值，不断+1;
2. 参数4的阈值方法与其他方法结合使用，cv2.THRESH_OTSU只是为了自适应确定阈值，具体分割方法还需要另外指定
"""
# 读取图像
img = cv2.imread(r"./images/noisy.png", 0)

# 固定阈值分割法
_, th1 = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
# Otsu阈值分割法
_, th2 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# 先进性高斯滤波，再使用Otsu阈值法
blur = cv2.GaussianBlur(img, (5, 5), 0)
_, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# 对比
images = [img, 0, th1, img, 0, th2, blur, 0, th3]
titles = [
    "Origin", "Histogram", "Global(v=100)",
    "Origin", "Histogram", "Otsu's",
    "Gaussian Filtered Image", "Histogram", "Otsu's"
]

# 循环显示
for i in range(3):
    # 绘制原图
    plt.subplot(3, 3, i * 3 + 1)
    plt.imshow(images[i * 3], cmap="gray")
    plt.title(titles[i * 3], fontsize=8)
    plt.xticks([]), plt.yticks([])

    # 绘制直方图plt.hist, ravel()函数将数组将成一维
    plt.subplot(3, 3, i * 3 + 2)
    plt.hist(images[i * 3].ravel(), 256)
    plt.title(titles[i * 3 + 1], fontsize=8)
    plt.xticks([]), plt.yticks([])

    # 绘制阈值分割图
    plt.subplot(3, 3, i * 3 + 3)
    plt.imshow(images[i * 3 + 2], cmap="gray")
    plt.title(titles[i * 3 + 2], fontsize=8)
    plt.xticks([]), plt.yticks([])
plt.show()




