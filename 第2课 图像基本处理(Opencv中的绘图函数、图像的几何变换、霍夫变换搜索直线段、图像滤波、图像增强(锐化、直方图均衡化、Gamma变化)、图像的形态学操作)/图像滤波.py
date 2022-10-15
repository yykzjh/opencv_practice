# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   yykzhjh@163.com
@DateTime :   2022/10/13 22:22
@Version  :   1.0
@License  :   (C)Copyright 2022
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt


"""
1. 滤波实际上是信号处理得一个概念，图像可以看成一个二维信号，其中像素点的灰度值代表信号的强弱；
2. 高频：图像上变化剧烈的部分；
3. 低频：图像灰度值变化缓慢，平坦的地方；
4. 根据图像高低频，设置高通和低通滤波器。高通滤波器可以检测变化尖锐，明显的地方，低通可以让图像变得平滑，消除噪声；
5. 滤波作用：高通滤波器用于边缘检测，低通滤波器用于图像平滑去噪；
6. 线性滤波：方框滤波/均值滤波/高斯滤波；
7. 非线性滤波：中值滤波/双边滤波；
8. 领域算子：利用给定像素周围的像素值决定此像素的最终输出值的一种算子；
9. 线性滤波:一种常用的领域算子，像素输出取决于输入像素的加权和：
"""

"""
线性滤波(方框滤波)：方框滤波（box Filter）被封装在一个名为boxFilter的函数中
boxFilter函数的作用是使用方框滤波器（box filter）来模糊一张图片，从src输入，从dst输出；
方框滤波核：
normalize = true 与均值滤波相同
normalize = false 很容易发生溢出
方框滤波函数：
cv2.boxFilter(src,depth，ksize，normalize)
    - 参数1：输入图像
    - 参数2：目标图像深度，默认和输入图像深度一致
    - 参数3：核大小，默认3X3
    - 参数4：normalize属性
"""
# # 读取图像,cv2.IMREAD_UNCHANGED原图深度为多少就是多少
# img = cv2.imread(r"./images/girl2.png", cv2.IMREAD_UNCHANGED)
# # 方框滤波1, 深度-1默认，均值滤波
# r = cv2.boxFilter(img, -1, (3, 3), normalize=1)
# # 方框滤波2, 深度-1默认
# d = cv2.boxFilter(img, -1, (3, 3), normalize=0)
#
# # 输出比较
# cv2.namedWindow("img", cv2.WINDOW_AUTOSIZE)
# cv2.namedWindow("r", cv2.WINDOW_AUTOSIZE)
# cv2.namedWindow("d", cv2.WINDOW_AUTOSIZE)
# cv2.imshow("img", img)
# cv2.imshow("r", r)
# cv2.imshow("d", d)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

"""
线性滤波(均值滤波)：均值滤波是一种最简单的滤波处理，它取的是卷积核区域内元素的均值
均值滤波函数：
cv2.blur(src, ksize)
    - 参数1：输入原图
    - 参数2：kernel的大小，一般为奇数
"""
# # 读取图像
# img = cv2.imread(r"./images/opencv.png")
# # 转换格式
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# # 进行均值滤波
# blur = cv2.blur(img, (11, 11))
# # 显示
# plt.subplot(121), plt.imshow(img), plt.title("Origin")
# plt.xticks([]), plt.yticks([])
# plt.subplot(122), plt.imshow(blur), plt.title("Blurred")
# plt.xticks([]), plt.yticks([])
# plt.show()

"""
线性滤波(高斯滤波):高斯滤波是一种线性平滑滤波，适用于消除高斯噪声，广泛应用于图像处理的减噪过程。
高斯滤波的卷积核权重并不相同，中间像素点权重最高，越远离中心的像素权重越小。其原理是一个2维高斯函数.
高斯滤波相比均值滤波效率要慢，但可以有效消除高斯噪声，能保留更多的图像细节，所以经常被称为最有用的滤波器。
高斯滤波函数：
cv2.Guassianblur(src, ksize, std)
    - 参数1：输入原图
    - 参数2：高斯核大小
    - 参数3：标准差σ，平滑时，调整σ实际是在调整周围像素对当前像素的影响程度，
      调大σ即提高了远处像素对中心像素的影响程度，滤波结果也就越平滑
"""
# # 读取图像
# img = cv2.imread(r"./images/median.png")
# # 转换格式
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# # 进行高斯滤波
# blur = cv2.GaussianBlur(img, (11, 11), 3)
# # 显示
# plt.subplot(121), plt.imshow(img), plt.title("Origin")
# plt.xticks([]), plt.yticks([])
# plt.subplot(122), plt.imshow(blur), plt.title("Blurred")
# plt.xticks([]), plt.yticks([])
# plt.show()

"""
非线性滤波(中值滤波)：中值滤波是一种非线性滤波，是用像素点邻域灰度值的中值代替该点的灰度值，中值滤波可以去除椒盐噪声和斑点噪声。
中值滤波函数：
cv2.medianBlur(img, ksize)
    - 参数1：输入原图
    - 参数2：核大小
"""
# # 读取图像
# img = cv2.imread(r"./images/median.png")
# # 转换格式
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# # 进行中值滤波
# blur = cv2.medianBlur(img, 9)
# # 显示
# plt.subplot(121), plt.imshow(img), plt.title("Origin")
# plt.xticks([]), plt.yticks([])
# plt.subplot(122), plt.imshow(blur), plt.title("Blurred")
# plt.xticks([]), plt.yticks([])
# plt.show()

"""
非线性滤波(双边滤波)：双边滤波是一种非线性的滤波方法，是结合图像的空间邻近度和像素值相似度的一种折中处理，
同时考虑空间信息和灰度相似性，达到保边去噪的目的，具有简单、非迭代、局部处理的特点。
双边滤波函数：
cv2.bilateralFilter(src=image, d, sigmaColor, sigmaSpace)
    - 参数1：输入原图
    - 参数2：像素的邻域直径
    - 参数3：灰度值相似性高斯函数标准差
    - 参数4：空间高斯函数标准差
"""
# 读取图像
img = cv2.imread(r"./images/bilateral.png")
# 转换格式
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# 进行双边滤波
blur = cv2.bilateralFilter(img, -1, 15, 10)
# 显示
plt.subplot(121), plt.imshow(img), plt.title("Origin")
plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(blur), plt.title("Blurred")
plt.xticks([]), plt.yticks([])
plt.show()










