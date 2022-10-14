# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   yykzhjh@163.com
@DateTime :   2022/10/14 18:03
@Version  :   1.0
@License  :   (C)Copyright 2022
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt


"""
1. 形态学：是图像处理中应用最为广泛的技术之一，主要用于从图像中提取对表达和描绘区域形状有意义的图像分量，
使后续的识别工作能够抓住目标对象最为本质的形状特征，如边界和连通区域等。
2. 结构元素：设有两幅图像B、X，若X是被处理的对象，而B是用来处理X的，则称B为结构元素(structure element)，
又被形象地称做刷子，结构元素通常都是一些比较小的图像
"""

"""
1. 图像的膨胀（Dilation）和腐蚀（Erosion）是两种基本的形态学运算，
2. 膨胀类似于“领域扩张”，将图像中的白色部分进行扩张，其运行结果图比原图的白色区域更大；
3. 腐蚀类似于“领域被蚕食”，将图像中白色部分进行缩减细化，其运行结果图比原图的白色区域更小。
"""

"""
图像腐蚀函数：
cv2.erode（src,element,anchor,iterations）
    - 参数1：src，原图像
    - 参数2：element，腐蚀操作的内核(结构元素)，默认为一个简单的3x3矩形
    - 参数3：anchor，默认为Point(-1,-1),内核中心点
    - 参数4：iterations，腐蚀次数,默认值1
"""
# # 读取图像
# img = cv2.imread(r"./images/morphology.png")
# # 转换颜色格式
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# # 定义结构元素
# kernel = np.ones((3, 3), dtype=np.uint8)
# # 进行腐蚀操作
# erosion = cv2.erode(img, kernel, iterations=1)
# # 显示
# plt.subplot(121), plt.imshow(img), plt.title("Origin")
# plt.xticks([]), plt.yticks([])
# plt.subplot(122), plt.imshow(erosion), plt.title("Erosion")
# plt.xticks([]), plt.yticks([])
# plt.show()

"""
膨胀(dilation)可以看做是腐蚀的对偶运算
图像膨胀函数：
cv2.dilate（src,element,anchor,iterations）
    - 参数1：src，原图像
    - 参数2：element，膨胀操作的内核(结构元素)，默认为一个简单的3x3矩形
    - 参数3：anchor，默认为Point(-1,-1),内核中心点
    - 参数4：iterations，膨胀次数,默认值1
"""
# # 读取图像
# img = cv2.imread(r"./images/morphology.png")
# # 转换颜色格式
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#
# # 定义结构元素
# # kernel = np.ones((3, 3), dtype=np.uint8)
# # # 十字星
# # kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (7, 7))
# # # 椭圆形
# # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
# # # 矩形
# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
#
# # 进行膨胀操作
# erosion = cv2.dilate(img, kernel, iterations=1)
# # 显示
# plt.subplot(121), plt.imshow(img), plt.title("Origin")
# plt.xticks([]), plt.yticks([])
# plt.subplot(122), plt.imshow(erosion), plt.title("Dilation")
# plt.xticks([]), plt.yticks([])
# plt.show()

"""
开运算：
开运算 = 先腐蚀运算，再膨胀运算(看上去把细微连在一起的两块目标分开了)
开运算总结：
1. 开运算能够除去孤立的小点，毛刺和小桥，而总的位置和形状不变。
2. 开运算是一个基于几何运算的滤波器。
3. 结构元素大小的不同将导致滤波效果的不同。
4. 不同的结构元素的选择导致了不同的分割，即提取出不同的特征。
"""
# # 读取图像
# img = cv2.imread(r"./images/open.png")
# # 转换颜色格式
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#
# # 定义结构元素
# # kernel = np.ones((3, 3), dtype=np.uint8)
# # # 十字星
# # kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (7, 7))
# # # 椭圆形
# # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
# # # 矩形
# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
#
# # 进行开运算
# opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
# # 显示
# plt.subplot(121), plt.imshow(img), plt.title("Origin")
# plt.xticks([]), plt.yticks([])
# plt.subplot(122), plt.imshow(opening), plt.title("Opening")
# plt.xticks([]), plt.yticks([])
# plt.show()

"""
闭运算：
闭运算 = 先膨胀运算，再腐蚀运算(看上去将两个细微连接的图块封闭在一起)
闭运算总结：
1. 闭运算能够填平小湖（即小孔），弥合小裂缝，而总的位置和形状不变。
2. 闭运算是通过填充图像的凹角来滤波图像的。
3. 结构元素大小的不同将导致滤波效果的不同。
4. 不同结构元素的选择导致了不同的分割。
"""
# # 读取图像
# img = cv2.imread(r"./images/close.png")
# # 转换颜色格式
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#
# # 定义结构元素
# # kernel = np.ones((3, 3), dtype=np.uint8)
# # # 十字星
# # kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (7, 7))
# # # 椭圆形
# # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
# # # 矩形
# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
#
# # 进行闭运算
# closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
# # 显示
# plt.subplot(121), plt.imshow(img), plt.title("Origin")
# plt.xticks([]), plt.yticks([])
# plt.subplot(122), plt.imshow(closing), plt.title("Closing")
# plt.xticks([]), plt.yticks([])
# plt.show()

"""
形态学梯度（Gradient）：
1. 基础梯度：基础梯度是用膨胀后的图像减去腐蚀后的图像得到差值图像，也是opencv中支持的计算形态学梯度的方法，而此方法得到梯度有称为基本梯度。
2. 内部梯度：是用原图像减去腐蚀之后的图像得到差值图像，称为图像的内部梯度。
3. 外部梯度：图像膨胀之后再减去原来的图像得到的差值图像，称为图像的外部梯度。
"""
# # 读取图像
# img = cv2.imread(r"./images/morphology.png")
# # 转换颜色格式
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#
# # 定义结构元素
# # kernel = np.ones((3, 3), dtype=np.uint8)
# # # 十字星
# # kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (7, 7))
# # # 椭圆形
# # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
# # # 矩形
# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
#
# # 进行形态学梯度运算,默认是基础梯度
# gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
# # 显示
# plt.subplot(121), plt.imshow(img), plt.title("Origin")
# plt.xticks([]), plt.yticks([])
# plt.subplot(122), plt.imshow(gradient), plt.title("Gradient")
# plt.xticks([]), plt.yticks([])
# plt.show()


"""
顶帽（Top Hat）：原图像与开运算图的区别（差值），突出原图像中比周围亮的区域
"""
# # 读取图像
# img = cv2.imread(r"./images/morphology.png")
# # 转换颜色格式
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#
# # 定义结构元素
# # kernel = np.ones((3, 3), dtype=np.uint8)
# # # 十字星
# # kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (7, 7))
# # # 椭圆形
# # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
# # # 矩形
# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
#
# # 进行形态学顶帽运算
# tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
# # 显示
# plt.subplot(121), plt.imshow(img), plt.title("Origin")
# plt.xticks([]), plt.yticks([])
# plt.subplot(122), plt.imshow(tophat), plt.title("TopHat")
# plt.xticks([]), plt.yticks([])
# plt.show()

"""
黑帽（Black Hat）：闭操作图像 - 原图像,突出原图像中比周围暗的区域
"""
# 读取图像
img = cv2.imread(r"./images/morphology.png")
# 转换颜色格式
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 定义结构元素
# kernel = np.ones((3, 3), dtype=np.uint8)
# # 十字星
# kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (7, 7))
# # 椭圆形
# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
# # 矩形
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))

# 进行形态学黑帽运算
blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
# 显示
plt.subplot(121), plt.imshow(img), plt.title("Origin")
plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(blackhat), plt.title("BlackHat")
plt.xticks([]), plt.yticks([])
plt.show()


