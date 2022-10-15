# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   yykzhjh@163.com
@DateTime :   2022/10/15 18:12
@Version  :   1.0
@License  :   (C)Copyright 2022
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

"""
1. 图像梯度：图像梯度即图像中灰度变化的度量，求图像梯度的过程是二维离散函数求导过程。
边缘其实就是图像上灰度级变化很快的点的集合。
2. 模板卷积：要理解梯度图的生成，就要先了解模板卷积的过程，模板卷积是模板运算的一种方式，其步骤如下：
    （1）将模板在输入图像中漫游，并将模板中心与图像中某个像素位置重合；
    （2）将模板上各个系数与模板下各对应像素的灰度相乘；
    （3）将所有乘积相加（为保持灰度范围，常将结果再除以模板系数之和，后面梯度算子模板和为0的话就不需要除了）；
    （4）将上述运算结果（模板的响应输出）赋给输出图像中对应模板中心位置的像素。
3. 梯度图：梯度图的生成和模板卷积相同，不同的是要生成梯度图，还需要在模板卷积完成后计算在点(x,y)梯度的幅值，
将幅值作为像素值，这样才算完。
注意：梯度图上每个像素点的灰度值就是梯度向量的幅度。
4. 梯度算子：梯度算子是一阶导数算子，是水平G(x)和竖直G(y)方向对应模板的组合，也有对角线方向。
常见的一阶算子：Roberts交叉算子、Prewitt算子、Sobel算子
5. Roberts交叉算子：其本质是一个对角线方向的梯度算子。
    Mx = [[0, 1], 
          [-1, 0]]  
    My = [[1, 0], 
          [0, -1]]
    优点：边缘定位较准，适用于边缘明显且噪声较少的图像。
    缺点：
        （1）没有描述水平和竖直方向的灰度变化，只关注了对角线方向，容易造成遗漏。
        （2）鲁棒性差。由于点本身参加了梯度计算，不能有效的抑制噪声的干扰。
6. Prewitt算子：是典型的3*3模板。 
    Mx = [[-1, 0, 1], 
          [-1, 0, 1], 
          [-1, 0, 1]]  
    My = [[1, 1, 1], 
          [0, 0, 0], 
          [-1, -1, -1]]
    优点：Prewitt算子引入了类似局部平均的运算，对噪声具有平滑作用，较Roberts算子更能抑制噪声。
7. Sobel算子：是增加了权重系数的Prewitt算子，其模板中心对应要求梯度的原图像坐标。
    Mx = [[-1, 0, 1], 
          [-2, 0, 2], 
          [-1, 0, 1]]  
    My = [[1, 2, 1], 
          [0, 0, 0], 
          [-1, -2, -1]]
    优点：Sobel算子引入了类似局部加权平均的运算，对边缘的定位比要比Prewitt算子好。
"""

"""
Sobel算子函数：
dst = cv2.Sobel(src, ddepth, dx, dy，ksize)
    - 参数1：需要处理的图像；
    - 参数2：图像的深度，-1表示采用的是与原图像相同的深度。目标图像的深度必须大于等于原图像的深度,一般用cv2.CV_64F
    - 参数3，4：dx和dy表示的是求导的阶数，0表示这个方向上没有求导，一般为0、1、2；
    - 参数5：ksize是Sobel算子的大小，必须为1、3、5、7
"""
# # 读取图像
# img = cv2.imread(r"./images/girl2.png", 0)
# # 用sobel算子计算梯度
# sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
# sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
# # 画图
# plt.subplot(1, 3, 1), plt.imshow(img, cmap="gray"), plt.title("Origin")
# plt.xticks([]), plt.yticks([])
# plt.subplot(1, 3, 2), plt.imshow(sobelx, cmap="gray"), plt.title("Sobel X")
# plt.xticks([]), plt.yticks([])
# plt.subplot(1, 3, 3), plt.imshow(sobely, cmap="gray"), plt.title("Sobel Y")
# plt.xticks([]), plt.yticks([])
# plt.show()

"""
Canny边缘检测算法:Canny算子是先平滑后求导数的方法。
John Canny研究了最优边缘检测方法所需的特性，给出了评价边缘检测性能优劣的三个指标：
    1、好的信噪比，即将非边缘点判定为边缘点的概率要低，将边缘点判为非边缘点的概率要低；
    2、高的定位性能，即检测出的边缘点要尽可能在实际边缘的中心；
    3、对单一边缘仅有唯一响应，即单个边缘产生多个响应的概率要低，并且虚假响应边缘应该得到最大抑制。
Canny边缘检测算法函数：
cv2.Canny(image, th1, th2，Size)
    - image：源图像
    - th1：阈值1
    - th2：阈值2
    - Size：可选参数，Sobel算子的大小
步骤：
    1. 彩色图像转换为灰度图像（以灰度图单通道图读入）
    2. 对图像进行高斯模糊（去噪）
    3. 计算图像梯度，根据梯度计算图像边缘幅值与角度
    4. 沿梯度方向进行非极大值抑制（边缘细化）
    5. 双阈值边缘连接处理
    6. 二值化图像输出结果
"""
# 以灰度图形式读入图像
img = cv2.imread(r"./images/canny.png", 0)
# Canny边缘提取
v1 = cv2.Canny(img, 80, 150, (3, 3))
v2  =cv2.Canny(img, 50, 100, (5, 5))
# 显示
ret = np.hstack((v1, v2))
cv2.imshow("img", ret)
cv2.waitKey(0)
cv2.destroyAllWindows()








