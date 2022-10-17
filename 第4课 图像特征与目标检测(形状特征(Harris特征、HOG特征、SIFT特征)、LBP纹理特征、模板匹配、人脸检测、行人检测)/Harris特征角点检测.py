# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   yykzhjh@163.com
@DateTime :   2022/10/17 16:56
@Version  :   1.0
@License  :   (C)Copyright 2022
"""
import cv2
import numpy as np



"""
1. 角点：在现实世界中，角点对应于物体的拐角，道路的十字路口、丁字路口等。
从图像分析的角度来定义角点可以有以下两种定义：
    1）角点可以是两个边缘的交点；
    2）角点是邻域内具有两个主方向的特征点；
2. 角点计算方法：
    1）前者通过图像边缘计算，计算量大，图像局部变化会对结果产生较大的影响；
    2）后者基于图像灰度的方法通过计算点的曲率及梯度来检测角点；
3. 角点所具有的特征：
    1）轮廓之间的交点；
    2）对于同一场景，即使视角发生变化，通常具备稳定性质的特征；
    3）该点附近区域的像素点无论在梯度方向上还是其梯度幅值上有着较大变化；
4. 性能较好的角点：
    1）检测出图像中“真实”的角点
    2）准确的定位性能
    3）很高的重复检测率
    4）噪声的鲁棒性
    5）较高的计算效率
5. Harris实现过程：
    1）计算图像在X和Y方向的梯度；
    2）计算图像两个方向梯度的乘积；
    3）使用高斯函数对三者进行高斯加权，生成矩阵M的A,B,C；
    4）计算每个像素的Harris响应值R，并对小于某一阈值t的R置为零；
    5）在3×3或5×5的邻域内进行非最大值抑制，局部最大值点即为图像中的角点；
6. Harris函数：
cv2.cornerHarris(img, blockSize, ksize, k)
    - img：数据类型为float32的输入图像
    - blockSize：角点检测中要考虑的领域大小
    - ksize：Sobel求导中使用的窗口大小
    - k：Harris 角点检测方程中的自由参数,取值参数为[0.04, 0.06]
"""
# 读取图像
img = cv2.imread(r"./images/harris2.png")
# 转换为灰度图
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)

# Harris角点检测，输入图像必须是float32，最后一个参数在0.04 ~ 0.06之间
dst = cv2.cornerHarris(gray, 2, 3, 0.06)

# 结果进行膨胀，可有可无
dst = cv2.dilate(dst, None)
print(dst)

# 设定阈值，不同图像阈值不同
img[dst > 0.01 * dst.max()] = [0, 0, 255]
print(dst.max())

# 展示
cv2.imshow("dst_img", img)
cv2.waitKey(0)
cv2.destroyWindow()


