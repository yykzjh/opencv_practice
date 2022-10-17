# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   yykzhjh@163.com
@DateTime :   2022/10/17 17:23
@Version  :   1.0
@License  :   (C)Copyright 2022
"""
import cv2
import numpy as np


"""
1. SIFT算法：SIFT，即尺度不变特征变换算法（Scale-invariant feature transform，SIFT），
是用于图像处理领域的一种算法。SIFT具有尺度不变性，可在图像中检测出关键点，是一种局部特征描述子。
其应用范围包含物体辨识、机器人地图感知与导航、影像缝合、3D模型建立、手势辨识、影像追踪和动作比对。
2. SIFT特性：
    1）独特性，也就是特征点可分辨性高，类似指纹，适合在海量数据中匹配；
    2）多量性，提供的特征多；
    3）高速性，就是速度快；
    4）可扩展，能与其他特征向量联合使用；
3. SIFT特点：
    1）旋转、缩放、平移不变性；
    2）解决图像仿射变换，投影变换的关键的匹配；
    3）光照影响小；
    4）目标遮挡影响小；
    5）噪声景物影响小；
4. SIFT算法步骤：
    1）尺度空间极值检测点检测；
    2）关键点定位：去除一些不好的特征点，保存下来的特征点能够满足稳定性等条件；
    3）关键点方向参数：获取关键点所在尺度空间的邻域，然后计算该区域的梯度和方向，根据计算得到的结果创建方向直方图，直方图的峰值为主方向的参数；
    4）关键点描述符：每个关键点用一组向量（位置、尺度、方向）将这个关键点描述出来，使其不随着光照、视角等等影响而改变；
    5）关键点匹配：分别对模板图和实时图建立关键点描述符集合，通过对比关键点描述符来判断两个关键点是否相同；
5. SIFT代码实现：
    - 返回的关键点是一个带有很多不用属性的特殊结构体，属性当中有坐标，方向、角度等。
    - 使用sift.compute()函数来进行计算关键点描述符， kp, des = sift.compute(gray, kp)
    - 如果未找到关键点，可使用函数，sift.detectAndCompute()直接找到关键点并计算。
    - 在第二个函数中，kp为关键点列表，des为numpy的数组，为关键点数目×128
"""
# 读取图像
img = cv2.imread(r"./images/harris2.png")
# 转换为灰度图
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 定义SIFT提取器
sift = cv2.xfeatures2d.SIFT_create()
# 找到关键点
kp = sift.detect(gray, None)
print(kp)
kp, des = sift.compute(gray, kp)
print(len(kp))
print(des.shape)
# 绘制关键点
img = cv2.drawKeypoints(gray, kp, img)

# 显示
cv2.imshow("sp", img)
cv2.waitKey(0)
cv2.destroyWindow()





