# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   yykzhjh@163.com
@DateTime :   2022/10/17 12:07
@Version  :   1.0
@License  :   (C)Copyright 2022
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt



"""
1. 方向梯度直方图（Histogram of Oriented Gradient, HOG）特征是一种在计算机视觉和图像处理中用来进行物体检测的特征描述子。
它通过计算和统计图像局部区域的梯度方向直方图来构成特征。Hog特征结合SVM分类器已经被广泛应用于图像识别中，尤其在行人检测中获得了极大的成功。
2. 主要思想：在一副图像中，目标的形状能够被梯度或边缘的方向密度分布很好地描述。
3. HOG特征实现过程：
    1）灰度化（将图像看做一个x,y,z（灰度）的三维图像）；
    2）采用Gamma校正法对输入图像进行颜色空间的标准化（归一化）；；
    3）计算图像每个像素的梯度（包括大小和方向）；
    4）将图像划分成小cells；
    5）统计每个cell的梯度直方图（不同梯度的个数），得到cell的描述子；
    6）将每几个cell组成一个block，得到block的描述子；
    7）将图像image内的所有block的HOG特征descriptor串联起来就可以得到HOG特征，该特征向量就是用来目标检测或分类的特征。
"""
# 判断矩形o是否完全包含在矩形i中
def is_inside(o, i):
    ox, oy, ow, oh = o
    ix, iy, iw, ih = i
    return ox > ix and oy > iy and ox + ow < ix + iw and oy + oh < iy + ih

# 对人体绘制颜色框
def draw_person(image, person):
    x, y, w, h = person
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

def main():
    # 读取图像
    img = cv2.imread(r"./images/people.jpg")
    # 启动检测器对象
    hog = cv2.HOGDescriptor()
    # 指定检测器类型为人体
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    # 检测图像, 图像、支持向量机参数(距离超平面的距离)、步长
    found, w = hog.detectMultiScale(img, 0.1, (1, 1))

    # 丢弃被其它矩形完全包含的矩形
    found_filtered = []
    for ri, r in enumerate(found):
        for qi, q in enumerate(found):
            if ri != qi and is_inside(r, q):
                break
        else:
            found_filtered.append(r)

    # 对最终的目标框进行颜色框定
    for person in found_filtered:
        draw_person(img, person)

    cv2.imshow("people detection", img)
    cv2.waitKey(0)
    cv2.destroyWindow()



if __name__ == '__main__':
    main()


