# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   yykzhjh@163.com
@DateTime :   2022/10/13 21:31
@Version  :   1.0
@License  :   (C)Copyright 2022
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt


def rectify_document_image(path):
    # 读取图片
    src = cv2.imread(r"./images/paper.png")

    # 获取图像大小
    h, w = src.shape[:2]

    # 将图像进行高斯模糊去噪声
    img = cv2.GaussianBlur(src, (3, 3), 0)
    # 进行灰度化处理
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 边缘检测（检测出图像的边缘信息）,apertureSize为核大小
    edges = cv2.Canny(gray, 50, 250, apertureSize=3)
    cv2.imshow("canny", edges)

    # 通过霍夫变换得到A4纸边缘，霍夫变换能检测出边缘图像中的线段，并以两个端点的坐标表示每条线段
    """
    image： 必须是二值图像，推荐使用canny边缘检测的结果图像； 
    rho: 线段以像素为单位的距离精度，double类型的，推荐用1.0 
    theta： 线段以弧度为单位的角度精度，推荐用numpy.pi/180 
    threshod: 累加平面的阈值参数，int类型，超过设定阈值才被检测出线段，值越大，基本上意味着检出的线段越长，检出的线段个数越少。根据情况推荐先用100试试
    lines：这个参数的意义未知，发现不同的lines对结果没影响，但是不要忽略了它的存在 
    minLineLength：线段以像素为单位的最小长度，根据应用场景设置 
    maxLineGap：同一方向上两条线段判定为一条线段的最大允许间隔（断裂），超过了设定值，则把两条线段当成一条线段，值越大，允许线段上的断裂越大，越有可能检出潜在的直线段
    """
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 10, minLineLength=90, maxLineGap=20)
    print(lines)

    # 找到文档的四个顶点
    x1, y1, x2, y2 = lines[2][0]
    x3, y3, x4, y4 = lines[0][0]

    # 确定透视变换前后对应的4个顶点
    pos1 = np.float32([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
    pos2 = np.float32([[0, 0], [188, 0], [0, 262], [188, 262]])
    # 获取透视变换矩阵
    M = cv2.getPerspectiveTransform(pos1, pos2)

    # 对原图进行透视变换
    result = cv2.warpPerspective(src, M, (190, 264))


    # 在原图图上画出两条直线
    cv2.line(src, (x1, y1), (x2, y2), (0, 0, 255), 1)
    cv2.line(src, (x3, y3), (x4, y4), (0, 0, 255), 1)
    # for i in range(2, len(lines)):
    #     tmpx1, tmpy1, tmpx2, tmpy2 = lines[i][0]
    #     cv2.line(src, (tmpx1, tmpy1), (tmpx2, tmpy2), (255, 0, 0), 1)
    # 显示图像
    cv2.imshow("origin image", src)
    cv2.imshow("result image", result)

    cv2.waitKey(0)
    cv2.destroyAllWindows()








if __name__ == '__main__':
    rectify_document_image(r"./images/paper.png")






