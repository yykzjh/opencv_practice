# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   yykzhjh@163.com
@DateTime :   2022/10/11 0:02
@Version  :   1.0
@License  :   (C)Copyright 2022
"""
import cv2
import numpy as np


if __name__ == '__main__':
    """
    绘制线段
    cv2.line(img, pts, color, thickness, lineType)
    """
    # # 创建一张黑色的背景图
    # img = np.zeros((512, 512, 3), dtype=np.uint8)
    #
    # # 绘制一条线宽为5的线段,图像、起点、终点、线段颜色、线宽(-1会填充)、线条的类型(cv2.LINE_AA、默认为8型)
    # cv2.line(img, (0, 0), (200, 500), (0, 0, 255), 5)
    # cv2.imshow("line", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    """
    绘制矩形
    cv2.rectangle(img, pts, color, thickness, lineType)
    """
    # # 创建一张黑色的背景图
    # img = np.zeros((512, 512, 3), dtype=np.uint8)
    #
    # # 画一个绿色边框的矩形，参数2：左上角坐标，参数3：右下角坐标
    # cv2.rectangle(img, (384, 0), (510, 128), (0, 255, 255), -1)
    # cv2.imshow("rectangle", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # """
    # 绘制圆
    # cv2.circle(img, pts, radius, color, thickness, lineType)
    # """
    # # 创建一张黑色的背景图
    # img = np.zeros((512, 512, 3), dtype=np.uint8)
    #
    # # 画一个填充红色的圆， 参数2：圆形坐标，参数3：半径
    # cv2.circle(img, (447, 63), 63, (0, 0, 255), -1)
    # cv2.imshow("circle", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    """
    绘制椭圆
    cv2.ellipse(img, 中心点坐标, (x轴长度,y轴长度), 椭圆的旋转角度(逆时针), 椭圆的起始角度(逆时针), 椭圆的结束角度(逆时针), color, thickness, lineType)
    """
    # # 创建一张黑色的背景图
    # img = np.zeros((512, 512, 3), dtype=np.uint8)
    #
    # # 画一个填充的半椭圆
    # cv2.ellipse(img, (256, 256), (100, 50), 0, 30, 180, (255, 0, 0), -1)
    # cv2.imshow("ellipse", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    """
    绘制多边形
    cv2.polylines(img, pts, isClosed, color, thickness, lineType)
    """
    # # 创建一张黑色的背景图
    # img = np.zeros((512, 512, 3), dtype=np.uint8)
    #
    # # 定义四个顶点坐标
    # pts = np.array([[10, 5], [50, 10], [70, 20], [20, 30]])
    # # 顶点个数：4， 矩阵变成4*1*2维
    # pts = pts.reshape((-1, 1, 2))
    #
    # # 绘制一个封闭的多边形
    # cv2.polylines(img, [pts], True, (0, 255, 255))
    # cv2.imshow("polygon", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    """
    添加文字
    cv2.putText(img, 要添加的文字, 文字的起始坐标(左下角为起点(宽,高)), 字体, 文字大小(缩放比例), 颜色, 线条宽度, 线条形状)
    """
    # 创建一张黑色的背景图
    img = np.zeros((512, 512, 3), dtype=np.uint8)

    # 添加文字
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, "OpenCV", (80, 280), font, 3, (0, 255, 255), 5, cv2.LINE_AA)
    cv2.imshow("polygon", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()









