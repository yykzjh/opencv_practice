# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   yykzhjh@163.com
@DateTime :   2022/10/12 21:22
@Version  :   1.0
@License  :   (C)Copyright 2022
"""
import cv2
import numpy as np




if __name__ == '__main__':
    """
    仿射变换函数
    cv2.warpAffine(src, M, dsize, flags, borderMode, borderValue)
    src - 输入图像
    M - 变换矩阵
    dsize - 输出图像的大小
    flags - 插值方法的组合(int类型)
    borderMode - 边界像素模式(int类型)
    borderValue - (重点)边界填充值;默认情况下为0
    上述参数中：
    M作为仿射变换矩阵，一般反映平移或旋转的关系，为InputArray类型的2X3的变换矩阵
    flags表示插值方式，默认为flags=cv2.INTER_LINEAR,表示线性插值
    此外还有：
    cv2.INTER_NEAREST (最近邻插值)
    cv2.INTER_LINEAR (线性插值)
    cv2.INTER_AREA (区域插值)
    cv2.INTER_CUBIC (三次样条插值)
    cv2.INTER_LANCZOS4 (Lanczos插值)
    """

    """
    图像平移
    """
    # # 读取一张图像
    # img = cv2.imread(r"./images/img2.png")
    # # 构造移动矩阵H,(x, y) -> (宽, 高)
    # H = np.float32([[1, 0, 50], [0, 1, 25]])
    # rows, cols = img.shape[:2]
    #
    # # 注意这里rows和cols需要反置，即先列后行
    # res = cv2.warpAffine(img, H, (2 * cols, 2 * rows))
    # cv2.imshow("origin image", img)
    # cv2.imshow("new image", res)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    """
    图像缩放
    cv2.resize(src, dsize=None, fx, fy, interpolation)
    src:原图
    dsize:输出图像尺寸，与比例因子二选一
    fx:沿水平轴的比例因子
    fy:沿垂直轴的比例因子
    interpolation:插值方法
        - cv2.INTER_NEAREST (最近邻插值)
        - cv2.INTER_LINEAR (线性插值)
        - cv2.INTER_AREA (区域插值)
        - cv2.INTER_CUBIC (三次样条插值)
        - cv2.INTER_LANCZOS4 (Lanczos插值)
    """
    # # 读取一张图像
    # img = cv2.imread(r"./images/img2.png")
    #
    # # 方法一：通过设置缩放比例来对图像进行放大或缩小
    # res1 = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    #
    # # 方法二：直接设置图像的大小，不需要缩放因子
    # h, w = img.shape[:2]
    # res2 = cv2.resize(img, (int(0.8*w), int(0.8*h)), interpolation=cv2.INTER_LANCZOS4)
    #
    # cv2.imshow("origin image", img)
    # cv2.imshow("res1 image", res1)
    # cv2.imshow("res2 image", res2)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    """
    图像旋转
    
    """



