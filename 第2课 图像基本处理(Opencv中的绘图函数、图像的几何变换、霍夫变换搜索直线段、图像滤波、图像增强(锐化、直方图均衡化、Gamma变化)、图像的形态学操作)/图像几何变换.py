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
    图像旋转：以图像的中心为原点旋转
    注意以下两点：
    1）图像旋转之前，为了避免信息的丢失，一定要有坐标平移
    2）图像旋转之后，会出现许多空洞点。对这些空洞点必须进行填充处理，否则画面效果不好，一般也称这种操作为插值处理
    获得旋转放射变换矩阵函数：
    cv2.getRotationMatrix2D(图片的旋转中心, 旋转角度, 缩放比例(0.5表示缩小一半,正为逆时针，负值为顺时针))
    仿射变换函数：
    cv2.warpAffine(img, M, (cols, rows))
    """
    # # 读取一张图像
    # img = cv2.imread(r"./images/img2.png", 1)
    # # 获取图像的高和宽
    # h, w = img.shape[:2]
    #
    # # 参数1：旋转中心；参数2：旋转角度；参数三3：缩放因子，正值为逆时针，负值为正时针
    # M = cv2.getRotationMatrix2D((w/2, h/2), 45, 1)
    #
    # # 注意这里rows和cols需要反置，即先列后行
    # res = cv2.warpAffine(img, M, (w, h), borderValue=(25, 255, 255))
    # cv2.imshow("origin image", img)
    # cv2.imshow("rotate image", res)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    """
    仿射变换：对图像进行旋转、平移、缩放等操作以达到数据增强的效果
    线性变换从几何直观上来看有两个要点：
    1）变换前是直线，变换后依然是直线
    2）直线的比例保持不变
    仿射变换有：平移、旋转、缩放、剪切、反射(投影)
    获取仿射变换矩阵函数：
    M = cv2.getAffineTransform(pos1, pos2)
        - pos1表示变换前的位置
        - pos2表示变换后的位置
    仿射变换函数：
    cv2.warpAffine(img, M, (cols, rows))
    """
    # # 读取图像
    # src = cv2.imread(r"./images/bird.png")
    # # 获取图像大小
    # h, w = src.shape[:2]
    # # 设置图像仿射变换矩阵
    # pos1 = np.float32([[50, 50], [200, 50], [50, 200]])
    # pos2 = np.float32([[10, 100], [200, 50], [100, 250]])
    # M = cv2.getAffineTransform(pos1, pos2)
    # # 图像仿射变换
    # result = cv2.warpAffine(src, M, (2*w, 2*h))
    # cv2.imshow("origin image", src)
    # cv2.imshow("affine image", result)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    """
    透视变换：本质是将图像投影到一个新的平面
    获取透视变换矩阵函数：
    M = cv2.getPerspectiveTransform(pos1, pos2)
        - pos1表示透视变换前的4个点对应位置
        - pos2表示透视变换后的4个点对应位置
    透视变换函数：
    cv2.warpPerspective(src, M, (cols, rows))
        - src表示原始图像
        - M表示透视变换矩阵
        - (cols, rows)表示变换后的图像大小，cols表示列数，rows表示行数
    """
    # 读取图像
    src = cv2.imread(r"./images/bird.png")
    # 获取图像大小
    h, w = src.shape[:2]
    # 设置图像透视变换矩阵
    pos1 = np.float32([[114, 82], [287, 156], [8, 100], [143, 177]])
    pos2 = np.float32([[0, 0], [188, 0], [0, 262], [188, 262]])
    M = cv2.getPerspectiveTransform(pos1, pos2)
    # 图像透视变换
    result = cv2.warpPerspective(src, M, (2*w, 2*h))
    cv2.imshow("origin image", src)
    cv2.imshow("perspective image", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()




