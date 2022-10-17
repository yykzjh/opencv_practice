# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   yykzhjh@163.com
@DateTime :   2022/10/17 17:41
@Version  :   1.0
@License  :   (C)Copyright 2022
"""
import cv2
import numpy as np


"""
1. LBP介绍：LBP(Local Binary Pattern，局部二值模式)，是一种用来描述图像局部纹理特征的算子，它具有旋转不变性和灰度不变性等显著的优点，
作者：T.Ojala、M.Pietikäinen和D.Harwood，提出时间：1994年
2. LBP原理：LBP算子定义在一个3×3的窗口内，以窗口中心像素为阈值，与相邻的8个像素的灰度值比较，若周围的像素值大于中心像素值，
则该位置被标记为1，否则标记为0。如此可以得到一个8位二进制数(通常还要转换为10进制，即LBP码，共256种)，将这个值作为窗口中心像素点的LBP值，
以此来反应这个3×3区域的纹理信息。
"""
# def LBP(src):
#     """
#     自己实现LBP
#     :param src: 灰度图像
#     :return:
#     """
#     # 获取图像维度信息
#     h, w = src.shape
#     # 拷贝一份图像
#     dst = src.copy()
#     # 遍历灰度图像素
#     for i in range(1, h - 1):
#         for j in range(1, w - 1):
#             # 初始化数据结构
#             lbp_value = np.zeros((1, 8), dtype=np.uint8)
#             neighbours = np.zeros((1, 8), dtype=np.uint8)
#             # 依次存储邻域灰度值
#             neighbours[0, 0] = src[i - 1, j - 1]
#             neighbours[0, 1] = src[i - 1, j]
#             neighbours[0, 2] = src[i - 1, j + 1]
#             neighbours[0, 3] = src[i, j - 1]
#             neighbours[0, 4] = src[i, j + 1]
#             neighbours[0, 5] = src[i + 1, j - 1]
#             neighbours[0, 6] = src[i + 1, j]
#             neighbours[0, 7] = src[i + 1, j + 1]
#             # 获取中心点的阈值
#             center = src[i, j]
#             # 得到lbp值
#             lbp_value[neighbours > center] = 1
#             # 计算lbp十进制值
#             lbp = lbp_value[0, 0] * 1 + lbp_value[0, 1] * 2 + lbp_value[0, 2] * 4 + lbp_value[0, 3] * 8 \
#                   + lbp_value[0, 4] * 16 + lbp_value[0, 5] * 32 + lbp_value[0, 6] * 64 + lbp_value[0, 7] * 128
#             # 用当前像素邻域计算出的lbp值更新dst
#             dst[i, j] = lbp
#     return dst


def LBP(src):
    '''
    :param src:灰度图像
    :return:
    '''
    height = src.shape[0]
    width = src.shape[1]
    dst = src.copy()
    lbp_value = np.zeros((1, 8), dtype=np.uint8)
    # print(lbp_value)
    neighbours = np.zeros((1, 8), dtype=np.uint8)
    # print(neighbours)
    for x in range(1, width - 1):
        for y in range(1, height - 1):
            neighbours[0, 0] = src[y - 1, x - 1]
            neighbours[0, 1] = src[y - 1, x]
            neighbours[0, 2] = src[y - 1, x + 1]
            neighbours[0, 3] = src[y, x - 1]
            neighbours[0, 4] = src[y, x + 1]
            neighbours[0, 5] = src[y + 1, x - 1]
            neighbours[0, 6] = src[y + 1, x]
            neighbours[0, 7] = src[y + 1, x + 1]
            center = src[y, x]
            for i in range(8):
                if neighbours[0, i] > center:
                    lbp_value[0, i] = 1
                else:
                    lbp_value[0, i] = 0

            lbp = lbp_value[0, 0] * 1 + lbp_value[0, 1] * 2 + lbp_value[0, 2] * 4 + lbp_value[0, 3] * 8 \
                  + lbp_value[0, 4] * 16 + lbp_value[0, 5] * 32 + lbp_value[0, 6] * 64 + lbp_value[0, 7] * 128

            # print(lbp)
            dst[y, x] = lbp

    return dst




def main():
    # 读取图像
    img = cv2.imread(r"./images/people.jpg", 0)
    print(img)
    # 显示原图灰度图
    cv2.imshow("src", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 计算LBP特征图
    new_img = LBP(img)
    print(new_img)
    # 显示LBP特征图
    cv2.imshow("dst", new_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()




if __name__ == '__main__':
    main()




