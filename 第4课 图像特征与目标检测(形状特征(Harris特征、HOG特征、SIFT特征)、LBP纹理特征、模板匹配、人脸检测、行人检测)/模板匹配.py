# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   yykzhjh@163.com
@DateTime :   2022/10/17 18:23
@Version  :   1.0
@License  :   (C)Copyright 2022
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt


"""
1. 模板匹配介绍：模板匹配是一种最原始、最基本的模式识别方法，研究某一特定目标的图像位于图像的什么地方，进而对图像进行定位。
在待检测图像上，从左到右，从上向下计算模板图像与重叠子图像的匹配度，匹配程度越大，两者相同的可能性越大。
2. 模板匹配函数：
result = cv2.matchTemplate(image, templ, method)
    - image参数表示待搜索图像
    - templ参数表示模板图像，必须不大于源图像并具有相同的数据类型
    - method参数表示计算匹配程度的方法
        - TM_SQDIFF_NORMED是标准平方差匹配，通过计算两图之间平方差来进行匹配，最好匹配为0，匹配越差，匹配值越大
        - TM_CCORR_NORMED是标准相关性匹配，采用模板和图像间的乘法操作，数越大表示匹配程度较高，0表示最坏的匹配效果，
        这种方法排除了亮度线性变化对相似度计算的影响。
        - TM_CCOEFF_NORMED是标准相关性系数匹配，两图减去了各自的平均值之外，还要各自除以各自的方差。
        将模板对其均值的相对值与图像对其均值的相关值进行匹配，1表示完美匹配，-1表示糟糕的匹配，0表示没有任何相关性(随机序列)。
minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc()
    - minVal参数表示返回的最小值
    - maxVal参数表示返回的最大值
    - minLoc参数表示返回的最小位置
    - maxLoc参数表示返回的最大位置
"""

def template_demo(tpl, target):
    """
    模板匹配
    :param tpl: 模板
    :param target: 源图像
    :return:
    """
    # 定义匹配的方法，三种模板匹配的方法
    methods = [cv2.TM_SQDIFF_NORMED, cv2.TM_CCORR_NORMED, cv2.TM_CCOEFF_NORMED]
    # 获取模板维度信息
    th, tw = tpl.shape[:2]
    # 遍历三种模板匹配的方法
    for md in methods:
        # 模板匹配，获得所有子图像的相似度系数
        result = cv2.matchTemplate(target, tpl, md)
        print(result)
        # 获取相似度系数最大值和最小值，以及相应子图像的位置
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        print(min_val, max_val, min_loc, max_loc)
        if md == cv2.TM_SQDIFF_NORMED:
            t1 = min_loc
        else:
            t1 = max_loc
        # 计算得到矩形右下角的坐标
        br = (t1[0] + tw, t1[1] + th)
        # 画出目标框
        cv2.rectangle(target, t1, br, (0, 0, 255), 2)
        cv2.namedWindow("match-" + str(md), cv2.WINDOW_NORMAL)
        cv2.imshow("match-" + str(md), target)



def main():
    # 加载模板图像
    tpl = cv2.imread(r"./images/sample2.jpg")
    # 加载源图像
    target = cv2.imread(r"./images/target1.jpg")
    # 显示模板图和源图像
    cv2.namedWindow("template image", cv2.WINDOW_NORMAL)
    cv2.imshow("template image", tpl)
    cv2.namedWindow("target image", cv2.WINDOW_NORMAL)
    cv2.imshow("target image", target)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    template_demo(tpl, target)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



if __name__ == '__main__':
    main()



