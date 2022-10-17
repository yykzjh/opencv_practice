# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   yykzhjh@163.com
@DateTime :   2022/10/15 19:00
@Version  :   1.0
@License  :   (C)Copyright 2022
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt


"""
1. 连通区域（Connected Component）一般是指图像中具有相同像素值且位置相邻的前景像素点组成的图像区域，
2. 连通区域分析是指将图像中的各个连通区域找出并标记。连通区域分析是一种在CV和图像分析处理的众多应用领域中较为常用和基本的方法。
例如：OCR识别中字符分割提取（车牌识别、文本识别、字幕识别等）、视觉跟踪中的运动前景目标分割与提取（行人入侵检测、遗留物体检测、
基于视觉的车辆检测与跟踪等）、医学图像处理（感兴趣目标区域提取）等。在需要将前景目标提取出来以便后续进行处理的应用场景中都能够用到连通区域分析方法，
通常连通区域分析处理的对象是一张二值化后的图像。
3. 两遍扫描法(Two-Pass),正如其名，指的就是通过扫描两遍图像，将图像中存在的所有连通域找出并标记。
第一次扫描：
    - 从左上角开始遍历像素点，找到第一个像素为255的点，label=1;
    - 当该像素的左邻像素和上邻像素为无效值时，给该像素置一个新的label值，label++;
    - 当该像素的左邻像素或者上邻像素有一个为有效值时，将有效值像素的labal赋给该像素的label值;
    - 当该像素的左邻像素和上邻像素都为有效值时，选取其中较小的labal值赋给该像素的label值
第二次扫描：
    - 对每个点的label进行更新，更新为对于其连通集合中最小的label
4. 区域生长算法：区域生长是一种串行区域分割的图像分割方法。
区域生长是指从某个像素出发，按照一定的准则，逐步加入邻近像素，当满足一定的条件时，区域生长终止。
区域生长的好坏决定于：
    1）初始点（种子点）的选取
    2）生长准则
    3）终止条件
区域生长是从某个或者某些像素点出发，最后得到整个区域，进而实现目标的提取。
区域生长的原理：将具有相似性质的像素集合起来构成区域。
算法步骤(BFS)：
    1. 对图像顺序扫描，找到第1个还没有归属的像素, 设该像素为(x0, y0);
    2. 以(x0, y0)为中心, 考虑(x0, y0)的4邻域像素(x, y)如果(x0, y0)满足生长准则, 
       将(x, y)与(x0, y0)合并(在同一区域内), 同时将(x, y)压入堆栈;
    3. 从堆栈中取出一个像素, 把它当作(x0, y0)返回到步骤2;
    4. 当堆栈为空时，返回到步骤1;
    5. 重复步骤1 - 4直到图像中的每个点都有归属时，生长结束。
5. 分水岭算法：任意的灰度图像可以被看做是地质学表面，高亮度的地方是山峰，低亮度的地方是山谷。
给每个孤立的山谷(局部最小值)不同颜色的水(标签)，当水涨起来，根据周围的山峰(梯度)，不同的山谷也就是不同的颜色会开始合并，
要避免山谷合并，需要在水要合并的地方建立分水岭，直到所有山峰都被淹没，所创建的分水岭就是分割边界线，这个就是分水岭的原理。
算法步骤：
    1. 将白色背景编程黑色背景 - 目的是为了后面变的变换做准备
    2. 使用filter2D与拉普拉斯算子实现图像对比度的提高
    3. 转为二值图像4. 距离变换
    5. 对距离变换结果进行归一化[0-1]之间
    6. 使用阈值，在此二值化，得到标记
    7. 腐蚀每个peak erode
    8. 发现轮廓 findContours
    9. 绘制轮廓 drawContours
    10.分水岭变换 watershed
    11.对每个分割区域着色输出结果
"""

# Step1. 加载图像
img = cv2.imread(r"./images/yezi.jpg")
cv2.imshow("img", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
# 转化为灰度图
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Step2. 阈值分割，将图像分为黑白两部分
_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Step3. 对图像进行“开运算”，先腐蚀再膨胀
kernel = np.ones((3, 3), dtype=np.uint8)
# 执行开运算
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

# Step4. 对“开运算”的结果进行膨胀，得到大部分都是背景的区域
sure_bg = cv2.dilate(opening, kernel, iterations=3)
cv2.imshow("sure_bg", sure_bg)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Step5. 通过distanceTransform获取前景区域
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)  # DIST_L1 DIST_C只能 对应掩膜为3   DIST_L2 可以为3或者5

print(dist_transform.max())
_, sure_fg  =cv2.threshold(dist_transform, 0.1 * dist_transform.max(), 255, 0)

cv2.imshow("sure_fg", sure_fg)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Step6. sure_bg与sure_fg相减，得到既有前景又有背景的重合区域，此区域和轮廓区域的关系未知
sure_fg = np.uint8(sure_fg)
unknow = cv2.subtract(sure_bg, sure_fg)
cv2.imshow("unknow", unknow)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Step7. 连通区域处理
ret, markers = cv2.connectedComponents(sure_fg, connectivity=8)  # 对连通区域进行标号  序号为 0 - N-1
print(ret)

# OpenCV分水岭算法对物体做的标注必须都大于1，背景为标号为0，因此对所有markers加1，变成了1 - N
markers = markers + 1
# 去掉属于背景区域的部分（即让其变为0，成为背景）
markers[unknow == 255] = 0

# Step. 分水岭算法
markers = cv2.watershed(img, markers)

# 标注为-1的像素点标红
img[markers == -1] = [0, 0, 255]
cv2.imshow("dst", img)
cv2.waitKey(0)
cv2.destroyAllWindows()




