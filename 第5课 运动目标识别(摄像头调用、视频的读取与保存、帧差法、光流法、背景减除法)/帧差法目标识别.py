# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   yykzhjh@163.com
@DateTime :   2022/10/18 16:16
@Version  :   1.0
@License  :   (C)Copyright 2022
"""
import cv2
import numpy as np



"""
1. 帧间差分法是通过对视频中相邻两帧图像做差分运算来标记运动物体的方法。当视频中存在移动物体的时候，相邻帧（或相邻三帧）之间在灰度上会有差别，
求取两帧图像灰度差的绝对值，则静止的物体在差值图像上表现出来全是0，而移动物体特别是移动物体的轮廓处由于存在灰度变化为非0
2. 优点：
    - 算法实现简单，程序设计复杂度低；
    - 对光线等场景变化不太敏感，能够适应各种动态环境，稳定性较好；
3. 缺点：
    - 不能提取出对象的完整区域，对象内部有“空洞”；
    - 只能提取出边界，边界轮廓比较粗，往往比实际物体要大；
    - 对快速运动的物体，容易出现糊影的现象，甚至会被检测为两个不同的运动物体；
    - 对慢速运动的物体，当物体在前后两帧中几乎完全重叠时，则检测不到物体；
4. 搜索轮廓函数：
contours, hierarchy = cv2.findContours(图像, 输出轮廓的组织形式, 轮廓的近似方法)
    - 参数1：带有轮廓信息的图像
    - 参数2：提取轮廓后，输出轮廓信息的组织形式
        - cv2.RETR_EXTERNAL：输出轮廓中只有外侧轮廓信息；
        - cv2.RETR_LIST：以列表形式输出轮廓信息，各轮廓之间无等级关系；
        - cv2.RETR_CCOMP：输出两层轮廓信息，即内外两个边界（下面将会说到contours的数据结构）；
        - cv2.RETR_TREE：以树形结构输出轮廓信息
    - 参数3：指定轮廓的近似办法
        - cv2.CHAIN_APPROX_NONE：存储轮廓所有点的信息，相邻两个轮廓点在图象上也是相邻的；
        - cv2.CHAIN_APPROX_SIMPLE：压缩水平方向，垂直方向，对角线方向的元素，只保留该方向的终点坐标；
        - cv2.CHAIN_APPROX_TC89_L1：使用teh-Chinl chain 近似算法保存轮廓信息
    - 返回值：
        - contours：list结构，列表中每个元素代表一个边沿信息。每个元素是(x,1,2)的三维向量，x表示该条边沿里共有多少个像素点，
        第三维的那个“2”表示每个点的横、纵坐标；注意：如果输入选择cv2.CHAIN_APPROX_SIMPLE，
        则contours中一个list元素所包含的x点之间应该用直线连接起来，这个可以用cv2.drawContours()函数观察一下效果。
        - hierarchy：返回类型是(x,4)的二维ndarray。x和contours里的x是一样的意思。如果输入选择cv2.RETR_TREE，则以树形结构组织输出，
        hierarchy的四列分别对应下一个轮廓编号、上一个轮廓编号、父轮廓编号、子轮廓编号，该值为负数表示没有对应项。
    - 求输出轮廓的长度和面积：
        - cv2.arcLength(contours[i], False)
        - cv2.contourArea(contours[i])
"""
# 打开视频文件
camera = cv2.VideoCapture(r"./videos/move_detect.flv")

# 视频文件输出参数设置
out_fps = 12.0  # 输出文件的帧率
fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', '2')
out1 = cv2.VideoWriter(r"./videos/v1.avi", fourcc, out_fps, (500, 400))
out2 = cv2.VideoWriter(r"./videos/v2.avi", fourcc, out_fps, (500, 400))

# 初始化当前帧的前帧
last_frame = None

# 遍历视频的每一帧
while camera.isOpened():
    # 读取一帧
    ret, frame = camera.read()
    # 判断是否到达了视频的结尾
    if not ret:
        break
    # 调整该帧的大小
    frame = cv2.resize(frame, (500, 400), interpolation=cv2.INTER_CUBIC)
    # 如果第一帧是None，对其进行初始化
    if last_frame is None:
        last_frame = frame
        continue
    # 计算当前帧和前一帧的不同
    frame_delta = cv2.absdiff(last_frame, frame)
    # 更新当前帧为新的前帧
    last_frame = frame.copy()
    # 结果转为灰度图
    thresh = cv2.cvtColor(frame_delta, cv2.COLOR_BGR2GRAY)
    # 图像二值化
    _, thresh = cv2.threshold(thresh, 25, 255, cv2.THRESH_BINARY)

    """
    去除图像噪声，先腐蚀再膨胀(形态学开运算)
    thresh = cv2.erode(thresh, None, iterations=1)
    thresh = cv2.dilate(thresh, None, iterations=2)
    """
    # 找到阈值图像上的轮廓位置
    cnts, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 遍历轮廓
    for c in cnts:
        # 忽略小轮廓，排除误差
        if cv2.contourArea(c) < 50:
            continue
        # 计算轮廓的边界框，在当前帧中画出该框
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # 显示当前帧
    cv2.imshow("frame", frame)
    cv2.imshow("frame_delta", frame_delta)
    cv2.imshow("thresh", thresh)

    # 保存视频
    out1.write(frame)
    out2.write(frame_delta)

    # 如果q键被按下，跳出循环
    if cv2.waitKey(20) & 0xff == ord('q'):
        break

# 清理资源并关闭打开的窗口
out1.release()
out2.release()
camera.release()
cv2.destroyAllWindows()


















