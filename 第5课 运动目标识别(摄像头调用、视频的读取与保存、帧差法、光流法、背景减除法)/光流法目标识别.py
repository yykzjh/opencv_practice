# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   yykzhjh@163.com
@DateTime :   2022/10/18 16:57
@Version  :   1.0
@License  :   (C)Copyright 2022
"""
import cv2
import numpy as np


"""
1. 光流法利用图像序列中像素在时间域上的变化以及相邻帧之间的相关性，根据上一帧与当前帧之间的对应关系，计算得到相邻帧之间物体的运动信息。
大多数的光流计算方法计算量巨大，结构复杂，且易受光照、物体遮挡或图像噪声的影响，鲁棒性差，故一般不被对精度和实时性要求比较高的监控系统所采用。
2. 光流是基于以下假设的：
    - 在连续的两帧图像之间（目标对象的）像素的灰度值不改变。
    - 相邻的像素具有相同的运动
"""
# 定义两个结构元素
es = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
# 打开视频
cap = cv2.VideoCapture(r"./videos/move_detect.flv")
# 获取第一帧图像
_, frame1 = cap.read()
# 转化为灰度图
prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
# 初始化一个hsv色彩空间的图像
hsv = np.zeros_like(frame1)
# 将第2个通道的值置为255
hsv[..., 1] = 255

# 视频文件输出参数设置
out_fps = 12.0
fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', '2')
sizes = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
out1 = cv2.VideoWriter(r"./videos/v6.avi", fourcc, out_fps, sizes)
out2 = cv2.VideoWriter(r"./videos/v8.avi", fourcc, out_fps, sizes)

# 循环遍历视频的每一帧
while cap.isOpened():
    # 获取当前帧
    ret, frame2 = cap.read()
    # 视频结尾，跳出循环
    if not ret:
        break
    # 转为灰度图
    gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    # 调用光流法函数计算，输出是一些坐标点
    flow = cv2.calcOpticalFlowFarneback(prvs, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    print(flow[..., 1])
    # 得到梯度幅度和梯度角度阵列
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    # 将梯度角度赋值给hsv的色环值
    hsv[..., 0] = ang * 180 / np.pi / 2
    # 将梯度幅度赋给hsv的亮度值
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

    # 将hsv色彩空间转换为bgr
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # 转化为灰度图
    draw = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    # 开运算
    draw = cv2.morphologyEx(draw, cv2.MORPH_OPEN, kernel)
    # 二值化
    _, draw = cv2.threshold(draw, 25, 255, cv2.THRESH_BINARY)

    # 搜索轮廓
    contours, hierarchy = cv2.findContours(draw.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 遍历所有连通的轮廓
    for c in contours:
        if cv2.contourArea(c) < 50:
            continue
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(frame2, (x, y), (x + w, y + h), (255, 255, 0), 2)

    # 显示
    cv2.imshow("frame2", bgr)
    cv2.imshow("draw", draw)
    cv2.imshow("frame1", frame2)

    # 存储
    out1.write(bgr)
    out2.write(frame2)

    # 如果q键被按下，跳出循环
    if cv2.waitKey(20) & 0xff == ord('q'):
        break

    prvs = gray

# 清理资源并关闭打开的窗口
out1.release()
out2.release()
cap.release()
cv2.destroyAllWindows()





















