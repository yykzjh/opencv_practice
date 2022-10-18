# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   yykzhjh@163.com
@DateTime :   2022/10/18 15:34
@Version  :   1.0
@License  :   (C)Copyright 2022
"""
import cv2
import numpy as np



"""
1. 开启摄像头函数：
cv2.VideoCapture()
• 参数说明：0,1代表电脑摄像头，或视频文件路径
2. 按帧读取视频函数：
ret, frame = cap.read()
    ret：返回布尔值True/False,如果读取帧是正确的则返回True，如果文件读取到结尾，它的返回值就为False；
    frame：每一帧的图像，是个三维矩阵
"""
# # 打开摄像头
# cap = cv2.VideoCapture(0)
# print(cap)
# while True:
#     # 获取一帧图像
#     ret, frame = cap.read()
#     # 转化为灰度图
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     cv2.imshow("frame", gray)
#     # 按下“q”键停止
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# cap.release()
# cv2.destroyAllWindows()

"""
播放、保存视频：
1. 指定写入视频帧编码格式函数：
fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
2. 创建VideoWriter对象函数：
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640,480))
    - 参数1：保存视频路径+名字；
    - 参数2：FourCC 为4 字节码，确定视频的编码格式；
    - 参数3：播放帧率
    - 参数4：大小，宽高
    - 参数5：默认为True,彩色图

"""
# # 调用摄像头
# cap = cv2.VideoCapture(0)
# # 创建编码方式
# # mp4:'X','V','I','D'; avi:'M','J','P','G' 或 'P','I','M','1'; flv:'F','L','V','1'
# fourcc = cv2.VideoWriter_fourcc('F', 'L', 'V', '1')
# # 创建VideoWriter对象，路径、编码方式、帧率、宽高
# out = cv2.VideoWriter(r"./videos/output_1.flv", fourcc, 20.0, (640, 480))
# # 创建循环结构进行连续读写
# while cap.isOpened():
#     ret, frame = cap.read()
#     if ret == True:
#         out.write(frame)
#         cv2.imshow('frame', frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#     else:
#         break
# cap.release()
# out.release()
# cv2.destroyAllWindows()

"""
修改视频格式
"""
# 打开视频
cap = cv2.VideoCapture(r"./videos/move_detect.flv")
# 创建编码方式
fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
# 获取视频帧率
fps = cap.get(cv2.CAP_PROP_FPS)
print(fps)

# 视频图像的宽、高
frame_width, frame_height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(frame_width, frame_height)
# 创建VideoWriter对象
out = cv2.VideoWriter(r"./videos/move_detect_new.avi", fourcc, fps, (frame_width, frame_height))
# 循环转换
while True:
    ret, frame = cap.read()
    if ret == True:
        # 水平翻转
        frame = cv2.flip(frame, 1)
        # 写入
        out.write(frame)
        cv2.imshow("frame", frame)
        if cv2.waitKey(25) & 0xff == ord('q'):
            break
    else:
        break
out.release()
cap.release()
cv2.destroyAllWindows()


























