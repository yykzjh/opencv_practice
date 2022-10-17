# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   yykzhjh@163.com
@DateTime :   2022/10/17 20:03
@Version  :   1.0
@License  :   (C)Copyright 2022
"""
import cv2
import dlib
import numpy as np


"""
1. 人脸识别概要：一般而言，一个完整的人脸识别系统包含四个主要组成部分，即人脸检测、人脸对齐、人脸特征提取以及人脸识别。
四部分流水线操作：
    - 人脸检测在图像中找到人脸的位置；
    - 人脸配准在人脸上找到眼睛、鼻子、嘴巴等面部器官的位置；
    - 通过人脸特征提取将人脸图像信息抽象为字符串信息；
    - 人脸识别将目标人脸图像与既有人脸比对计算相似度，确认人脸对应的身份。
2. 人脸检测(Face Detection)：人脸检测算法的输入是一张图片，输出是人脸框坐标序列(0个人脸框或1个人脸框或多个人脸框)。
一般情况下，输出的人脸坐标框为一个正朝上的正方形，但也有一些人脸检测技术输出的是正朝上的矩形，或者是带旋转方向的矩形。
3. 人脸对齐(Face Alignment)：根据输入的人脸图像，自动定位出人脸上五官关键点坐标的一项技术。
    - 人脸对齐算法的输入是“一张人脸图像”加“人脸坐标框”，输出五官关键点的坐标序列。五官关键点的数量是预先设定好的一个固定数值，
    可以根据不同的语义来定义(常见的有5点、68点、90点等等)。
    - 对人脸图像进行特征点定位，将得到的特征点利用仿射变换进行人脸矫正，若不矫正，非正面人脸进行识别准确率不高。
4. 人脸特征提取(Face Feature Extraction)：将一张人脸图像转化为一串固定长度的数值的过程。
具有表征某个人脸特点能力的数值串被称为“人脸特征(Face Feature)”。
5. 人脸识别(Face Recognition)：识别出输入人脸图对应身份的算法。输入一个人脸特征，通过和注册在库中N个身份对应的特征进行逐个比对，
找出“一个”与输入特征相似度最高的特征。将这个最高相似度值和预设的阈值相比较，如果大于阈值，则返回该特征对应的身份，否则返回“不在库中”。
"""

"""
人脸检测方法1，直接加载opencv训练好的检测器
"""
# # 读入图像
# img = cv2.imread(r"./images/3.png")
#
# # 加载人脸特征，该文件在环境安装目录\Lib\site-packages\cv2\data下
# face_cascade = cv2.CascadeClassifier(r"./weights/haarcascade_frontalface_default.xml")
# # 将读取的图像转化为COLOR_BGR2GRAY，减少计算强度
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
# # 检测出的人脸个数
# faces = face_cascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=4, minSize=(5, 5))
# print("Face : {0}".format(len(faces)))
# print(len(faces))
#
# # 用矩形圈出人脸的位置
# for (x, y, w, h) in faces:
#     cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
#
# cv2.imshow("Faces", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()



"""
人脸检测方法2，用dlib库，用了深度学习
"""
# 定义检测68个关键的模型路径
predictor_model = r"./weights/shape_predictor_68_face_landmarks.dat"
# 初始化人脸检测器
detector = dlib.get_frontal_face_detector()
# 初始化68个关键点预测器
predictor = dlib.shape_predictor(predictor_model)

# 读取图像
img = cv2.imread(r"./images/3.png")
# 转灰度图
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 检测人脸
rects = detector(img_gray, 0)
print(rects)
# 循环遍历每一个人脸矩形框
for i in range(len(rects)):
    # 获得当前人脸的关键点
    landmarks = np.array([[p.x, p.y] for p in predictor(img, rects[i]).parts()])
    print(landmarks, type(landmarks))
    # 遍历关键点
    for idx, point in enumerate(landmarks):
        # 关键点的坐标
        pos = (point[0], point[1])
        # 给每个关键点画一个圆，共68个
        cv2.circle(img, pos, 3, color=(0, 255, 0))
        # 利用cv2.puteText输出1-68
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, str(idx+1), pos, font, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

cv2.imshow("img", img)
cv2.waitKey(0)
cv2.destroyAllWindows()



















