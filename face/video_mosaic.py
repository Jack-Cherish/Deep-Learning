# Author : Jack Cui
# Website: https://cuijiahua.com/
import cv2
import face_recognition
import matplotlib.pyplot as plt
# %matplotlib inline # 在 jupyter 中使用的时候，去掉注释

import subprocess
import os
from PIL import Image

def video2mp3(file_name):
    """
    将视频转为音频
    :param file_name: 传入视频文件的路径
    :return:
    """
    outfile_name = file_name.split('.')[0] + '.mp3'
    cmd = 'ffmpeg -i ' + file_name + ' -f mp3 ' + outfile_name
    print(cmd)
    subprocess.call(cmd, shell=True)


def video_add_mp3(file_name, mp3_file):
    """
     视频添加音频
    :param file_name: 传入视频文件的路径
    :param mp3_file: 传入音频文件的路径
    :return:
    """
    outfile_name = file_name.split('.')[0] + '-f.mp4'
    subprocess.call('ffmpeg -i ' + file_name
                    + ' -i ' + mp3_file + ' -strict -2 -f mp4 '
                    + outfile_name, shell=True)

def mask_video(input_video, output_video, mask_path='mask.jpg'):
    # 打码图片
    mask = cv2.imread(mask_path)
    # 读取视频
    cap = cv2.VideoCapture(input_video)
    # 读取视频参数,fps、width、heigth
    CV_CAP_PROP_FPS = 5
    CV_CAP_PROP_FRAME_WIDTH = 3
    CV_CAP_PROP_FRAME_HEIGHT = 4
    v_fps = cap.get(CV_CAP_PROP_FPS)
    v_width = cap.get(CV_CAP_PROP_FRAME_WIDTH)
    v_height = cap.get(CV_CAP_PROP_FRAME_HEIGHT)
    # 设置写视频参数，格式为 mp4
    size = (int(v_width), int(v_height))
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    out = cv2.VideoWriter(output_video,fourcc, v_fps, size)
    
    # 已知人脸
    known_image = face_recognition.load_image_file("tz.jpg")
    biden_encoding = face_recognition.face_encodings(known_image)[0]
    # 读取视频
    cap = cv2.VideoCapture(input_video)
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            # 检测人脸
            face_locations = face_recognition.face_locations(frame)
            # 检测每一个人脸
            for (top_right_y, top_right_x, left_bottom_y,left_bottom_x) in face_locations:
                unknown_image = frame[top_right_y-50:left_bottom_y+50, left_bottom_x-50:top_right_x+50]
                unknown_encoding = face_recognition.face_encodings(unknown_image)[0]
                # 对比结果
                results = face_recognition.compare_faces([biden_encoding], unknown_encoding)
                # 是仝卓，就将打码贴图。
                if results[0] == True:
                    mask = cv2.resize(mask, (top_right_x-left_bottom_x, left_bottom_y-top_right_y))
                    frame[top_right_y:left_bottom_y, left_bottom_x:top_right_x] = mask
            # 写入视频
            out.write(frame)
        else:
            break
    
if __name__ == '__main__':
    # 将音频保存为cut.mp3
    video2mp3(file_name='cut.mp4')
    # 处理视频，自动打码，输出视频为output.mp4
    mask_video(input_video='cut.mp4', output_video='output.mp4')
    # 为 output.mp4 处理好的视频添加声音
    video_add_mp3(file_name='output.mp4', mp3_file='cut.mp3')
