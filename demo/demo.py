import numpy as np
import sys
import cv2
from PIL import Image
from ultralytics import YOLO

sys.path.append('.')

from train import train
from face_emotion_recongnition.utils import trans_predict
from face_emotion_recongnition.utils import get_emotions
from face_emotion_recongnition.utils import cut_image
from face_emotion_recongnition.utils import plot_boxes

if __name__ == '__main__':

    yolo = YOLO("../data/best.pt")  # 或者加载预训练好的模型

    model = train(load=True)

    cap = cv2.VideoCapture(0)  # 0表示默认摄像头，如果有多个摄像头，可以尝试使用1, 2, 等

    # 遍历视频帧

    while cap.isOpened():
        # 从视频中读取一帧
        success, frame = cap.read()
        # print(type(frame))
        if success:
            # 在该帧上运行YOLOv8推理
            results = yolo(frame)

            result = results[0]

            imgs = cut_image(result)
            labels = []
            for i in imgs:
                most = []
                for n in range(7):
                    pre_image = trans_predict(i)
                    ans = model.predict(pre_image)
                    most.append(ans)
                print(most)
                most_num = max(most, key=most.count)
                labels.append(get_emotions(most_num))

            # 在帧上可视化结果
            print(labels)
            annotated_frame = plot_boxes(labels, result)

            # 显示带注释的帧
            cv2.imshow("YOLOv8推理", annotated_frame)

            # 如果按下'q'则中断循环
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # 释放视频捕获对象并关闭显示窗口
    cap.release()
    cv2.destroyAllWindows()
