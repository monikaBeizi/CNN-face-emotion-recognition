import numpy as np
import sys
import cv2
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

            # img = Image.fromarray(frame, )
            #
            # img.show()

            result = results[0]

            imgs = cut_image(result)

            labels = []

            for i in imgs:
                pre_image = trans_predict(i)
                ans = model.predict(pre_image)
                labels.append(get_emotions(ans))

            # 在帧上可视化结果

            annotated_frame = plot_boxes(labels, result)

            # 显示带注释的帧
            cv2.imshow("YOLOv8推理", annotated_frame)

            # 如果按下'q'则中断循环
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        # else:
        #     # 如果视频结束则中断循环
        #     break

    # 释放视频捕获对象并关闭显示窗口
    cap.release()
    cv2.destroyAllWindows()

    #
    #
    #
    # grey_image = np.array(grey_image)
    #
    # pre_image= trans_predict(grey_image)
    # ans = model.predict(pre_image)
    #
    # print(get_emotions(ans))