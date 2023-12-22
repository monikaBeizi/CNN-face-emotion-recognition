import cv2
import numpy as np


def plot_boxes(labels, result):
    '''
    # 将识别出来的标签标到原始图像上，并画框框出人脸

    # labels: VGG网络识别出来的各个脸的情绪的标签，为列表格式

    # result: yolov8的目标检测结果

    # result.boxes.xyxy：包含了所要裁剪的图片的位置信息

    # result.orig_img：原始图片，为ndarray格式

    # 返回值：标好框和标签的图像
    '''


    img = result.orig_img
    for i in range(len(result.boxes.xyxy)):
        xyxy = result.boxes.xyxy[i]
        tl = round(0.002 * max(img.shape[0:2])) + 1
        color = (0, 0, 255)
        c1, c2 = ((int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])))
        cv2.rectangle(img, c1, c2, color, lineType=cv2.LINE_AA, thickness=tl)

        tf = max(tl-1, 1)
        t_size = cv2.getTextSize(labels[i], 0, fontScale=tl/3, thickness=tf)[0]
        c2 = (c1[0] + t_size[0], c1[1] - t_size[1] - 3)
        cv2.rectangle(img, c1, c2, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, labels[i], (c1[0], c1[1] - 2), cv2.FONT_HERSHEY_SIMPLEX, tl / 3, [225, 255, 255], thickness=tf,
                    lineType=cv2.LINE_AA)

    return img