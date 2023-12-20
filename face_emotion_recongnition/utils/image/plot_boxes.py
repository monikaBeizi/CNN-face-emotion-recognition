import cv2
import numpy as np


def plot_boxes(labels, result):
    img = result.orig_img
    for i in range(len(result.boxes.xyxy)):
        xyxy = result.boxes.xyxy[i]
        tl = round(0.002 * max(img.shape[0:2])) + 1
        color = (0, 0, 255)
        c1, c2 = ((int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])))
        cv2.rectangle(img, c1, c2, color, lineType=cv2.LINE_AA, thickness=tl)
        # print(i)
        # lable
        tf = max(tl-1, 1)
        t_size = cv2.getTextSize(labels[i], 0, fontScale=tl/3, thickness=tf)[0]
        c2 = (c1[0] + t_size[0], c1[1] - t_size[1] - 3)
        cv2.rectangle(img, c1, c2, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, labels[i], (c1[0], c1[1] - 2), cv2.FONT_HERSHEY_SIMPLEX, tl / 3, [225, 255, 255], thickness=tf,
                    lineType=cv2.LINE_AA)

    return img