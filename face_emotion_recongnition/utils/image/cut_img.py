import numpy as np
from PIL import Image


def cut_image(result):
    '''
    将识别出的人脸剪切出来，并添加到一个列表中

    # 参数：yolov8的预测结果

    # 返回值：包含所有人脸的灰度图像，保存到了一个列表中

    # result.boxes.xyxy：包含了所要裁剪的图片的位置信息

    # result.orig_img：原始图片，为ndarray格式

    '''
    imgs = []
    for i in range(len(result.boxes.xyxy)):
        xyxy = result.boxes.xyxy[i]
        img = result.orig_img
        img1 = img[int(xyxy[1]): int(xyxy[3]), int(xyxy[0]): int(xyxy[2])]
        img1 = np.array(Image.fromarray(img1).convert('L'))
        # img = Image.fromarray(a)
        # img.show()
        imgs.append(img1)
    return imgs