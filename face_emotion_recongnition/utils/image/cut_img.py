import numpy as np
from PIL import Image


def cut_image(result):
    imgs = []
    for i in range(len(result.boxes.xyxy)):
        xyxy = result.boxes.xyxy[i]
        img = result.orig_img
        img1 = img[ int(xyxy[1]): int(xyxy[3]), int(xyxy[0]): int(xyxy[2])]
        a = np.zeros(img1.shape[:2])
        a[:, :] = img1[:, :, 0] * 0.114 + img1[:, :, 1] * 0.587 + img1[:, :, 2] * 0.299
        # img = Image.fromarray(a)
        # img.show()
        imgs.append(a)
    return imgs