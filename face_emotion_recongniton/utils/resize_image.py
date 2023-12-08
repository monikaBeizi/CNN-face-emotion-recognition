import os
import cv2
import numpy as np

BILINEAR = cv2.INTER_LINEAR

def resize_image(image, size, method=BILINEAR):
    """
    resize_image

    参数:
    image: numpy数组
    size: 输出图片的尺寸
    methon: 使用的插值方法

    return:
    返回的是一个numpy数组
    """
    
    if( type(image) != np.ndarray):
        raise ValueError(
            "输入的image不是一个numpy数组", type(image)
        )
    
    else:
        return cv2.resize(image, size, interpolation=method)