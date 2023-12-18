from ...utils import numpy as np

def greyToRGB(image):
    """
    将单通道的image 灰度图像, 复制到三个RGB通道上
    变成彩色图像

    # 参数:
    image : np数组

    # return:
    返回RGB图像
    """

    image = np.array(image)[:, :, np.newaxis]
    image = np.concatenate((image, image, image), axis=2)


    return image