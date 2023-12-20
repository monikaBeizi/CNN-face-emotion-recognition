from .resize_image import resize_image
from .greyToRGB import greyToRGB

from ..get_cfg.get_trans import get_trans

def trans_predict(image):
    """
    接收一个原始的图像, 并将其转化为可以提供给模型预测的大小

    # 参数:
    image: 由np数组表示的图片type:ndarra 

    # return:
    返回一个48 * 48大小的图片
    """
    trans = get_trans()
    small_image = resize_image(image, (48, 48))
    small_image = greyToRGB(small_image)
    small_image = trans(small_image)
    small_image = small_image.reshape(1, 3, 44, 44)

    return small_image.float()