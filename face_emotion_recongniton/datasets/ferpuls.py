from ..abstract import Loader
from ..utils import get_class_names

class FER_plus:
    """
    用来加载FER2013 人脸情绪数据集的类, 标签为FERpuls
    数据集来自:https://github.com/microsoft/FERPlus

    参数：
    path: 含有icml_face_data.csv(数据集) 和 fer2013new.csv(标签)的路径
    split: 用来区分数据集用途的
    class_names: 一个含有数据集中标签种类的列表

    image_size: 图片将要被改变的大小
    """

    def __init__(self, path, split='train', class_names="ALL",
                  image_size=(48, 48)) -> None:
        
        if class_names == "ALL":
            class_names = get_class_names('FERplus')
        
        super(FER_plus,self).__init__(self, path, split, class_names, 'FERplus')