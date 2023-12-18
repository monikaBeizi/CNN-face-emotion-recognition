import os

from ..abstract import Loader, Dataset
from ..utils import get_class_names, get_trans, resize_image
from ..utils import greyToRGB, extractLabels
from ..utils import numpy as np
from ..utils import pandas as pd

class FER_plus(Loader):
    """
    用来加载FER2013 人脸情绪数据集的类, 标签为FERpuls
    数据集来自:https://github.com/microsoft/FERPlus
    数据集的第二列特征就是改数据的用处, 因此用split来区分训练集,测试集,验证集

    # 参数：
    path: 含有icml_face_data.csv(数据集) 和 fer2013new.csv(标签)的路径
    split: 用来区分数据集用途的
    class_names: 一个含有数据集中标签种类的列表

    image_size: 图片将要被改变的大小

    # return
    data: 返回处理好的数据,data是列表, 里面的每个图片和标签用字典分别来保存
    'image': 图片像素的numpy数组
    'emotion': 关于各项情绪的可能性的, 一行numpy数组
    """

    def __init__(self, path, split='train', class_names="ALL",
                  image_size=(48, 48)) -> None:
        
        # 如果是另取的类名的话就用另取的
        if class_names == "ALL":
            class_names = get_class_names('FERplus')
        
        super(FER_plus,self).__init__(path, split, class_names, 'FERplus')

        self.image_size = image_size
        self.images_path = os.path.join(self.path, 'fer2013.csv')
        self.labels_path = os.path.join(self.path, 'fer2013new.csv')

        self.split_to_filter = {
            'train' : 'Training', 'val': 'PublicTest', 'test': 'PrivateTest', 'all': 'All'
        }

    def load_data(self):
        
        # 读取数据和标签
        images = pd.read_csv(self.images_path)
        labels= pd.read_csv(self.labels_path)

        # 提取像素和标签，并把标签归一化
        new_fer = extractLabels(images=images, labels=labels, split=self.split_to_filter[self.split])

        # 将图片放入data中,  reshape成(-1, 48, 48)
        faces = np.zeros((len(new_fer['pixels']), *self.image_size))
        for sample_arg, face in enumerate(new_fer['pixels']):
            face = np.array(face.split(), dtype=int).reshape(48, 48)
            face = resize_image(face, self.image_size)

            faces[sample_arg, :, :] = face

        # 获取标签
        emotions = new_fer[['neutral', 'happiness', 'surprise', 'sadness',
       'anger', 'disgust', 'fear', 'contempt']].values
        
        # 返回处理好的数据,data是列表，里面的每个图片和标签用字典分别来保存
        data = []
        for face, emotion in zip(faces, emotions):
            sample = {'image': face, 'label': emotion}
            data.append(sample)
        return data
    
class FER_plus_dataSet(Dataset):
    """
    调用了同文件的FER_plus类来获取数据集, 然后转化成适合用于加载DataLoader类的Dataset类型
    后续应该将其并入到FER_plus类中, FER_plus同时继承两个类
    """

    def __init__(self, path, split='train', class_names="ALL",
                image_size=(48, 48)) -> None:
    
        fer_plus = FER_plus(path, split, class_names, image_size)
        self.data = fer_plus.load_data()
        self.trans = get_trans()

    def __getitem__(self, index):

        data = self.data[index]['image']
        
        data = greyToRGB(data)
        data = self.trans(data)

        target = self.data[index]['label']

        return data, target

    def __len__(self):

        return len(self.data)