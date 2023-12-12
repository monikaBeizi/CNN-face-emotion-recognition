import os
import numpy as np

from ..abstract import Loader, Dataset
from ..utils import get_class_names, resize_image


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
        self.images_path = os.path.join(self.path, 'icml_face_data.csv')
        self.labels_path = os.path.join(self.path, 'fer2013new.csv')

        self.split_to_filter = {
            'train' : 'Training', 'val': 'PublicTest', 'test': 'PrivateTest'
        }

    def load_data(self):
        # 通过genfromtxt读取数据，第二列用处，第三列字符串形式的像素集合
        data = np.genfromtxt(self.images_path, str, '#', ',', 1)

        # 加载对应self.split对应的数据
        data = data[ data[:, -2] == self.split_to_filter[self.split]]

        # 将加载的data 编程的形状变为(-1, 48, 48)
        faces = np.zeros((len(data), *self.image_size))
        for sample_arg, sample in enumerate(data):
            face = np.array(sample[2].split(' '), dtype=int).reshape(48, 48)
            face = resize_image(face, self.image_size)

            faces[sample_arg, :, :] = face


        # 从ferPlus 中去除emotion标签

        emotions = np.genfromtxt(self.labels_path, str, '#', ',', 1)
        # 从中去除 与self.split对应的标签
        emotions = emotions[ emotions[:, 0] == self.split_to_filter[self.split]]
        # 将数值化的特征提取出来
        emotions = emotions[:, 2:10].astype(float)
        # 对每一行求和用来归一化，并且分辨是否含有NULL行
        N = np.sum(emotions, axis=1)

        # 去除不含特征值的图片
        mask = N != 0
        N, face, emotions = N[mask], faces[mask], emotions[mask]
        
        # 对emotions标签的每一列进行归一化
        emotions = emotions / np.expand_dims(N, 1)

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

    def __getitem__(self, index):

        data = self.data[index]['image']
        target = self.data[index]['label']

        return data, target
    
    def __len__(self):

        return len(self.data)