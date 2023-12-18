import sys
sys.path.append('.')

from face_emotion_recongnition.datasets import FER_plus_dataSet
from face_emotion_recongnition.utils import get_trans
from face_emotion_recongnition.models import EmotionRecognition
from torch.utils.data import DataLoader


def train(load=True, evaluate=False):
    """
    训练模型

    # 参数:
    load: 是否加载训练好的模型
    evaluate: 是否用测试集来测试模型预测成功率并打印

    # return:
    返回训练好的模型
    """
    trans = get_trans()
    trainLoader = 0

    if load==False:
        # 加载数据集
        fer = FER_plus_dataSet('data/fer2013/', split='train', class_names="ALL")
        
        # 实例化DataLoader
        trainLoader = DataLoader(fer, batch_size=128, shuffle=True)

    # 训练模型(True 为加载训练好的模型)
    vgg = EmotionRecognition(cfg='VGG16')
    vgg.train(trainLoader, 100, True)
    
    if evaluate==True:
        fertest = FER_plus_dataSet('data/fer2013/', split='test', class_names="ALL")
        testLoader = DataLoader(fertest, batch_size=1, shuffle=False)
        vgg.evaluate(testLoader)

    return vgg

if __name__ == '__main__':
    vgg = train(load=True, evaluate=True)