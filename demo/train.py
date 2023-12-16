import sys
sys.path.append('..')

from face_emotion_recongniton.datasets import FER_plus_dataSet
from face_emotion_recongniton.utils import get_trans
from face_emotion_recongniton.models import EmotionRecognition
from torch.utils.data import DataLoader

trans = get_trans()

# 加载数据集
fer = FER_plus_dataSet('../data/fer2013/', split='train', class_names="ALL")
fertest = FER_plus_dataSet('../data/fer2013/', split='test', class_names="ALL")

# 实例化DataLoader
trainLoader = DataLoader(fer, batch_size=128, shuffle=True)
testLoader = DataLoader(fertest, batch_size=1, shuffle=False)

# 训练模型(True 为加载训练好的模型)
model = EmotionRecognition(cfg='VGG16')
model.train(trainLoader, 100, False)