import pandas as pd

from .normalization import normalization

def extractLabels(images, labels, split):
    """
    分别提取images列表中的像素, 和labels中的标签, 并将其归一化

    # 参数:
    images: 含有pixels的列表
    labels: 含有ferplus的八种情绪的标签的列表
    split:  字符串,用来区分提取的数据集是用来训练的还是测试的


    # return:
    返回的是一个第一行为像素,第二行后到最后一行为八个标签的csv列表
    """
    # 去除表格中目标数据以外的数据
    if split != 'All':
      images = images[images['Usage'] == split]
      labels = labels[labels['Usage'] == split]

    # 提取出像素和标签
    new_fer = pd.DataFrame()
    new_fer['pixels'] = images['pixels']
    new_fer[['neutral', 'happiness', 'surprise', 'sadness',
       'anger', 'disgust', 'fear', 'contempt']] = labels[['neutral', 'happiness', 'surprise', 'sadness',
       'anger', 'disgust', 'fear', 'contempt']]

    # 去除new_fer中无效行(即8个标签都是0)
    new_fer = new_fer[new_fer[['neutral', 'happiness', 'surprise', 'sadness',
       'anger', 'disgust', 'fear', 'contempt']].sum(axis=1) != 0]
    
    new_fer = new_fer.apply(normalization, axis=1)
    
    return new_fer