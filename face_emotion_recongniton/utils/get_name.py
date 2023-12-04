def get_class_names(datasets_name='FERplus'):
    """
    用来存数据集对应的标签名

    参数：
    datasets_name: 数据集的名字

    返回: 
    一个关于数据集标签的字符串列表

    报错: 
    输入了一个暂时还没有存储数据集的名

    """

    if datasets_name == 'FERplus':
        return ['neutral', 'happiness', 'surprise', 'sadness',
                'anger', 'disgust', 'fear', 'contempt']
    
    elif datasets_name == 'FER':
        return ['angry', 'disgust', 'fear', 'happy',
                'sad', 'surprise', 'neutral']
    

    else:
        raise ValueError('输入的数据集名错误', datasets_name)
    