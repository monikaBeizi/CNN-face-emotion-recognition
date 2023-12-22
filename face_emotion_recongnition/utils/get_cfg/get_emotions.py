def get_emotions(predict):
    """
    用于数字0~7对应的情绪

    # 参数:
    模型predict的结果

    # return:
    对应predict的emotion

    # emotions对应的中文(放在这方便改代码):
    emotions = {
        0: '中立', 1: '快乐', 2: '惊讶', 3: '悲伤',
        4: '愤怒', 5: '厌恶', 6: '恐惧', 7:'蔑视'
    }

    """

    emotions = {
        0: 'neutral', 1: 'happiness', 2: 'surprise', 3: 'sadness',
        4: 'anger', 5: 'anger', 6: 'fear', 7:'contempt'
    }

    return emotions[predict]

    