def normalization(x):
    """
    用于apply方法的函数

    # 参数:
    x :为列表的每一行

    # return
    
    x归一化后
    """
    if x[1:].sum() == 0:
        raise ValueError("该行为无效值", x)
    x[1:] = x[1:] / x[1:].sum()

    return x