import torch
import torch.nn as nn

class Xception(nn.Module):
    """
    用于构建Xception模型的

    # 参数：
    input_shape : 一个有三个整数的列表[H, W, 3]
    num_classes : 整数, 具体用处如名
    load: 默认是None, 或者输入已经训练的模型的对应数据集的名字。
        (目前没有, 后续添加)

    """

    def __init__(self, input_shape, num_classes, load=None):
        
        pass

    def cfg_Xception(self, input_shape, num_classes):
        
        stem_kernels = [32, 64]
        block_data = [128, 128, 256, 256, 512, 512, 1024]
        model_inputs = (input_shape, num_classes, stem_kernels, block_data)
        
        model = self.build_xception(*model_inputs)

    def build_xception(self, input_shape, num_classes, stem_kernels,
                       block_data, l2_reg=0.01):
        """
        实例化xception模型

        # 参数:

        input_shape : 列表，对应
        """
        pass
            

    
