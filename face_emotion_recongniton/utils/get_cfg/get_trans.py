from ...utils import transforms

def get_trans():
    '''
    用于获取配置好的torchvision的transform
    '''

    trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(44)
    ])

    return trans