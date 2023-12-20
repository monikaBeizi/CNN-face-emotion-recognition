class Loader:
    """
    加载数据集的抽象方法

    # 参数
    path: 数据集的相对路径
    split: 用来区分加载的数据集是用于训练,还是验证或者测试的
    class_names: 一组关于数据集标签的字符串列表
    name: 数据集的名字
    
    # 方法
    load_data()

    """

    def __init__(self, path, split, class_names, name) -> None:
        self.path = path
        self.split = split
        self.class_names = class_names
        self.name = name

    def load_data(self):
        """
        用于加载数据集的抽象方法
        """

        raise NotImplementedError()
    
    # 虽然这么写对咱们没什么用就是了

    @property
    def path(self):
        return self._path
    
    @path.setter
    def path(self, path):
        self._path = path


    @property
    def split(self):
        return self._split
    
    @split.setter
    def split(self, split):
        self._split = split


    @property
    def class_names(self):
        return self._class_names
    
    @class_names.setter
    def class_names(self, class_names):
        self._class_names = class_names


    @property
    def name(self):
        return self._name
    
    @name.setter
    def name(self, name):
        self._name = name


    # 用来查看有多少个类别
    @property
    def class_nums(self):
        if isinstance(self.class_names, list):
            return len(self.class_names)
        
        else:
            raise ValueError('class_names 不是一个列表')