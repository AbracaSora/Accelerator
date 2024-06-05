import random


class Dataset:
    def __init__(self):
        """
        初始化数据集
        """
        self.train = ([], [])
        self.validation = ([], [])

    def load(self, train, validation):
        """
        :param train:  训练集
        :param validation:  验证集
        :return:  None

        加载数据集
        """
        self.train = train
        self.validation = validation

    def insert(self, data: tuple):
        """
        :param data: 新数据, (图像, 鼠标位置)
        :return: None

        插入数据
        """
        if random.random() <= 0.7:
            self.train[0].append(data[0])
            self.train[1].append(data[1])
        else:
            self.validation[0].append(data[0])
            self.validation[1].append(data[1])

    @property
    def size(self):
        """
        :return: 数据集大小
        """
        return len(self.train[0])
