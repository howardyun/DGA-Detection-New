import torch
from torch.utils.data import Dataset
import pandas as pd
import os
import glob

# # pd的打印
# # 显示所有列
# pd.set_option('display.max_columns', None)
# # 显示所有行
# pd.set_option('display.max_rows', None)
# # 设置value的显示长度为100，默认为50
# pd.set_option('max_colwidth', 500)

# unicode编码下的40个特殊字符
elements = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
            'v', 'w', 'x', 'y', 'z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '-', '.', '@', '%']


# 处理字母转数字
def AlpMapDigits(source_str):
    max_length = 255
    # 创建字符到下标的映射字典
    char_to_index = {char: index for index, char in enumerate(elements)}
    # 将字符串中的每个字符映射成数组的下标
    mapped_indices = [char_to_index[char] for char in source_str]
    # 填充零
    zero_num = max_length - len(mapped_indices)
    for i in range(zero_num):
        mapped_indices.insert(0, 0)
        pass
    return mapped_indices
    pass


# 处理标签，False为0，True为1
def ChangeLabel(flag):
    if not flag:
        return 0
        pass
    else:
        return 1
    pass


# 处理标签为True的数据
class DGATrueDataset(Dataset):
    """
    DGA加载data中数据，对数据进行处理返回dataset
    """

    def __init__(self, root_csv_path, train=True):
        """
        :param root_csv_path: csv文件的根路径
        :param train: True返回训练集逻辑的dataset，这个dataset也可以用dataLoader分割出训练集和测试集，
        False直接返回测试集逻辑的dataset
        """
        self.test_data = None
        self.train_data = None
        self.root_csv_path = root_csv_path
        self.train = train

        # 文件夹是否存在
        if not self._check_exists():
            raise RuntimeError('Dataset not found.')

        # 处理训练集和测试集
        csv_files = glob.glob(os.path.join(root_csv_path, '*.csv'))
        if self.train:
            # 处理训练集
            all_dataframe = pd.DataFrame()
            # 获取csv文件列表
            # 拼接文件夹下所有csv文件数据
            for file in csv_files:
                dataframe = pd.read_csv(file, header=None)

                # 提取域名和标签列，域名编码
                # 处理大写
                dataframe[1] = dataframe[1].str.lower()

                # 按列切割，只要两列，一列域名，一列标签
                dataframe = dataframe.iloc[:, 1:3]
                dataframe.columns = [0, 1]

                # 逐一编码
                dataframe[0] = dataframe[0].apply(AlpMapDigits)
                dataframe[1] = dataframe[1].apply(ChangeLabel)

                # all_dataframe = all_dataframe.append(dataframe, ignore_index=True)
                all_dataframe = pd.concat([all_dataframe, dataframe], ignore_index=True)
                pass

            # 获取第一列域名和倒数第一列标签的数据，并转成numpy
            self.train_data = all_dataframe.iloc[:, 0].values
            self.target = all_dataframe.iloc[:, -1].values
            pass
        else:
            # 处理训练集
            all_dataframe = pd.DataFrame()
            # 获取csv文件列表
            # 拼接文件夹下所有csv文件数据
            for file in csv_files:
                dataframe = pd.read_csv(file, header=None)

                # 提取域名和标签列，域名编码
                # 处理大写
                dataframe[1] = dataframe[1].str.lower()

                # 按列切割，只要两列，一列域名，一列标签
                dataframe = dataframe.iloc[:, 1:3]
                dataframe.columns = [0, 1]

                # 逐一编码
                dataframe[0] = dataframe[0].apply(AlpMapDigits)
                dataframe[1] = dataframe[1].apply(ChangeLabel)

                # all_dataframe = all_dataframe.append(dataframe, ignore_index=True)
                all_dataframe = pd.concat([all_dataframe, dataframe], ignore_index=True)
                pass

            # 获取第一列域名的数据，并转成numpy
            self.test_data = all_dataframe.iloc[:, 0].values
            pass
        pass

    def __getitem__(self, index):
        if self.train:
            # dataI, targetI = self.train_data[index, :], self.target[index]
            dataI, targetI = self.train_data[index], self.target[index]
            dataI = torch.tensor(dataI)
            targetI = torch.tensor(targetI)
            return dataI, targetI
        else:
            dataI = self.test_data.iloc[index]
            dataI = torch.tensor(dataI)
            return dataI
        pass

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)
        pass

    def _check_exists(self):
        return os.path.exists(self.root_csv_path)

    pass


# 处理标签为False的数据
class DGAFalseDataset(Dataset):
    """
    DGA加载data中数据，对数据进行处理返回dataset
    """

    def __init__(self, root_csv_path, train=True):
        """
        :param root_csv_path: csv文件的根路径
        :param train: train模式还是test模式，默认train模型
        """
        self.test_data = None
        self.train_data = None
        self.root_csv_path = root_csv_path
        self.train = train

        # 文件夹是否存在
        if not self._check_exists():
            raise RuntimeError('Dataset not found.')

        # 处理训练集和测试集
        csv_files = glob.glob(os.path.join(root_csv_path, '*.csv'))
        if self.train:
            # 处理训练集
            all_dataframe = pd.DataFrame()
            # 获取csv文件列表
            # 拼接文件夹下所有csv文件数据
            for file in csv_files:
                dataframe = pd.read_csv(file, header=None, nrows=30000)

                # 提取域名和标签列，域名编码
                # 按列切割，只要两列，一列域名，一列标签
                dataframe = dataframe.iloc[:, [0, -1]]
                dataframe.columns = [0, 1]

                # 处理大写
                dataframe[0] = dataframe[0].str.lower()

                # 逐一编码
                dataframe[0] = dataframe[0].apply(AlpMapDigits)
                dataframe[1] = dataframe[1].apply(ChangeLabel)

                # all_dataframe = all_dataframe.append(dataframe, ignore_index=True)
                all_dataframe = pd.concat([all_dataframe, dataframe], ignore_index=True)
                pass

            # print(all_dataframe.head(20))
            # 获取第一列域名和倒数第一列标签的数据，并转成numpy
            self.train_data = all_dataframe.iloc[:, 0].values
            self.target = all_dataframe.iloc[:, -1].values
            pass
        else:
            # 处理训练集
            all_dataframe = pd.DataFrame()
            # 获取csv文件列表
            # 拼接文件夹下所有csv文件数据
            for file in csv_files:
                dataframe = pd.read_csv(file, header=None, nrows=30000)

                # 提取域名和标签列，域名编码
                # 按列切割，只要两列，一列域名，一列标签
                dataframe = dataframe.iloc[:, [0, -1]]
                dataframe.columns = [0, 1]

                # 处理大写
                dataframe[0] = dataframe[0].str.lower()

                # 逐一编码
                dataframe[0] = dataframe[0].apply(AlpMapDigits)
                dataframe[1] = dataframe[1].apply(ChangeLabel)

                # all_dataframe = all_dataframe.append(dataframe, ignore_index=True)
                all_dataframe = pd.concat([all_dataframe, dataframe], ignore_index=True)
                pass

            # 获取第一列域名的数据，并转成numpy
            self.test_data = all_dataframe.iloc[:, 0].values
            pass
        pass

    def __getitem__(self, index):
        if self.train:
            # dataI, targetI = self.train_data[index, :], self.target[index]
            dataI, targetI = self.train_data[index], self.target[index]
            dataI = torch.tensor(dataI)
            targetI = torch.tensor(targetI)
            return dataI, targetI
        else:
            dataI = self.test_data.iloc[index]
            dataI = torch.tensor(dataI)
            return dataI
        pass

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)
        pass

    def _check_exists(self):
        return os.path.exists(self.root_csv_path)

    pass
