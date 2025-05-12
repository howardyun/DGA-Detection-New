import math

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
    # # 创建字符到下标的映射字典
    # char_to_index = {char: index + 1 for index, char in enumerate(elements)}
    # # 将字符串中的每个字符映射成数组的下标
    # mapped_indices = [char_to_index[char] for char in source_str]
    # 填充零--向前
    # zero_num = max_length - len(mapped_indices)
    # for i in range(zero_num):
    #     mapped_indices.insert(0, 0)
    #     pass

    # 填充零--向后
    # zero_num = max_length - len(mapped_indices)
    # for i in range(zero_num):
    #     mapped_indices.append(0)  # 向后填充0
    #     pass

    max_length = 255
    # 创建字符到下标的映射字典
    char_to_index = {char: index + 1 for index, char in enumerate(elements)}
    # 将字符串中的每个字符映射成数组的下标
    mapped_indices = [char_to_index[char] for char in source_str]

    # 如果字符串长度小于最大长度，则重复填充字符直到达到最大长度
    if len(mapped_indices) < max_length:
        repeat_count = math.ceil(max_length / len(mapped_indices))
        mapped_indices = (mapped_indices * repeat_count)[:max_length]

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
class DGATrueDataset_ysx(Dataset):
    """
    DGA加载data中数据，对数据进行处理返回dataset
    更新后数据集形态变成：域名，二分类标签，多分类标签。
    其中二分类标签0为良性数据，1为恶性数据
    其中多分类标签按照数据集不同家族标注
    """

    def __init__(self, dataframe, train=True, bni=True):
        """
        :param dataframe: 固定批次数据生成的数据帧dataframe
        :param train: 返回训练集数据
        :param bni: True表示二分类
        """
        self.test_data = None
        self.train_data = None
        self.train = train
        self.bni = bni

        if self.train:
            # 处理训练集
            all_dataframe = pd.DataFrame()
            # 按列切割，需要三列
            # 结构为域名，二分类标签，多分类标签
            dataframe = dataframe.iloc[:, 0:3]
            dataframe.columns = [0, 1, 2]
            # 逐一编码

            dataframe[0] = dataframe[0].apply(AlpMapDigits)
            # print(dataframe[0][0])
            # exit(1000)

            all_dataframe = pd.concat([all_dataframe, dataframe], ignore_index=True)

            # 获取第一列域名
            self.train_data = all_dataframe.iloc[:, 0].values
            if self.bni:
                # 如果是二分类，返回二分类标签
                self.target = all_dataframe.iloc[:, -2].values
            else:
                # 如果是多分类，返回多分类标签
                self.target = all_dataframe.iloc[:, -1].values
            pass

        else:
            # 处理预测集，预测集除了返回训练集的中的域名编码，二分类或者多分类标签，还有一列未编码后的域名
            # 处理训练集
            all_dataframe = pd.DataFrame()
            # 按列切割，需要三列
            # 结构为域名，二分类标签，多分类标签
            dataframe = dataframe.iloc[:, 0:3]
            dataframe.columns = [0, 1, 2]
            # 逐一编码
            dataframe[3] = dataframe[0]
            dataframe[0] = dataframe[0].apply(AlpMapDigits)
            all_dataframe = pd.concat([all_dataframe, dataframe], ignore_index=True)

            # 获取第一列域名
            self.test_data = all_dataframe.iloc[:, 0].values
            if self.bni:
                # 如果是二分类，返回二分类标签
                self.target = all_dataframe.iloc[:, 1].values
                self.name = all_dataframe.iloc[:, -1].values
                pass
            else:
                # 如果是多分类，返回多分类标签
                self.target = all_dataframe.iloc[:, 2].values
                self.name = all_dataframe.iloc[:, -1].values
                pass
            pass
        pass

    def __getitem__(self, index):
        if self.train:
            dataI, targetI = self.train_data[index], self.target[index]
            dataI = torch.tensor(dataI)
            targetI = torch.tensor(targetI)
            return dataI, targetI
        else:
            dataI, targetI, dgaName = self.test_data[index], self.target[index], self.name[index]
            dataI = torch.tensor(dataI)
            targetI = torch.tensor(targetI)
            return dataI, targetI, dgaName
        pass

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)
        pass

    pass
