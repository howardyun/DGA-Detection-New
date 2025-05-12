import pandas as pd
import os
import glob
import csv

elements = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
            'v', 'w', 'x', 'y', 'z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '-', '.', '@', '%']


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


# 指定路径下的所有csv添加标签列
def SetLabel(root_csv_path, flag):
    """
    :param root_csv_path: csv文件文件夹根路径
    :param flag: 需要添加的标签
    :return:
    """
    # 获取csv文件列表
    csv_files = glob.glob(os.path.join(root_csv_path, '*.csv'))

    # 为每一行添加一列
    for file in csv_files:
        dataframe = pd.read_csv(file, header=None)
        # 生成行数的标签
        label_num = dataframe.shape[0]
        label_index = dataframe.shape[1]
        label_list = [flag] * label_num

        # 加入新的一列
        dataframe[label_index] = label_list
        dataframe.to_csv(file, index=False, header=False)
        print(f'完成添加：{file}')
        pass
    pass


def Set_label_list_form_benign(root_csv_path, target_csv_path):
    """
    :param root_csv_path: csv文件文件夹根路径
    :param target_csv_path: 输出文件夹
    :return:
    """
    csv_files = glob.glob(os.path.join(root_csv_path, '*.csv'))
    print(csv_files)
    for file in csv_files:
        dataframe = pd.read_csv(file, header=None)
        # 提取域名和标签列，域名编码
        # 处理大写
        dataframe[1] = dataframe[1].str.lower()
        # 按列切割，只要两列，一列域名，一列标签
        dataframe = dataframe.iloc[:, 1:3]
        dataframe.columns = [0, 1]
        dataframe[0] = dataframe[0].apply(AlpMapDigits)
        label_list = [0] * dataframe.shape[0]
        df = pd.DataFrame({'domainname_vec': dataframe[0], 'label': label_list})
        df.to_csv(target_csv_path, index=False, header=False)


def Set_label_list_form_malicious(root_csv_path_1, root_csv_path_2, target_csv_path_1, target_csv_path_2):
    """
    :param root_csv_path: csv文件小文件夹根路径
    :param root_csv_path_1: csv文件大文件夹根路径
    :param target_csv_path: 输出文件夹
    :return:
    """
    print("第一个文件夹:")
    # 用于记录已有的DGA家族
    hash_table = {}
    index = 1
    csv_files1 = glob.glob(os.path.join(root_csv_path_1, '*.csv'))
    for file in csv_files1:
        filename = file.split('/')[-1]
        label = index
        # 判断文件是否在,如果在label等于之前的标签
        if (filename in hash_table):
            print("same:" + filename)
            label = hash_table[filename]
        else:
            # 如果不在index+1
            hash_table[filename] = index
            index = index + 1
        dataframe = pd.read_csv(file, header=None)
        # 提取域名和标签列，域名编码
        # 按列切割，只要两列，一列域名，一列标签
        dataframe = dataframe.iloc[:, [0, -1]]
        dataframe.columns = [0, 1]
        # 处理大写
        dataframe[0] = dataframe[0].str.lower()
        # 逐一编码
        dataframe[0] = dataframe[0].apply(AlpMapDigits)
        label_list = [label] * dataframe.shape[0]
        df = pd.DataFrame({'domainname_vec': dataframe[0], 'label': label_list})
        df.to_csv(target_csv_path_1 + '/' + filename, index=False, header=False)

    print("第二个文件夹:")
    csv_files2 = glob.glob(os.path.join(root_csv_path_2, '*.csv'))
    for file in csv_files2:
        print(filename)
        filename = file.split('/')[-1]
        label = index
        # 判断文件是否在
        if (filename in hash_table):
            print("same:" + filename)
            label = hash_table[filename]
        else:
            hash_table[filename] = index
            index = index + 1
        dataframe = pd.read_csv(file, header=None)
        # 提取域名和标签列，域名编码
        # 按列切割，只要两列，一列域名，一列标签
        dataframe = dataframe.iloc[:, [0, -1]]
        dataframe.columns = [0, 1]
        # 处理大写
        dataframe[0] = dataframe[0].str.lower()
        # 逐一编码
        dataframe[0] = dataframe[0].apply(AlpMapDigits)
        label_list = [label] * dataframe.shape[0]
        df = pd.DataFrame({'domainname_vec': dataframe[0], 'label': label_list})
        df.to_csv(target_csv_path_2 + '/' + filename, index=False, header=False)

    return 0


if __name__ == '__main__':
    print(1)
    Set_label_list_form_benign(f'../../data/Benign', f'../../data/Benign_vec/Benign.csv')
    Set_label_list_form_malicious(f'../../data/DGA/2016-09-19-dgarchive_full',
                                  f'../../data/DGA/2020-06-19-dgarchive_full',
                                  f'../../data/DGA_vec/2016-09-19-dgarchive_full',
                                  f'../../data/DGA_vec/2020-06-19-dgarchive_full')
    # SetLabel(f'../data/DGA/2020-06-19-dgarchive_full', False)

    pass
