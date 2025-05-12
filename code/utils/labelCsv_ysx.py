import random
from pathlib import Path
import pandas as pd
import os
import glob
import csv

import torch

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
    print(root_csv_path)
    csv_files = glob.glob(os.path.join(root_csv_path, '*.csv'))
    print(csv_files)
    for file in csv_files:
        print(file)
        dataframe = pd.read_csv(file, header=None)
        # 提取域名和标签列，域名编码
        # 处理大写
        dataframe[1] = dataframe[1].str.lower()
        # 按列切割，只要两列，一列域名，一列标签
        dataframe = dataframe.iloc[:, 1:3]
        dataframe.columns = [0, 1]
        # 逐一编码
        label_list = [0] * dataframe.shape[0]
        # 二分类和多分类标签标注
        df = pd.DataFrame({'domainname_vec': dataframe[0], 'label_bin': label_list, 'label_multi': label_list})
        df.to_csv(target_csv_path, index=False, header=False)
        pass
    pass


def Set_label_list_form_malicious(root_csv_path_1, root_csv_path_2, target_csv_path_1, target_csv_path_2):
    """
    :param root_csv_path_1: csv文件小文件夹根路径
    :param root_csv_path_2: csv文件大文件夹根路径
    :param target_csv_path_1: 输出文件夹
    :param target_csv_path_2: 输出文件夹
    :return:
    """
    print("第一个文件夹:")
    # 用于记录已有的DGA家族
    hash_table = {}
    index = 1
    csv_files1 = glob.glob(os.path.join(root_csv_path_1, '*.csv'))
    for file in csv_files1:
        # 不同python版本对glob解释不同，如果只需要文件名，需要用os
        # filename = file.split('/')[-1]
        filename = os.path.basename(file)
        label = index
        # 判断文件是否在,如果在label等于之前的标签
        if filename in hash_table:
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
        label_list = [label] * dataframe.shape[0]
        # 二分类和多分类标签标注
        df = pd.DataFrame(
            {'domainname_vec': dataframe[0], 'label_bin': [1] * dataframe.shape[0], 'label_multi': label_list})
        df.to_csv(target_csv_path_1 + '/' + filename, index=False, header=False)

    print("第二个文件夹:")
    csv_files2 = glob.glob(os.path.join(root_csv_path_2, '*.csv'))
    for file in csv_files2:
        # 不同python版本对glob解释不同，如果只需要文件名，需要用os
        # filename = file.split('/')[-1]
        filename = os.path.basename(file)
        label = index
        # 判断文件是否在
        if filename in hash_table:
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
        label_list = [label] * dataframe.shape[0]
        df = pd.DataFrame(
            {'domainname_vec': dataframe[0], 'label_bin': [1] * dataframe.shape[0], 'label_multi': label_list})
        df.to_csv(target_csv_path_2 + '/' + filename, index=False, header=False)
        pass
    return 0


def mix_data_generate_train_test(benign_root_csv_path, malicious_root_csv_path, year, full_flag, partial_num=100000):
    """
    :param benign_root_csv_path: 良性数据集路径
    :param malicious_root_csv_path: 恶性数据集路径
    :param year: 年份表示
    :param full_flag: 是否为全数据集
    :param partial_num: 默认部分数据集时,每个文件提取数据数量
    :return:
    """
    csv_files_benign = glob.glob(os.path.join(benign_root_csv_path, '*.csv'))
    csv_files_malicious = glob.glob(os.path.join(malicious_root_csv_path, '*.csv'))

    dataframe_benign = pd.DataFrame()
    dataframe_malicious = pd.DataFrame()
    # 分别读入良性/恶意数据集
    for file in csv_files_benign:
        dataframe_benign = pd.concat([dataframe_benign, pd.read_csv(file, header=None, nrows=partial_num)],
                                     ignore_index=True)
        # dataframe_benign = pd.concat([dataframe_benign, pd.read_csv(file, header=None)], ignore_index=True)
    if full_flag:
        for file in csv_files_malicious:
            dataframe_malicious = pd.concat([dataframe_malicious, pd.read_csv(file, header=None)], ignore_index=True)
        pass
    else:
        for file in csv_files_malicious:
            dataframe_malicious = pd.concat([dataframe_malicious, pd.read_csv(file, header=None, nrows=partial_num)],
                                            ignore_index=True)
        pass
    benign_dataset_size = len(dataframe_benign)
    print(f"良性数据集大小: {benign_dataset_size}")
    malicious_dataset_size = len(dataframe_malicious)
    print(f"恶性数据集大小: {malicious_dataset_size}")
    print(f"pos weight: benign / malicious: {benign_dataset_size / malicious_dataset_size}")

    # 打乱数据集并且重置索引
    dataframe_benign = dataframe_benign.sample(frac=1).reset_index(drop=True)
    dataframe_malicious = dataframe_malicious.sample(frac=1).reset_index(drop=True)

    # 分割良性数据集
    split_index = int(0.8 * len(dataframe_benign))
    benign_train_df = dataframe_benign[:split_index]
    benign_test_df = dataframe_benign[split_index:]
    # 分割恶性数据集
    split_index = int(0.8 * len(dataframe_malicious))
    malicious_train_df = dataframe_malicious[:split_index]
    malicious_test_df = dataframe_malicious[split_index:]

    # 拼接数据帧
    train_df = pd.concat([benign_train_df, malicious_train_df], ignore_index=True)
    test_df = pd.concat([benign_test_df, malicious_test_df], ignore_index=True)

    # 打乱拼接好的数据帧
    train_df = train_df.sample(frac=1).reset_index(drop=True)
    print(f"Train shape: {train_df.shape}")
    test_df = test_df.sample(frac=1).reset_index(drop=True)
    print(f"Test shape: {test_df.shape}")

    if full_flag:
        train_df.to_csv('../../data/train' + year + '.csv', index=None, header=None)
        test_df.to_csv('../../data/test' + year + '.csv', index=None, header=None)
        pass
    else:
        train_df.to_csv('../../data/train_partial' + year + '.csv', index=None, header=None, mode='w')
        test_df.to_csv('../../data/test_partial' + year + '.csv', index=None, header=None, mode='w')
        pass
    pass


def extract_data(benign_root_csv_path, malicious_root_csv_path, year):
    """
    生成只有60%数据的每个文件
    :param benign_root_csv_path: 良性训练集
    :param malicious_root_csv_path: 恶性训练集
    :param year: 年份，2016还是2020
    :return:
    """
    # 创建文件夹
    benign_extract_dir = "../../data/extract_remain_data/" + year + "/benign/extract/"
    benign_remain_dir = "../../data/extract_remain_data/" + year + "/benign/remain/"
    malicious_extract_dir = "../../data/extract_remain_data/" + year + "/malicious/extract/"
    malicious_remain_dir = "../../data/extract_remain_data/" + year + "/malicious/remain/"
    if not os.path.exists(benign_extract_dir):
        os.makedirs(benign_extract_dir, exist_ok=True)
        pass
    if not os.path.exists(benign_remain_dir):
        os.makedirs(benign_remain_dir, exist_ok=True)
        pass
    if not os.path.exists(malicious_extract_dir):
        os.makedirs(malicious_extract_dir, exist_ok=True)
        pass
    if not os.path.exists(malicious_remain_dir):
        os.makedirs(malicious_remain_dir, exist_ok=True)
        pass

    # 全部文件数组
    csv_files_benign = glob.glob(os.path.join(benign_root_csv_path, '*.csv'))
    csv_files_malicious = glob.glob(os.path.join(malicious_root_csv_path, '*.csv'))
    for file in csv_files_benign:
        # 文件名
        filename = os.path.basename(file)
        # 数据帧
        df = pd.read_csv(file)
        extract_size = int(0.6 * len(df))

        # 随机抽取60%数据
        benign_extract = df.sample(n=extract_size)
        # 剩余40%数据
        benign_remain = df.drop(benign_extract.index)
        # 拼接抽取数据数据帧
        benign_extract.to_csv(benign_extract_dir + filename, index=None, header=None, mode='a')
        benign_remain.to_csv(benign_remain_dir + filename, index=None, header=None, mode='a')
        pass
    for file in csv_files_malicious:
        # 文件名
        filename = os.path.basename(file)
        # 数据帧
        df = pd.read_csv(file)
        # 六四分割
        extract_size = int(0.6 * len(df))

        # 随机抽取60%数据
        malicious_extract = df.sample(n=extract_size)
        # 剩余40%数据
        malicious_remain = df.drop(malicious_extract.index)

        # 拼接抽取数据数据帧
        malicious_extract.to_csv(malicious_extract_dir + filename, index=None, header=None, mode='a')
        malicious_remain.to_csv(malicious_remain_dir + filename, index=None, header=None, mode='a')
        pass
    pass


def remain_data(benign_root_csv_path, malicious_root_csv_path, year, pencentage=1.0):
    """
    读取剩余到的40%数据作为预测集
    :param benign_root_csv_path: 良性训练集
    :param malicious_root_csv_path: 恶性训练集
    :param year: 年份，2016还是2020
    :param pencentage: 抽取的比例
    """
    # 创建文件夹
    target_dir = "../../data/extract_remain_data/" + year
    if not os.path.exists(target_dir):
        os.makedirs(target_dir, exist_ok=True)
        pass
    predict_file_path = "../../data/extract_remain_data/" + year + "/predict.csv"

    predict_df = pd.DataFrame()
    # 全部文件数组
    csv_files_benign = glob.glob(os.path.join(benign_root_csv_path, '*.csv'))
    csv_files_malicious = glob.glob(os.path.join(malicious_root_csv_path, '*.csv'))
    # 数据帧
    for file in csv_files_benign:
        item_df = pd.read_csv(file, header=None)
        # 按比例抽取
        item_size = int(len(item_df) * pencentage)
        extract_df = item_df.sample(n=item_size)
        # 拼接
        predict_df = pd.concat([predict_df, extract_df], ignore_index=True)
        pass
    for file in csv_files_malicious:
        item_df = pd.read_csv(file, header=None)
        item_size = int(len(item_df) * pencentage)
        extract_df = item_df.sample(n=item_size)
        predict_df = pd.concat([predict_df, extract_df], ignore_index=True)
        pass

    # 打乱
    predict_df = predict_df.sample(frac=1).reset_index(drop=True)
    print(f"predict size: {len(predict_df)}")
    predict_df.to_csv(predict_file_path, index=None, header=None, mode='w')
    pass


def extract_remain_data(benign_root_csv_path, malicious_root_csv_path, year, pencentage=1.0):
    """
    读取抽取到的60%数据,分割80%作为训练集,20%作为测试集
    :param benign_root_csv_path: 良性训练集
    :param malicious_root_csv_path: 恶性训练集
    :param year: 年份，2016还是2020
    :param pencentage: 三千万数据比例,0.5就是提取一千五百万
    :return:
    """
    # 创建文件夹
    target_dir = "../../data/extract_remain_data/" + year
    if not os.path.exists(target_dir):
        os.makedirs(target_dir, exist_ok=True)
        pass
    # 保存文件路径
    train_file_path = "../../data/extract_remain_data/" + year + "/train.csv"
    test_file_path = "../../data/extract_remain_data/" + year + "/test.csv"
    # 最小数据集数据量
    limit = 10

    benign_df = pd.DataFrame()
    malicious_df = pd.DataFrame()
    # 全部文件数组
    csv_files_benign = glob.glob(os.path.join(benign_root_csv_path, '*.csv'))
    csv_files_malicious = glob.glob(os.path.join(malicious_root_csv_path, '*.csv'))
    # 数据帧
    for file in csv_files_benign:
        item_df = pd.read_csv(file, header=None)
        train_test_size = int(len(item_df) * pencentage)
        train_test_df = item_df.sample(n=train_test_size)
        # 拼接
        benign_df = pd.concat([benign_df, train_test_df], ignore_index=True)
        pass
    for file in csv_files_malicious:
        item_df = pd.read_csv(file, header=None)
        train_test_size = int(len(item_df) * pencentage)
        if train_test_size <= limit:
            train_test_df = item_df
            pass
        else:
            train_test_df = item_df.sample(n=train_test_size)
            pass
        malicious_df = pd.concat([malicious_df, train_test_df], ignore_index=True)
        pass
    # 防止抽取时抽取出聚块的数据
    benign_df = benign_df.sample(frac=1).reset_index(drop=True)
    malicious_df = malicious_df.sample(frac=1).reset_index(drop=True)
    # 从抽取的60%数据中八二分割
    split_index = int(0.8 * len(benign_df))
    benign_train_df = benign_df[:split_index]
    benign_test_df = benign_df[split_index:]
    split_index = int(0.8 * len(malicious_df))
    malicious_train_df = malicious_df[:split_index]
    malicious_test_df = malicious_df[split_index:]
    # 组合训练集和测试集数据
    train_data = pd.concat([benign_train_df, malicious_train_df], ignore_index=True)
    test_data = pd.concat([benign_test_df, malicious_test_df], ignore_index=True)
    print(
        f'train benign size: {len(benign_train_df)}, train malicious size: {len(malicious_train_df)}, pos weight: benign / malicious = {len(benign_train_df) / len(malicious_train_df)}')
    print(f'train size: {len(train_data)}, test size: {len(test_data)}')
    train_data = train_data.sample(frac=1).reset_index(drop=True)
    test_data = test_data.sample(frac=1).reset_index(drop=True)
    train_data.to_csv(train_file_path, index=None, header=None, mode='w')
    test_data.to_csv(test_file_path, index=None, header=None, mode='w')
    pass


def csv_write_row(src_file, text1, text2, array1, array2):
    """
    鲁棒性挑选的测试训练和预测文件名
    :param src_file:
    :param text1:
    :param text2:
    :param array1:
    :param array2:
    :return:
    """
    # 写入到CSV文件
    with open(src_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([text1])
        pass
    for i in range(len(array1)):
        with open(src_file, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([array1[i]])
            pass
        pass
    with open(src_file, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([text2])
        pass
    for i in range(len(array2)):
        with open(src_file, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([array2[i]])
            pass
        pass
    pass


def lb_data_generate_predict_percentage(benign_root_csv_path, malicious_root_csv_path, year='2016', percentage=1.0):
    """
    按比例生成预测集数据
    :param benign_root_csv_path:
    :param malicious_root_csv_path:
    :param year:
    :param percentage:
    :return:
    """
    predict_target_dir = '../../data/lb_predict_data/'
    if not os.path.exists(predict_target_dir):
        os.makedirs(predict_target_dir, exist_ok=True)
        pass
    # 全部文件数组
    csv_files_benign = glob.glob(os.path.join(benign_root_csv_path, '*.csv'))
    csv_files_malicious = glob.glob(os.path.join(malicious_root_csv_path, '*.csv'))

    # 随机获取恶性数据文件数组的一半数据
    # 打乱数组
    # random.shuffle(csv_files_malicious)
    # 获取分成两半的恶性数据文件数组
    split_index = len(csv_files_malicious) // 2
    # 测试训练文件名数组
    first_half = csv_files_malicious[:split_index]
    # 预测文件名数组
    second_half = csv_files_malicious[split_index:]

    # 组合鲁棒性预测数据
    lb_predict = pd.DataFrame()
    for file in second_half:
        item_df = pd.read_csv(file, header=None)
        predict_df_size = int(len(item_df) * percentage)
        predict_df = item_df.sample(n=predict_df_size)
        lb_predict = pd.concat([lb_predict, predict_df], ignore_index=True)
        pass
    # 无需分割和拼接,需要打乱数据帧
    predict_df = lb_predict.sample(frac=1).reset_index(drop=True)
    print(f"predict_df_size: {len(predict_df)}")
    # 写入csv
    predict_df.to_csv(predict_target_dir + 'lb_predict_' + year + '.csv', index=None, header=None, mode='w')
    pass
    pass


def lb_data_generate_train_test(benign_root_csv_path, malicious_root_csv_path, year, full_flag, partial_num=50000):
    """
    返回测试鲁棒性的全数据集数据
    鲁棒性测试训练数据组成: 对应年份一半恶性数据家族数据+全良性数据集
    鲁棒性预测数据组成:剩下一半恶性数据家族数据
    :param benign_root_csv_path: 良性训练集
    :param malicious_root_csv_path: 恶性训练集
    :param year: 年份，2016还是2020
    :param full_flag: true生成全数据集,false生成部分数据集
    :param partial_num: 部分数据集中提取每个文件的数据条数
    :return:
    """
    # 全数据集存放文件夹
    full_target_dir = "../../data/lb_full_data/"
    partial_target_dir = "../../data/lb_partial_data/"
    if full_flag:
        if not os.path.exists(full_target_dir):
            os.makedirs(full_target_dir)
            pass
        pass
    else:
        if not os.path.exists(partial_target_dir):
            os.makedirs(partial_target_dir)
            pass
        pass

    # 全部文件数组
    csv_files_benign = glob.glob(os.path.join(benign_root_csv_path, '*.csv'))
    csv_files_malicious = glob.glob(os.path.join(malicious_root_csv_path, '*.csv'))

    # 随机获取恶性数据文件数组的一半数据
    # 打乱数组
    # random.shuffle(csv_files_malicious)
    # 获取分成两半的恶性数据文件数组
    split_index = len(csv_files_malicious) // 2
    # 测试训练文件名数组
    first_half = csv_files_malicious[:split_index]
    # 预测文件名数组
    second_half = csv_files_malicious[split_index:]
    # 写入注记文件
    if full_flag:
        csv_write_row(full_target_dir + 'lb_type' + year + '.csv', 'train and test file', 'predict file', first_half,
                      second_half)
        pass
    else:
        csv_write_row(partial_target_dir + 'lb_type' + year + '.csv', 'train and test file', 'predict file', first_half,
                      second_half)
        pass

    # 组合鲁棒性测试训练数据
    lb_benign = pd.DataFrame()
    lb_malicious = pd.DataFrame()
    # 良性数据集提取全部数据
    for file in csv_files_benign:
        lb_benign = pd.concat([lb_benign, pd.read_csv(file, header=None)], ignore_index=True)
        pass
    if full_flag:
        # 恶性数据集提取全部数据
        for file in first_half:
            lb_malicious = pd.concat([lb_malicious, pd.read_csv(file, header=None)], ignore_index=True)
            pass
        print(f'full data pos weight: benign / malicious = {len(lb_benign) / len(lb_malicious)}')
        pass
    else:
        # 恶性数据集提取部分数据
        for file in first_half:
            lb_malicious = pd.concat([lb_malicious, pd.read_csv(file, header=None, nrows=partial_num)],
                                     ignore_index=True)
            pass
        print(f'partial data pos weight: benign / malicious = {len(lb_benign) / len(lb_malicious)}')
        pass

    # 分割良性数据集
    split_index = int(0.8 * len(lb_benign))
    benign_train_df = lb_benign[:split_index]
    benign_test_df = lb_benign[split_index:]
    # 分割恶性数据集
    split_index = int(0.8 * len(lb_malicious))
    malicious_train_df = lb_malicious[:split_index]
    malicious_test_df = lb_malicious[split_index:]
    # 拼接数据帧并且打乱数据帧
    train_df = pd.concat([benign_train_df, malicious_train_df], ignore_index=True)
    test_df = pd.concat([benign_test_df, malicious_test_df], ignore_index=True)
    train_df = train_df.sample(frac=1).reset_index(drop=True)
    test_df = test_df.sample(frac=1).reset_index(drop=True)
    # 写入csv
    if full_flag:
        train_df.to_csv(full_target_dir + 'lb_train' + year + '.csv', index=None, header=None, mode='w')
        test_df.to_csv(full_target_dir + 'lb_test' + year + '.csv', index=None, header=None, mode='w')
        pass
    else:
        train_df.to_csv(partial_target_dir + 'lb_train' + year + '.csv', index=None, header=None, mode='w')
        test_df.to_csv(partial_target_dir + 'lb_test' + year + '.csv', index=None, header=None, mode='w')
        pass

    # 组合鲁棒性预测数据
    lb_predict = pd.DataFrame()
    if full_flag:
        for file in second_half:
            lb_predict = pd.concat([lb_predict, pd.read_csv(file, header=None)], ignore_index=True)
            pass
        pass
    else:
        for file in second_half:
            lb_predict = pd.concat([lb_predict, pd.read_csv(file, header=None, nrows=partial_num)], ignore_index=True)
            pass
        pass
    # 无需分割和拼接,需要打乱数据帧
    predict_df = lb_predict.sample(frac=1).reset_index(drop=True)
    # 写入csv
    if full_flag:
        predict_df.to_csv(full_target_dir + 'lb_predict' + year + '.csv', index=None, header=None, mode='w')
        pass
    else:
        predict_df.to_csv(partial_target_dir + 'lb_predict' + year + '.csv', index=None, header=None, mode='w')
        pass

    pass


def multi_label(csv_path: str):
    """
    计算多分类中每一个标签的权重
    :param csv_path:
    :return:
    """
    # 数据帧
    target_df = pd.read_csv(csv_path, header=None)
    # 转换为numpy数组
    targets = target_df[2].to_numpy()
    # 每个类别的样本数量
    class_counts = torch.bincount(torch.tensor(targets))
    # class_proportions = class_counts / class_counts.sum()
    # # 将类别转化为权重
    # class_weights = class_proportions / class_proportions.sum()
    # 计算每个类别的样本数量的倒数作为权重
    class_weights = 1.0 / class_counts
    normalized_weights = class_weights / class_weights.sum()
    print("class_weights:", normalized_weights)
    print("shape", normalized_weights.shape)
    return normalized_weights
    pass


def multi_data(benign_root_csv_path, malicious_root_csv_path, year, pencentage=1.0):
    """
    :param benign_root_csv_path:  良性数据集目录
    :param malicious_root_csv_path:  恶性数据集目录
    :param year: 年份
    :param pencentage: 数据比例
    :return:
    """
    target_dir = '../../data/multi_data/' + year + '/'
    if not os.path.exists(target_dir):
        os.makedirs(target_dir, exist_ok=True)
        pass
    # 保存文件路径
    train_file_path = target_dir + 'train.csv'
    test_file_path = target_dir + 'test.csv'
    predict_file_path = target_dir + 'predict.csv'
    # 数据帧最少抽取的数量
    limit = 10

    # 三个数据帧,不在生成临时文件,直接抽取
    benign_final_df = pd.DataFrame()
    malicious_final_df = pd.DataFrame()
    predict_final_df = pd.DataFrame()

    benign_train_df = pd.DataFrame()
    benign_test_df = pd.DataFrame()
    malicious_train_df = pd.DataFrame()
    malicious_test_df = pd.DataFrame()
    # 全部文件数组
    csv_files_benign = glob.glob(os.path.join(benign_root_csv_path, '*.csv'))
    csv_files_malicious = glob.glob(os.path.join(malicious_root_csv_path, '*.csv'))
    # 构建数据帧
    for file in csv_files_benign:
        # 读取数据形成数据帧,获取每个文件大小
        # 训练测试集抽取pencentage
        item_df = pd.read_csv(file, header=None)
        train_test_size = int(len(item_df) * pencentage)
        train_test_df = item_df.sample(n=train_test_size)
        # 拼接
        split_index = int(0.8 * len(train_test_df))
        train_df = train_test_df[:split_index]
        test_df = train_test_df[split_index:]
        benign_train_df = pd.concat([benign_train_df, train_df], ignore_index=True)
        benign_test_df = pd.concat([benign_test_df, test_df], ignore_index=True)
        # benign_final_df = pd.concat([benign_final_df, train_test_df], ignore_index=True)

        # 从剩下部分抽取pencentage
        remain_df = item_df.drop(train_test_df.index)
        predict_size = int(len(remain_df) * pencentage)
        predict_df = remain_df.sample(n=predict_size)
        # 拼接
        predict_final_df = pd.concat([predict_final_df, predict_df], ignore_index=True)
        pass
    for file in csv_files_malicious:
        # 读取数据形成数据帧,获取每个文件大小
        # 训练集和测试集从每个文件抽取pencentage,预测集抽取剩下数据中的pencentage
        item_df = pd.read_csv(file, header=None)
        train_test_size = int(len(item_df) * pencentage)
        # 抽取训练集和测试集
        train_test_df = item_df.sample(n=train_test_size)
        # 拼接
        # split_index = int(0.8 * len(train_test_df))
        # 只有数据帧总数大于1的时候才实现分割，否则就是抽取相同的一个
        if train_test_size <= limit:
            # 低于最小情况就不抽取数据
            train_df = item_df
            test_df = item_df
            pass
        else:
            # 否则，才进行抽取
            split_index = int(0.8 * len(train_test_df))
            train_df = train_test_df[:split_index]
            test_df = train_test_df[split_index:]
            pass
        malicious_train_df = pd.concat([malicious_train_df, train_df], ignore_index=True)
        malicious_test_df = pd.concat([malicious_test_df, test_df], ignore_index=True)

        # 剥离前面这些数据
        remain_df = item_df.drop(train_test_df.index)
        predict_size = int(len(remain_df) * pencentage)
        if predict_size < limit:
            # 少于最小值的时候就不抽取
            predict_df = item_df
            pass
        else:
            # 否则，开始抽取
            # 从剩下数据中再抽pencentage作为预测数据
            predict_df = remain_df.sample(n=predict_size)
            pass

        # 拼接数据
        predict_final_df = pd.concat([predict_final_df, predict_df], ignore_index=True)
        if len(train_df) == 0 or len(test_df) == 0 or len(predict_df) == 0:
            print("file-->malicious train and test, predict:", file, len(train_df), len(test_df), len(predict_df))
            pass
        pass

    # # 防止抽取时抽取出聚块的数据
    # benign_final_df = benign_final_df.sample(frac=1).reset_index(drop=True)
    # malicious_final_df = malicious_final_df.sample(frac=1).reset_index(drop=True)
    # # 测试训练集八二分割
    # split_index = int(0.8 * len(benign_final_df))
    # benign_train_df = benign_final_df[:split_index]
    # benign_test_df = benign_final_df[split_index:]
    # split_index = int(0.8 * len(malicious_final_df))
    # malicious_train_df = malicious_final_df[:split_index]
    # malicious_test_df = malicious_final_df[split_index:]

    # 组合训练测试集数据
    train_data = pd.concat([benign_train_df, malicious_train_df], ignore_index=True)
    test_data = pd.concat([benign_test_df, malicious_test_df], ignore_index=True)
    print(f'train benign size: {len(benign_train_df)}, train malicious size: {len(malicious_train_df)}')
    print(f'train size: {len(train_data)}, test size: {len(test_data)}')
    # 打乱
    train_data = train_data.sample(frac=1).reset_index(drop=True)
    test_data = test_data.sample(frac=1).reset_index(drop=True)
    # 生成文件
    train_data.to_csv(train_file_path, index=None, header=None, mode='w')
    test_data.to_csv(test_file_path, index=None, header=None, mode='w')

    # 预测集,不用组合
    # 打乱
    predict_data = predict_final_df.sample(frac=1).reset_index(drop=True)
    print(f'predict size: {len(predict_data)}')
    predict_data.to_csv(predict_file_path, index=None, header=None, mode='w')
    pass


if __name__ == '__main__':
    # 给良性数据集打标签
    # Set_label_list_form_benign(f'../../data/Benign', f'../../data/Benign_vec/Benign.csv')
    # # 给恶意数据集打标签
    # Set_label_list_form_malicious(f'../../data/DGA/2016-09-19-dgarchive_full',
    #                               f'../../data/DGA/2020-06-19-dgarchive_full',
    #                               f'../../data/DGA_vec/2016-09-19-dgarchive_full',
    #                               f'../../data/DGA_vec/2020-06-19-dgarchive_full')
    # SetLabel(f'../data/DGA/2020-06-19-dgarchive_full', False)

    # 组合良性数据集和恶性数据集,全数据集
    # mix_data_generate_train_test(f'../../data/Benign_vec',
    #                              f'../../data/DGA_vec/2016-09-19-dgarchive_full', '2016', True)
    # mix_data_generate_train_test(f'../../data/Benign_vec',
    #                              f'../../data/DGA_vec/2016-09-19-dgarchive_full', '2016', False)

    # 随机抽取60 % 剩余40 % 数据
    # extract_data(f'../../data/Benign_vec',
    #              f'../../data/DGA_vec/2016-09-19-dgarchive_full', '2016')

    # # 生成抽取后的预测数据
    # 两千万
    # remain_data(f'../../data/extract_remain_data/2016/benign/remain/',
    #             f'../../data/extract_remain_data/2016/malicious/remain', '2016')
    # 一千万
    # remain_data(f'../../data/extract_remain_data/2016/benign/remain/',
    #             f'../../data/extract_remain_data/2016/malicious/remain', '2016_10000000', 0.5)
    # 五百万
    # remain_data(f'../../data/extract_remain_data/2016/benign/remain/',
    #             f'../../data/extract_remain_data/2016/malicious/remain', '2016_5000000', 0.25)
    # 一百万
    # remain_data(f'../../data/extract_remain_data/2016/benign/remain/',
    #             f'../../data/extract_remain_data/2016/malicious/remain', '2016_1000000', 0.05)

    # # 生成抽取后的训练测试数据
    # 这里抽取后的源数据集是三千万级别,生成比例需要根据三千万进行调整
    # 三千万
    # extract_remain_data(f'../../data/extract_remain_data/2016/benign/extract/',
    #                     f'../../data/extract_remain_data/2016/malicious/extract', '2016')
    # 一千万
    # extract_remain_data(f'../../data/extract_remain_data/2016/benign/extract/',
    #                     f'../../data/extract_remain_data/2016/malicious/extract', '2016_10000000', 0.33)
    # 五百万
    # extract_remain_data(f'../../data/extract_remain_data/2016/benign/extract/',
    #                     f'../../data/extract_remain_data/2016/malicious/extract', '2016_5000000', 0.17)
    # 一百万
    # extract_remain_data(f'../../data/extract_remain_data/2016/benign/extract/',
    #                     f'../../data/extract_remain_data/2016/malicious/extract', '2016_1000000', 0.033)

    # 生成全数据集鲁棒性数据
    # lb_data_generate_train_test(f'../../data/Benign_vec',
    #                             f'../../data/DGA_vec/2016-09-19-dgarchive_full', '2016', True)
    # lb_data_generate_train_test(f'../../data/Benign_vec',
    #                             f'../../data/DGA_vec/2016-09-19-dgarchive_full', '2016', False, 100000)
    # 单独按比例生成预测集数据
    lb_data_generate_predict_percentage(f'../../data/Benign_vec',
                                        f'../../data/DGA_vec/2016-09-19-dgarchive_full', '2016', 0.05)

    # 生成多分类训练测试数据
    # 三千万
    # multi_data(f'../../data/Benign_vec', f'../../data/DGA_vec/2016-09-19-dgarchive_full',
    #            '2016_30000000', 0.6)
    # 一千万
    # multi_data(f'../../data/Benign_vec', f'../../data/DGA_vec/2016-09-19-dgarchive_full',
    #            '2016_10000000', 0.2)
    # 五百万
    # multi_data(f'../../data/Benign_vec', f'../../data/DGA_vec/2016-09-19-dgarchive_full',
    #            '2016_5000000', 0.1)
    # 一百万
    # multi_data(f'../../data/Benign_vec', f'../../data/DGA_vec/2016-09-19-dgarchive_full',
    #            '2016_1000000', 0.021)
    pass
