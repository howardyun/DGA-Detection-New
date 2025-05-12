import random
from pathlib import Path
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


def remain_data(benign_root_csv_path, malicious_root_csv_path, year):
    """
    读取剩余到的40%数据作为预测集
    :param benign_root_csv_path: 良性训练集
    :param malicious_root_csv_path: 恶性训练集
    :param year: 年份，2016还是2020
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
        predict_df = pd.concat([predict_df, pd.read_csv(file, header=None)], ignore_index=True)
        pass
    for file in csv_files_malicious:
        predict_df = pd.concat([predict_df, pd.read_csv(file, header=None)], ignore_index=True)
        pass

    # 打乱
    predict_df = predict_df.sample(frac=1).reset_index(drop=True)
    predict_df.to_csv(predict_file_path, index=None, header=None, mode='w')
    pass


def extract_remain_data(benign_root_csv_path, malicious_root_csv_path, year):
    """
    读取抽取到的60%数据,分割80%作为训练集,20%作为测试集
    :param benign_root_csv_path: 良性训练集
    :param malicious_root_csv_path: 恶性训练集
    :param year: 年份，2016还是2020
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

    benign_df = pd.DataFrame()
    malicious_df = pd.DataFrame()
    # 全部文件数组
    csv_files_benign = glob.glob(os.path.join(benign_root_csv_path, '*.csv'))
    csv_files_malicious = glob.glob(os.path.join(malicious_root_csv_path, '*.csv'))
    # 数据帧
    for file in csv_files_benign:
        benign_df = pd.concat([benign_df, pd.read_csv(file, header=None)], ignore_index=True)
        pass
    for file in csv_files_malicious:
        malicious_df = pd.concat([malicious_df, pd.read_csv(file, header=None)], ignore_index=True)
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
    print(f'pos weight: benign / malicious = {len(benign_train_df) / len(malicious_train_df)}')
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


if __name__ == '__main__':
    benign_root_csv_path = '../../data/Benign_vec'
    malicious_root_csv_path = '../../data/DGA_vec/2016-09-19-dgarchive_full'
    target_dir = "../../data/MiniDataset"
    if not os.path.exists(target_dir):
        os.makedirs(target_dir, exist_ok=True)
        pass
    # 保存文件路径
    train_file_path = "../../data/MiniDataset/" + '' + "/train.csv"
    test_file_path = "../../data/extract_remain_data/" + '' + "/test.csv"

    benign_df = pd.DataFrame()
    malicious_df = pd.DataFrame()
    # 全部文件数组
    csv_files_benign = glob.glob(os.path.join(benign_root_csv_path, '*.csv'))
    csv_files_malicious = glob.glob(os.path.join(malicious_root_csv_path, '*.csv'))
    # 数据帧
    for file in csv_files_benign:
        benign_df = pd.concat([benign_df, pd.read_csv(file, header=None)], ignore_index=True)
        pass
    for file in csv_files_malicious:
        malicious_df = pd.concat([malicious_df, pd.read_csv(file, header=None)], ignore_index=True)
        pass

    # 防止抽取时抽取出聚块的数据
    benign_df = benign_df.sample(frac=1).reset_index(drop=True)
    malicious_df = malicious_df.sample(frac=1).reset_index(drop=True)

    # 从抽取的60%数据中八二分割
    split_index = 1000
    benign_train_df = benign_df[:split_index]
    benign_test_df = benign_df[:split_index]

    split_index = 1000
    malicious_train_df = malicious_df[:split_index]
    malicious_test_df = malicious_df[:split_index]

    # 组合训练集和测试集数据
    train_data = pd.concat([benign_train_df, malicious_train_df], ignore_index=True)
    test_data = pd.concat([benign_test_df, malicious_test_df], ignore_index=True)
    print(f'pos weight: benign / malicious = {len(benign_train_df) / len(malicious_train_df)}')
    train_data = train_data.sample(frac=1).reset_index(drop=True)
    test_data = test_data.sample(frac=1).reset_index(drop=True)
    train_data.to_csv(train_file_path, index=None, header=None, mode='w')
    test_data.to_csv(test_file_path, index=None, header=None, mode='w')
