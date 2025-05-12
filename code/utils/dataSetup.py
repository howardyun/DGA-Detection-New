import os
from torchvision import datasets, transforms
from code.DGADataset import DGATrueDataset, DGAFalseDataset
from torch.utils.data import DataLoader, random_split

NUM_WORKERS = os.cpu_count()


# 创建dataLoader
def CreateDataloaders(data_dir: str,
                      flag: bool,
                      train_or_test: bool,
                      batch_size: int,
                      num_workers: int = NUM_WORKERS):
    """
    :param data_dir: 数据文件夹
    :param flag: 加载标签False或者标签True的方法
    :param train_or_test: 数据集逻辑是训练集还是测试集,True为训练集,默认为True
    :param batch_size: 分块
    :param num_workers: cpu线程工作数量
    :return:
    """
    # 加载不同标签的数据集
    print("获取数据集")
    if train_or_test:
        data_set = DGATrueDataset(data_dir, flag)
        pass
    else:
        data_set = DGAFalseDataset(data_dir, flag)
        pass
    dataset_size = len(data_set)

    # 划分数据集为80%训练集，20%验证集
    print("划分数据集")
    train_size = int(0.8 * dataset_size)
    test_size = dataset_size - train_size
    train_dataset, test_dataset = random_split(data_set,
                                               [train_size, test_size])

    # 创建dataLoader
    print("创建dataLoader")
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_dataloader, test_dataloader
    pass
