import csv
import datetime
import time

import torch
from pathlib import Path
from datetime import datetime


def GetCurrentTime():
    """
    获取当前时间,用于成为文件夹名字
    :return:
    """
    # 生成时间戳作为文件夹名
    # 防止多个进程同时写入文件时出错
    timestamp = int(time.time())
    # 将时间戳转换为 datetime 对象
    datetime_obj = datetime.fromtimestamp(timestamp)
    # 格式化日期时间字符串
    formatted_datetime = datetime_obj.strftime("%Y_%m_%d_%H_%M_%S")
    return str(formatted_datetime)
    pass


def SaveModelPath(target_dir: str, current_name: str, lb_flag: bool, bin_or_multi: bool):
    # 创建第一层目标文件夹
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)
    # 创建第二层文件夹,如果是鲁棒性测试就是lb,不是就是dga
    if lb_flag:
        # 第二层路径
        second_dir = 'lb'
        second_dir_path = target_dir_path / second_dir
        second_dir_path.mkdir(parents=True, exist_ok=True)
        pass
    else:
        # 第二层路径
        second_dir = 'dga'
        second_dir_path = target_dir_path / second_dir
        second_dir_path.mkdir(parents=True, exist_ok=True)
        pass
    # 第三层路径
    # 获取当前文件下总数
    total = 0
    for entry in second_dir_path.iterdir():
        total += 1
        pass
    # 日期文件夹
    third_dir = f"{total}_" + f"{current_name}_" + GetCurrentTime() if bin_or_multi else f"{total}_multi_" + f"{current_name}_" + GetCurrentTime()
    third_dir_path = second_dir_path / third_dir
    third_dir_path.mkdir(parents=True, exist_ok=True)
    model_save_path = third_dir_path
    return model_save_path
    pass


# 保存模型工具函数
def SaveModel(model: torch.nn.Module, model_name: str, model_save_path: str):
    """
    :param model: 保存的模型
    :param model_name: 模型名.pth结尾
    :param model_save_path:
    :param lb_flag: 是否为鲁棒性测试,
    :return:
    """
    # 断言保证文件类型
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "模型结尾需为 'pt' or 'pth'"
    model_save_path = Path(model_save_path) / model_name

    # 保存model模型参数字典
    print(f"保存模型路径: {model_save_path}")
    torch.save(obj=model.state_dict(), f=model_save_path)
    pass


# 加载模型参数函数
def LoadModel(model: torch.nn.Module, target_dir: str, model_name: str):
    """
    :param model: 要加载参数的模型
    :param target_dir: 模型文件夹
    :param model_name: 模型名
    :return:
    """
    target_dir_path = Path(target_dir)
    # 断言保证文件类型
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "模型结尾需为 'pt' or 'pth'"
    model_load_path = target_dir_path / model_name

    model.load_state_dict(torch.load(f=model_load_path))
    return model
    pass


def SaveResultsPath(current_name: str, lb_flag: bool, bin_or_multi: bool):
    # 创建模型记录文件夹
    first_dir_path = Path("../modelRecord/train") if bin_or_multi else Path("../modelMultiRecord/train")
    first_dir_path.mkdir(parents=True, exist_ok=True)

    # 路径为dga还是lb
    second_dir_path = (
        Path("../modelRecord/train/lb") if lb_flag else Path("../modelRecord/train/dga")) if bin_or_multi else Path(
        "../modelMultiRecord/train/dga")
    second_dir_path.mkdir(parents=True, exist_ok=True)

    # 获取当前文件下总数
    total = 0
    for entry in second_dir_path.iterdir():
        total += 1
        pass
    # 日期文件夹
    third_dir = f"{total}_" + f"{current_name}_" + GetCurrentTime() if bin_or_multi else f"{total}_multi" + f"{current_name}_" + GetCurrentTime()
    third_dir_path = second_dir_path / third_dir
    third_dir_path.mkdir(parents=True, exist_ok=True)

    # 最终统计的csv文件
    csv_file = "record.csv"
    csv_file_path = third_dir_path / csv_file
    return csv_file_path
    pass


def SaveResults(model_name: str, model_epoch: int, results: dict, csv_file_path: str):
    """
    保存模型训练数据
    :param model_name: 模型名字
    :param model_epoch: 模型训练次数
    :param results: 结果集
    :param csv_file_path: 结果存放文件夹
    :return:
    """

    # 不存在建立文件
    if not Path(csv_file_path).exists():
        with open(str(csv_file_path), mode='w', newline="") as csvfile:
            pass
        pass

    # 写入标题
    csv_title = [model_name, "train_loss", "train_acc", "test_loss", "test_acc", 'test_precision', 'test_recall',
                 'test_f1']
    with open(str(csv_file_path), mode='a', newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(csv_title)
        pass

    # 写入一行数据
    train_loss, train_acc, test_loss, test_acc, test_precision, test_recall, test_f1 = results['train_loss'], results[
        'train_acc'], results['test_loss'], \
        results['test_acc'], results['test_precision'], results['test_recall'], results['test_f1']
    for index in range(model_epoch):
        # 一行数据
        csv_item = [f"epoch: {index + 1}", train_loss[index], train_acc[index], test_loss[index], test_acc[index],
                    test_precision[index], test_recall[index], test_f1[index]]
        with open(str(csv_file_path), mode='a', newline="") as csvfile:
            writer = csv.writer(csvfile)
            # 写入一行数据
            writer.writerow(csv_item)
            pass
        pass
    pass
