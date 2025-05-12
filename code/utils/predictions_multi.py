import time
from datetime import datetime
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from DGADataset_ysx import DGATrueDataset_ysx
from torch.utils.data import DataLoader
import csv
import os
import pandas as pd
import torch


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


def SaveMultiFilePath(current_name: str, lb_flag: bool):
    """
    获取同一批模型训练存放路径
    :param current_name:
    :param lb_flag:
    :return: 返回模型预测数据集时所有文件的存放地址
    """
    # 创建模型记录文件夹
    first_dir_path = Path("../modelMultiRecord/predict")
    first_dir_path.mkdir(parents=True, exist_ok=True)

    # 路径为dga还是lb
    second_dir_path = Path("../modelMultiRecord/predict/lb") if lb_flag else Path("../modelMultiRecord/predict/dga")
    second_dir_path.mkdir(parents=True, exist_ok=True)

    # 日期文件夹
    # 获取当前文件下总数
    total = 0
    for entry in second_dir_path.iterdir():
        total += 1
        pass
    # 日期文件夹
    third_dir = f"{total}_multi_" + f"{current_name}_" + GetCurrentTime()
    third_dir_path = second_dir_path / third_dir
    third_dir_path.mkdir(parents=True, exist_ok=True)

    # 最终统计的csv文件
    acc_pre_f1_file = 'acc_pre_f1.csv'
    acc_pre_f1_file_path = third_dir_path / acc_pre_f1_file

    # 不存在建立文件
    if not Path(acc_pre_f1_file_path).exists():
        with open(str(acc_pre_f1_file_path), mode='w', newline="") as csvfile:
            pass
        pass
    # 返回文件路径
    return third_dir_path, acc_pre_f1_file_path
    pass


def SaveMultiResults(model_name, X, y, label, target_path):
    """
    多分类保存结果
    :param model_name: 模型名
    :param X: X dga域名tensor矩阵
    :param y: y 分类标签tensor矩阵
    :param label: 预测数据结果tensor矩阵
    :param target_path: 目标文件存放路径目录
    :return:
    """
    # 每个模型预测结果文件路径
    file_path = str(target_path) + '/' + model_name + '.csv'
    # 写入文件
    if not Path(file_path).exists():
        with open(str(file_path), mode='w', newline="") as csvfile:
            pass
        pass
    # 判断文件大小
    if os.path.getsize(file_path) == 0:
        # 写入标题
        df = pd.DataFrame(columns=['domain', 'label', model_name], index=None)
        df.to_csv(file_path, index=False)
        # 不使用pandas进行插入,高内存低io,用csv插入,低内存高io
        for dga_name, y, label in zip(X, y, label):
            row_item = [dga_name, y.item(), label.item()]
            with open(str(file_path), mode='a', newline="", errors='ignore') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(row_item)
                pass
            pass
        pass
    else:
        for dga_name, y, label in zip(X, y, label):
            row_item = [dga_name, y.item(), label.item()]
            with open(str(file_path), mode='a', newline="", errors='ignore') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(row_item)
                pass
            pass
        pass
    return file_path
    pass


def SaveMultiAccPreF1(model_name: str, data_file_path: str, results_path: str, acc_pre_f1_path: str):
    """
    多分类计算准确率精确率等,用sklearn方便一点
    这个方法评估的是模型整体对多分类的性能,还不是模型对每个多分类标签的性能
    :param model_name: 模型名
    :param data_file_path: 数据文件路径
    :param results_path: 模型记录文件路径
    :param acc_pre_f1_path: 准确率精准率F1-score等文件路径
    :return:
    """
    # 判断是否为新建文件
    if os.path.getsize(acc_pre_f1_path) == 0:
        # 写入标题
        with open(str(acc_pre_f1_path), mode='w', newline="") as csvfile:
            df = pd.DataFrame(index=None)
            df[0] = ['model_name', 'file_name', 'model_accuracy', 'model_precision', 'model_recall', 'model_f1']
            df.to_csv(acc_pre_f1_path, index=False, header=False)
            pass
        pass
    # 数据帧
    # 计算准确率精准率f1
    df = pd.read_csv(results_path, usecols=[1, 2])
    # 变成tensor
    tensor_label = torch.tensor(df['label'].to_numpy())
    tensor_pred = torch.tensor(df[model_name].to_numpy())
    # 用sklearn计算
    accuracy = accuracy_score(tensor_label, tensor_pred)
    # zero_division = 0表示没有计算出准确值时默认为零
    precision = precision_score(tensor_label, tensor_pred, average='macro', zero_division=0)
    recall = recall_score(tensor_label, tensor_pred, average='macro', zero_division=0)
    f1 = f1_score(tensor_label, tensor_pred, average='macro', zero_division=0)

    # 插入数据
    df = pd.read_csv(acc_pre_f1_path, header=None)
    columns = len(df.columns.tolist())
    df[columns] = [model_name, data_file_path, accuracy, precision, recall, f1]
    df.to_csv(acc_pre_f1_path, index=False, header=False)
    pass


# 开放一个计算多酚类AccPreF1的方法
def PredictionMulti(model: torch.nn.Module,
                    model_name: str,
                    file: str,
                    results_file_dir: str,
                    acc_pre_f1_file_path: str,
                    device: torch.device,
                    full_flag: bool,
                    BATCH_SIZE: int,
                    partial_data=1000):
    """
    :param model: 预测是用的模型
    :param model_name: 模型名字
    :param file: 预测文件路径
    :param results_file_dir: 每个模型预测结果的文件目录路径
    :param acc_pre_f1_file_path: 每个模型预测结果统计的文件路径
    :param device: 设备
    :param full_flag: 全数据集标志
    :param BATCH_SIZE: 批次数量
    :param partial_data: 非全数据集数据
    :return: 整体的准确率
    """
    # 预测数据的dataLoader
    predict_df = pd.read_csv(file) if full_flag else pd.read_csv(file, nrows=partial_data)
    # False拿到预测集,第二个False拿到多分类
    dataloader = DataLoader(DGATrueDataset_ysx(predict_df, False, False), batch_size=BATCH_SIZE, shuffle=True)

    # 设备设置模型
    model.to(device)

    # 打开模型评估模式和推理模式
    model.eval()

    # 评估预测准确率, 现在用于检测torch计算中tensor优化问题
    pred_acc = 0

    # 结果文件路径
    results_path = ''
    with torch.inference_mode():
        for batch, (X, y, dga_name) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            pred_logits = model(X).squeeze()
            y = y.float()

            # 多分类计算
            pred_label = torch.argmax(pred_logits, dim=1)
            try:
                pred_acc += torch.eq(pred_label, y).sum().item() / len(pred_label)
                pass
            except:
                pred_label = pred_label.unsqueeze(0)
                pred_acc += torch.eq(pred_label, y).sum().item() / len(pred_label)
                pass

            # 写入结果
            results_path = SaveMultiResults(model_name=model_name, X=dga_name, y=y.cpu(), label=pred_label.cpu(),
                                            target_path=results_file_dir)
            pass
        pass

    # 计算最终结果
    # 一个模型预测结果
    SaveMultiAccPreF1(model_name=model_name, data_file_path=file, results_path=results_path,
                      acc_pre_f1_path=acc_pre_f1_file_path)
    pass
