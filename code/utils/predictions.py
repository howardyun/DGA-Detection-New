import csv
import os.path
import time
from datetime import datetime
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
# from DataIterator import DataIterator
# from code.DataIterator import DataIterator
import pandas as pd
import torch
from torch.utils.data import DataLoader
from DGADataset_ysx import DGATrueDataset_ysx


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


def SavePredictionsResults(results: dict, lb_flag: bool):
    """
    方法遗弃
    :param results: 结果集
    :param lb_flag: 是否为鲁棒性标识
    :return:
    """
    # 创建模型记录文件夹
    first_dir_path = Path("../modelRecord/predict")
    first_dir_path.mkdir(parents=True, exist_ok=True)

    # 路径为dga还是lb
    second_dir_path = Path("../modelRecord/predict/lb") if lb_flag else Path("../modelRecord/predict/dga")
    second_dir_path.mkdir(parents=True, exist_ok=True)

    # 日期文件夹
    third_dir = GetCurrentTime()
    third_dir_path = second_dir_path / third_dir
    third_dir_path.mkdir(parents=True, exist_ok=True)

    # 最终统计的csv文件
    csv_file = "record.csv"
    csv_file_path = third_dir_path / csv_file
    # 不存在建立文件
    if not Path(csv_file_path).exists():
        with open(str(csv_file_path), mode='w', newline="") as csvfile:
            pass
        pass

    # 写入标题
    csv_title = ['predict model', 'predict name', "predict acc"]
    with open(str(csv_file_path), mode='a', newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(csv_title)
        pass

    # 写入一行数据
    model_name_list, file_name_list, predict_acc_list = results['predict model'], results['predict name'], results[
        'predict acc']
    for index in range(len(results['predict name'])):
        csv_item = [model_name_list[index], file_name_list[index], predict_acc_list[index]]
        with open(str(csv_file_path), mode='a', newline="") as csvfile:
            writer = csv.writer(csvfile)
            # 写入一行数据
            writer.writerow(csv_item)
            pass
        pass
    pass


def SaveFilePath(current_name: str, lb_flag: bool):
    """
    :param lb_flag:
    :return: 返回模型预测数据集时所有文件的存放地址
    """
    # 创建模型记录文件夹
    first_dir_path = Path("../modelRecord/predict")
    first_dir_path.mkdir(parents=True, exist_ok=True)

    # 路径为dga还是lb
    second_dir_path = Path("../modelRecord/predict/lb") if lb_flag else Path("../modelRecord/predict/dga")
    second_dir_path.mkdir(parents=True, exist_ok=True)

    # 日期文件夹
    # 获取当前文件下总数
    total = 0
    for entry in second_dir_path.iterdir():
        total += 1
        pass
    # 日期文件夹
    third_dir = f"{total}_" + f"{current_name}_" + GetCurrentTime()
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


def SaveResults(model_name, X, y, label, target_path):
    """
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


def SaveFamilyResults(model_name, file_path, X, y, label, target_path):
    """
    :param model_name: 模型名
    :param file_path: 家族文件路径
    :param X: X dga域名数据帧
    :param y: y 分类标签数据帧
    :param label: 预测数据结果数据帧
    :param target_path: 目标文件存放路径目录
    :return:
    """
    file_name = file_path.split('\\')[-1]
    # 每个模型预测结果文件路径
    file_path = str(target_path) + '/' + model_name + '/' + file_name
    # 写入文件
    if not Path(file_path).exists():
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        Path(file_path).touch()
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
        # 不使用pandas进行插入,高内存低io,用csv插入,低内存高io
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


def SaveAccPreF1(model_name: str, data_file_path: str, results_path: str, acc_pre_f1_path: str):
    """
    二分类手动计算准确率这些值,后可以更改成sklearn方法
    手动计算是为了分别查看0样本和1样本分别作为TP视角(主视角)时的效果
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
            df[0] = ['model_name', 'file_name', 'model_accuracy', 'model_precision_0', 'model_recall_0', 'model_f1_0',
                     'model_precision_1', 'model_recall_1', 'model_f1_1']
            df.to_csv(acc_pre_f1_path, index=False, header=False)
            pass
        pass
    # 数据帧
    # 计算准确率精准率f1
    df = pd.read_csv(results_path, usecols=[1, 2])
    # 变成tensor
    tensor_label = torch.tensor(df['label'].to_numpy())
    tensor_pred = torch.tensor(df[model_name].to_numpy())
    # 计算准确率
    accuracy = torch.eq(tensor_label, tensor_pred).sum().item() / len(tensor_label)
    # 计算查准率
    tp_0 = ((tensor_pred == 0) & (tensor_label == 0)).sum().item()
    fp_0 = ((tensor_pred == 0) & (tensor_label == 1)).sum().item()
    try:
        precision_0 = tp_0 / (tp_0 + fp_0)
    except ZeroDivisionError as e:
        precision_0 = 0
    # 计算召回率
    fn_0 = ((tensor_pred == 1) & (tensor_label == 0)).sum().item()
    try:
        recall_0 = tp_0 / (tp_0 + fn_0)
    except ZeroDivisionError as e:
        recall_0 = 0
    # 计算f1-score
    try:
        f1_0 = (2 * precision_0 * recall_0) / (precision_0 + recall_0)
    except ZeroDivisionError as e:
        f1_0 = 0

    # 计算查准率
    tp_1 = ((tensor_pred == 1) & (tensor_label == 1)).sum().item()
    fp_1 = ((tensor_pred == 1) & (tensor_label == 0)).sum().item()
    try:
        precision_1 = tp_1 / (tp_1 + fp_1)
    except ZeroDivisionError as e:
        precision_1 = 0
    # 计算召回率
    fn_1 = ((tensor_pred == 0) & (tensor_label == 1)).sum().item()
    try:
        recall_1 = tp_1 / (tp_1 + fn_1)
    except ZeroDivisionError as e:
        recall_1 = 0
    # 计算f1-score
    try:
        f1_1 = (2 * precision_1 * recall_1) / (precision_1 + recall_1)
    except ZeroDivisionError as e:
        f1_1 = 0

    # 插入数据
    df = pd.read_csv(acc_pre_f1_path, header=None)
    columns = len(df.columns.tolist())
    df[columns] = [model_name, data_file_path, accuracy, precision_0, recall_0, f1_0, precision_1, recall_1, f1_1]
    df.to_csv(acc_pre_f1_path, index=False, header=False)
    pass


def SaveFamilyAccPreF1(model_name: str, data_file_path: str, results_path: str, acc_pre_f1_path: str):
    """
    计算家族
    二分类手动计算准确率这些值,后可以更改成sklearn方法
    手动计算查看1样本作为TP视角(主视角)时的效果,因为家族预测时全是1样本(恶意数据),没有0样本(良性数据)
    :param model_name: 模型名
    :param data_file_path: 数据文件路径
    :param results_path: 模型记录文件路径
    :param acc_pre_f1_path: 准确率精准率F1-score等文件路径
    :return:
    """
    # 判断是否为新建文件
    if os.path.getsize(acc_pre_f1_path) == 0:
        # 写入标题
        df = pd.DataFrame(columns=['model_name', 'file_name', 'model_accuracy',
                                   'model_precision_1', 'model_recall_1', 'model_f1_1'], index=None)
        df.to_csv(str(acc_pre_f1_path), index=False)
        pass
    # 数据帧
    # 计算准确率精准率f1
    df = pd.read_csv(results_path, usecols=[1, 2])
    # 变成tensor
    tensor_label = torch.tensor(df['label'].to_numpy())
    tensor_pred = torch.tensor(df[model_name].to_numpy())
    # 计算准确率
    accuracy = torch.eq(tensor_label, tensor_pred).sum().item() / len(tensor_label)
    # 计算查准率
    tp_1 = ((tensor_pred == 1) & (tensor_label == 1)).sum().item()
    fp_1 = ((tensor_pred == 1) & (tensor_label == 0)).sum().item()
    try:
        precision_1 = tp_1 / (tp_1 + fp_1)
    except ZeroDivisionError as e:
        precision_1 = 0
    # 计算召回率
    fn_1 = ((tensor_pred == 0) & (tensor_label == 1)).sum().item()
    try:
        recall_1 = tp_1 / (tp_1 + fn_1)
    except ZeroDivisionError as e:
        recall_1 = 0
    # 计算f1-score
    try:
        f1_1 = (2 * precision_1 * recall_1) / (precision_1 + recall_1)
    except ZeroDivisionError as e:
        f1_1 = 0

    data_row = [model_name, data_file_path, accuracy, precision_1, recall_1, f1_1]
    df_row = pd.DataFrame([data_row])
    df_row.to_csv(acc_pre_f1_path, mode='a', header=False, index=False)
    pass


"""
增强上面的SaveFamilyAccPreF1, SaveAccPreF1两个方法
"""


def CalAccPreF1(y, label):
    """
    :param y: 标签
    :param label: 预测的标签
    :return:
    """
    accuracy, precision_0, recall_0, f1_0, precision_1, recall_1, f1_1 = 0, 0, 0, 0, 0, 0, 0
    # 用sklearn库直接计算
    # 准确率
    accuracy += accuracy_score(y, label)
    # 默认情况下1,1为TP,既将恶性样本预测为恶性的情况
    # 0,0为TN,既将良性样本预测为良性的情况
    precision_1 += precision_score(y, label, zero_division=0)
    recall_1 += recall_score(y, label, zero_division=0)
    f1_1 += f1_score(y, label, zero_division=0)

    # 但数据集中原本代表0的是良性样本,这里要将0,0设为TP;1,1设为TN
    # 特殊情况:0,0为TP,既将良性样本预测为良性
    # 1,1为TN,即将恶性样本预测为恶性情况
    # 要修改下pos_label
    precision_0 += precision_score(y, label, pos_label=0, zero_division=0)
    recall_0 += recall_score(y, label, pos_label=0, zero_division=0)
    f1_0 += f1_score(y, label, pos_label=0, zero_division=0)

    return accuracy, precision_0, recall_0, f1_0, precision_1, recall_1, f1_1
    pass


def CalSaveResult(accuracy, precision_0, recall_0, f1_0, precision_1, recall_1, f1_1, model_name, data_file_path,
                  acc_pre_f1_file_path):
    # 判断是否为新建文件
    if os.path.getsize(acc_pre_f1_file_path) == 0:
        # 写入标题
        with open(str(acc_pre_f1_file_path), mode='w', newline="") as csvfile:
            df = pd.DataFrame(index=None)
            df[0] = ['model_name', 'file_name', 'model_accuracy', 'model_precision_0', 'model_recall_0', 'model_f1_0',
                     'model_precision_1', 'model_recall_1', 'model_f1_1']
            df.to_csv(acc_pre_f1_file_path, index=False, header=False)
            pass
        pass

    # 插入数据
    df = pd.read_csv(acc_pre_f1_file_path, header=None)
    columns = len(df.columns.tolist())
    df[columns] = [model_name, data_file_path, accuracy, precision_0, recall_0, f1_0, precision_1, recall_1, f1_1]
    df.to_csv(acc_pre_f1_file_path, index=False, header=False)
    pass


def Predictions(model: torch.nn.Module,
                model_name: str,
                file: str,
                results_file_dir: str,
                acc_pre_f1_file_path: str,
                device: torch.device,
                full_flag: bool,
                BATCH_SIZE: int,
                partial_data=1000,
                lb_flag=False):
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
    :param lb_flag: 是否为鲁棒预测
    :return: 整体的准确率
    """
    # 预测数据的dataLoader
    predict_df = pd.read_csv(file) if full_flag else pd.read_csv(file, nrows=partial_data)
    dataloader = DataLoader(DGATrueDataset_ysx(predict_df, False), batch_size=BATCH_SIZE, shuffle=True)

    # 设备设置模型
    model.to(device)

    # 打开模型评估模式和推理模式
    model.eval()

    # 评估预测准确率等指标
    accuracy, precision_0, recall_0, f1_0, precision_1, recall_1, f1_1 = 0, 0, 0, 0, 0, 0, 0

    # 结果文件路径
    results_path = ''
    with torch.inference_mode():
        for batch, (X, y, dga_name) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            pred_logits = model(X).squeeze()
            y = y.float()

            # 这里没再次sigmoid，模型中已经激化过
            # 二分类训练计算
            pred_label = torch.round(pred_logits)

            try:
                # 返回本轮批次的结果
                current_accuracy, current_precision_0, current_recall_0, current_f1_0, current_precision_1, current_recall_1, current_f1_1 = CalAccPreF1(
                    y.cpu(), pred_label.cpu())
                pass
            except:
                pred_label = pred_label.unsqueeze(0)
                current_accuracy, current_precision_0, current_recall_0, current_f1_0, current_precision_1, current_recall_1, current_f1_1 = CalAccPreF1(
                    y.cpu(), pred_label.cpu())
                pass

            # 写入结果
            if lb_flag:
                results_path = SaveResults(model_name=model_name, X=dga_name, y=y.cpu(), label=pred_label.cpu(),
                                           target_path=results_file_dir)
                pass
            # 增强后不再有临时文件,而是累计结果
            accuracy += current_accuracy
            precision_0 += current_precision_0
            recall_0 += current_recall_0
            f1_0 += current_f1_0
            precision_1 += current_precision_1
            recall_1 += current_recall_1
            f1_1 += current_f1_1
            pass
        pass

    # 计算最终结果,一个模型预测结果
    # SaveAccPreF1(model_name=model_name, data_file_path=file, results_path=results_path,
    #              acc_pre_f1_path=acc_pre_f1_file_path)
    # 增强后直接将累计平均即可
    accuracy /= len(dataloader)
    precision_0 /= len(dataloader)
    recall_0 /= len(dataloader)
    f1_0 /= len(dataloader)
    precision_1 /= len(dataloader)
    recall_1 /= len(dataloader)
    f1_1 /= len(dataloader)
    CalSaveResult(accuracy, precision_0, recall_0, f1_0, precision_1, recall_1, f1_1, model_name, file,
                  acc_pre_f1_file_path)
    pass


def FindRow(csv_file, text):
    """
    :param csv_file: 查找文本的文件
    :param text: 查找文本
    """
    index = 0
    with open(csv_file, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        # 获取索引
        for i, row in enumerate(reader):
            if text in row:
                index = i
            pass
        pass

    file_list = []
    with open(csv_file, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        for i, row in enumerate(reader):
            if i > index:
                file_list.append(row[0].replace("../../", "../"))
                pass
            pass
        pass
    return file_list
    pass


def PredictionFamily(model: torch.nn.Module,
                     model_name: str,
                     file: str,
                     results_file_dir: str,
                     acc_pre_f1_file_path: str,
                     device: torch.device,
                     full_flag: bool,
                     BATCH_SIZE: int,
                     partial_data=1000):
    """
    预测dga家族
    :param model: 预测是用的模型
    :param model_name: 模型名
    :param file: 预测文件路径
    :param results_file_dir: 每个模型预测结果的文件目录路径
    :param acc_pre_f1_file_path: 每个模型预测结果统计的文件路径
    :param device: 设备
    :param full_flag: 家族预测是否用全数据集
    :param BATCH_SIZE: 批次数量
    :param partial_data: 非全数据集时需要的数据
    :return:
    """
    file_list = FindRow(file, "predict file")
    if full_flag:
        # 全数据集
        for file in file_list:
            # 配置数据
            predict_df = pd.read_csv(file)
            dataloader = DataLoader(DGATrueDataset_ysx(predict_df, False), batch_size=BATCH_SIZE, shuffle=True)

            # 设备设置模型
            model.to(device)

            # 打开模型评估模式和推理模式
            model.eval()

            # 评估预测准确率
            pred_acc = 0

            # 结果文件路径
            results_path = ''
            with torch.inference_mode():
                for batch, (X, y, dga_name) in enumerate(dataloader):
                    X, y = X.to(device), y.to(device)

                    pred_logits = model(X).squeeze()
                    y = y.float()

                    # 这里没再次sigmoid，模型中已经激化过
                    # 二分类训练计算
                    pred_label = torch.round(pred_logits)
                    try:
                        pred_acc += torch.eq(pred_label, y).sum().item() / len(pred_label)
                        pass
                    except:
                        pred_label = pred_label.unsqueeze(0)
                        pred_acc += torch.eq(pred_label, y).sum().item() / len(pred_label)
                        pass

                        # 写入结果
                    results_path = SaveFamilyResults(model_name=model_name, file_path=file, X=dga_name, y=y.cpu(),
                                                     label=pred_label.cpu(),
                                                     target_path=results_file_dir)
                    pass
                pass

            # 计算最终结果
            # 一个模型预测结果
            SaveFamilyAccPreF1(model_name=model_name, data_file_path=file, results_path=results_path,
                               acc_pre_f1_path=acc_pre_f1_file_path)
            pass
        pass
    else:
        # 部分数据集
        for file in file_list:
            # 配置数据
            # 非全数据集
            predict_df = pd.read_csv(file, nrows=partial_data)
            dataloader = DataLoader(DGATrueDataset_ysx(predict_df, False), batch_size=BATCH_SIZE, shuffle=True)

            # 设备设置模型
            model.to(device)

            # 打开模型评估模式和推理模式
            model.eval()

            # 评估预测准确率
            pred_acc = 0

            # 结果文件路径
            results_path = ''
            with torch.inference_mode():
                for batch, (X, y, dga_name) in enumerate(dataloader):
                    X, y = X.to(device), y.to(device)

                    pred_logits = model(X).squeeze()
                    y = y.float()

                    # 这里没再次sigmoid，模型中已经激化过
                    # 二分类训练计算
                    pred_label = torch.round(pred_logits)
                    try:
                        pred_acc += torch.eq(pred_label, y).sum().item() / len(pred_label)
                        pass
                    except:
                        pred_label = pred_label.unsqueeze(0)
                        pred_acc += torch.eq(pred_label, y).sum().item() / len(pred_label)
                        pass

                    # 写入结果
                    results_path = SaveFamilyResults(model_name=model_name, file_path=file, X=dga_name, y=y.cpu(),
                                                     label=pred_label.cpu(),
                                                     target_path=results_file_dir)
                    pass
                pass

            # 计算最终结果
            # 一个模型预测结果
            SaveFamilyAccPreF1(model_name=model_name, data_file_path=file, results_path=results_path,
                               acc_pre_f1_path=acc_pre_f1_file_path)
            pass
        pass
    pass
