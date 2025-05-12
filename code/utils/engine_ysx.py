import csv
import os
import string
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from typing import Dict, List, Tuple
from itertools import islice
from DataIterator import DataIterator, MultiDataIterator
from DGADataset_ysx import DGATrueDataset_ysx


# 单步模型训练函数
def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device) -> Tuple[float, float]:
    """
    :param model: 要训练的pytorch模型
    :param dataloader: 训练模型dataLoader实例
    :param loss_fn: pytorch损失函数
    :param optimizer: pytorch优化器
    :param device: 目标设备,cuda或者cpu
    :return:
    """
    # 模型进入训练模式
    model.train()

    # 训练损失值和训练准确值
    train_loss, train_acc = 0, 0

    # 抽取dataLoader中的数据
    for batch, (X, y) in enumerate(dataloader):
        # 设备无关代码
        X, y = X.to(device), y.to(device)

        # 预测
        y_pred = model(X).squeeze()
        y = y.float()

        # 计算和累积损失
        try:
            loss = loss_fn(y_pred, y)
            train_loss += loss.item()
            pass
        except:
            y_pred = torch.unsqueeze(y_pred, dim=0)
            loss = loss_fn(y_pred, y)
            train_loss += loss.item()
            pass

        # 这里没再次sigmoid，模型中已经激化过
        # 二分类训练计算
        y_label = torch.round(y_pred)
        train_acc += torch.eq(y_label, y).sum().item() / len(y_label)

        # 优化器设置零梯度
        optimizer.zero_grad()

        # 反向求导
        loss.backward()

        # 优化器步进
        optimizer.step()

        pass

    # 调整指标以获得每个批次的平均损失和准确性
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc
    pass


# 模型名
global_model_name = ""
# 训练源文件
global_test_file = ""
# 训练结果
global_target_dir = ""
# 又训练结果产生的F1
global_acc_pre_f1_file_path = ""


# 单步模型测试函数
def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              device: torch.device):
    """
    :param model: 要训练的pytorch模型
    :param dataloader: 训练模型dataLoader实例
    :param loss_fn: pytorch损失函数
    :param device: 目标设备,cuda或者cpu
    :return:
    """
    model.eval()

    test_loss, test_acc = 0, 0
    # 准确率等
    accuracy, precision, recall, f1 = 0, 0, 0, 0
    results_path = ""

    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            test_pred_logits = model(X).squeeze()
            y = y.float()

            # 处理tensor张量计算失误问题
            try:
                loss = loss_fn(test_pred_logits, y)
                test_loss += loss.item()
                pass
            except:
                test_pred_logits = torch.unsqueeze(test_pred_logits, dim=0)
                loss = loss_fn(test_pred_logits, y)
                test_loss += loss.item()
                pass

            # 这里没再次sigmoid，模型中已经激化过
            # 二分类训练计算
            test_label = torch.round(test_pred_logits)
            # 调整计算
            y, test_label = y.cpu().detach().numpy(), test_label.cpu().detach().numpy()
            # 用sklearn库直接计算
            accuracy += accuracy_score(y, test_label)
            precision += precision_score(y, test_label, zero_division=0)
            recall += recall_score(y, test_label, zero_division=0)
            f1 += f1_score(y, test_label, zero_division=0)
            pass
        pass

    test_loss = test_loss / len(dataloader)
    # 准确率
    test_acc = accuracy / len(dataloader)
    test_pre = precision / len(dataloader)
    test_rec = recall / len(dataloader)
    test_f1 = f1 / len(dataloader)
    return test_loss, test_acc, test_pre, test_rec, test_f1
    pass


# 主要训练函数
def train_ysx(model: torch.nn.Module,
              train_file: string,
              test_file: string,
              optimizer: torch.optim.Optimizer,
              loss_fn: torch.nn.Module,
              epochs: int,
              device: torch.device,
              BATCH_SIZE: int) -> Dict[str, List]:
    """
    :param model: pytorch模型
    :param train_file: 训练数据文件
    :param test_file: 测试数据文件
    :param optimizer: 优化器
    :param loss_fn: 损失函数
    :param epochs: 训练次数
    :param device: 目标设备
    :param BATCH_SIZE: 训练批次
    :return:
    """
    # 最终需要的准确率数据
    results = {"train_loss": [],
               "train_acc": [],
               "test_loss": [],
               "test_acc": [],
               "test_precision": [],
               "test_recall": [],
               "test_f1": []
               }

    # 设备无关代码
    model.to(device)
    # 循环训练
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = 0, 0
        test_loss, test_acc, test_precision, test_recall, test_f1 = 0, 0, 0, 0, 0
        # 优化训练集和测试集读取，都采用迭代器读取，原因是全数据训练集四千万+，测试集一千万+
        # 最终迭代器步进因改为训练集一百万一次，测试集二十五万一次
        # 这个迭代器对象不可重置读取位置，只能重新创建充值读取位置
        train_data_iterator = DataIterator(train_file, chunksize=1000)
        test_data_iterator = DataIterator(test_file, chunksize=250)

        # data_flag是True时全数据集，False时非全数据集
        # 非全数据集总量是data_iter * 上面设置的chunsize
        # 手动计算分块数量
        train_chunk_num = 0
        for data_chunk in train_data_iterator:
            train_loader = DataLoader(data_chunk, batch_size=BATCH_SIZE, shuffle=True)
            # 获取训练数据
            current_train_loss, current_train_acc = train_step(model=model,
                                                               dataloader=train_loader,
                                                               loss_fn=loss_fn,
                                                               optimizer=optimizer,
                                                               device=device)
            # 分块累计
            train_chunk_num += 1
            train_loss += current_train_loss
            train_acc += current_train_acc
            pass
        # 累计求平均值
        train_loss = train_loss / train_chunk_num
        train_acc = train_acc / train_chunk_num
        # 手动计算分块数量
        test_chunk_num = 0
        for data_chunk in test_data_iterator:
            test_loader = DataLoader(data_chunk, batch_size=BATCH_SIZE, shuffle=True)
            # 获取测试数据
            current_test_loss, current_test_acc, current_test_precision, current_test_recall, current_test_f1 = test_step(
                model=model,
                dataloader=test_loader,
                loss_fn=loss_fn,
                device=device)
            # 分块累计
            test_chunk_num += 1
            test_loss += current_test_loss
            test_acc += current_test_acc
            test_precision += current_test_precision
            test_recall += current_test_recall
            test_f1 += current_test_f1
            pass
        # 累计平均
        test_loss = test_loss / test_chunk_num
        test_acc = test_acc / test_chunk_num
        test_precision = test_precision / test_chunk_num
        test_recall = test_recall / test_chunk_num
        test_f1 = test_f1 / test_chunk_num
        # 每轮信息
        print(
            f"Epoch: {epoch + 1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f}"
        )
        # 数据加入数据字典
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)
        results["test_precision"].append(test_precision)
        results["test_recall"].append(test_recall)
        results["test_f1"].append(test_f1)
        pass

        # 返回最终数据
    return results
    pass
