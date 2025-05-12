import torch
import os
from torch import nn
from DGADataset import DGATrueDataset, DGAFalseDataset
from torch.utils.data import DataLoader, random_split, ConcatDataset
import sys

from utils.engine_ysx import train_ysx

sys.path.append('../model')
# 所有可用模型
from model.cnn.cnn_torch import CNNModel
from model.lstm.lstm_torch import LSTMModel
from model.mit.mit_torch import MITModel
from model.ann.ann_torch import Net
from model.bilbohybrid.bilbohybrid_torch import BilBoHybridModel
# 所有工具类函数
from utils.engine import train
from utils.saveModel import SaveModel, LoadModel
from utils.predictions import Predictions
from torch.utils.data import ConcatDataset

NUM_EPOCHS = 5
BATCH_SIZE = 32
NUM_WORKERS = os.cpu_count()
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
train_file = '../../data/train2016.csv'
test_file = '../../data/test2016.csv'


def readData():
    pass


if __name__ == '__main__':
    # input_flag = input("0训练模型, 1模型预测")
    input_flag = 0
    if int(input_flag) == 0:
        print(f"确定模型,设备为: {device}")

        # 确定训练模型
        model_ann = Net(255, 255, 255)
        model_cnn = CNNModel(255, 255, 255, 5)
        model_lstm = LSTMModel(255, 255)
        model_mit = MITModel(255, 255)
        model_bbyb = BilBoHybridModel(255, 255, 5)
        # 查看
        if torch.cuda.device_count() > 1:
            print(f"使用 {torch.cuda.device_count()} GPUs!")
            model_ann = nn.DataParallel(model_ann)
            model_cnn = nn.DataParallel(model_cnn)
            model_lstm = nn.DataParallel(model_lstm)
            model_mit = nn.DataParallel(model_mit)
            model_bbyb = nn.DataParallel(model_bbyb)


        # 二分类函数损失函数和优化器
        # 定义二元交叉熵损失函数，并使用 pos_weight 参数
        # 正样本和负样本比例总体为1：41
        pos_weight = torch.tensor([1 / 41])
        pos_weight = pos_weight.to(device)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        # 模型优化器
        ann_lr = 0.00001
        cnn_lr = 0.001
        lstm_lr = 0.001
        mit_lr = 0.0001
        bbyb_lr = 0.001
        optimizer_ann = torch.optim.SGD(params=model_ann.parameters(),
                                        lr=ann_lr,
                                        weight_decay=0.00001)
        optimizer_cnn = torch.optim.SGD(params=model_cnn.parameters(),
                                        lr=cnn_lr,
                                        weight_decay=0.001)
        optimizer_lstm = torch.optim.SGD(params=model_lstm.parameters(),
                                         lr=lstm_lr,
                                         weight_decay=0.001)
        optimizer_mit = torch.optim.SGD(params=model_mit.parameters(),
                                        lr=mit_lr,
                                        weight_decay=0.0001)
        optimizer_bbyb = torch.optim.SGD(params=model_bbyb.parameters(),
                                         lr=bbyb_lr)

        print("训练模型ANN开始")
        # 训练模型，标签为True
        print("训练模型ANN")
        train_ysx(model=model_ann,
                  train_file=train_file,
                  test_file=test_file,
                  loss_fn=loss_fn,
                  optimizer=optimizer_ann,
                  epochs=NUM_EPOCHS,
                  device=device,
                  BATCH_SIZE=BATCH_SIZE)
        # 在训练新的数据集之前，清零梯度，并将模型设置为训练状态
        optimizer_ann.zero_grad()
        model_ann.train()

        print("训练模型CNN开始")
        # 训练模型，标签为True
        print("训练模型CNN")
        train_ysx(model=model_cnn,
                  train_file=train_file,
                  test_file=test_file,
                  loss_fn=loss_fn,
                  optimizer=optimizer_cnn,
                  epochs=NUM_EPOCHS,
                  device=device,
                  BATCH_SIZE=BATCH_SIZE)
        # 在训练新的数据集之前，清零梯度，并将模型设置为训练状态
        optimizer_cnn.zero_grad()
        model_cnn.train()

        print("训练模型LSTM开始")
        # 训练模型，标签为True
        print("训练模型LSTM")
        train_ysx(model=model_lstm,
                  train_file=train_file,
                  test_file=test_file,
                  loss_fn=loss_fn,
                  optimizer=optimizer_lstm,
                  epochs=NUM_EPOCHS,
                  device=device,
                  BATCH_SIZE=BATCH_SIZE)
        # 在训练新的数据集之前，清零梯度，并将模型设置为训练状态
        optimizer_lstm.zero_grad()
        model_lstm.train()

        print("训练模型MIT开始")
        # 训练模型，标签为True
        print("训练模型MIT")
        train_ysx(model=model_mit,
                  train_file=train_file,
                  test_file=test_file,
                  loss_fn=loss_fn,
                  optimizer=optimizer_mit,
                  epochs=NUM_EPOCHS,
                  device=device,
                  BATCH_SIZE=BATCH_SIZE)
        # 在训练新的数据集之前，清零梯度，并将模型设置为训练状态
        optimizer_mit.zero_grad()
        model_mit.train()

        print("训练模型BBYB开始")
        # 训练模型，标签为True
        print("训练模型BBYB")
        train_ysx(model=model_bbyb,
                  train_file=train_file,
                  test_file=test_file,
                  loss_fn=loss_fn,
                  optimizer=optimizer_bbyb,
                  epochs=NUM_EPOCHS,
                  device=device,
                  BATCH_SIZE=BATCH_SIZE)
        # 在训练新的数据集之前，清零梯度，并将模型设置为训练状态
        optimizer_bbyb.zero_grad()
        model_bbyb.train()

        # save_flag = input("0不保存模型, 1保存模型")
        save_flag = 1
        if int(save_flag) == 1:
            print("保存模型ANN")
            SaveModel(model=model_ann,
                      target_dir="../../ModelOutputFIle/modelPth",
                      model_name="ANNModel.pth")
            print("保存模型CNN")
            SaveModel(model=model_cnn,
                      target_dir="../../ModelOutputFIle/modelPth",
                      model_name="CNNModel.pth")
            print("保存模型LSTM")
            SaveModel(model=model_lstm,
                      target_dir="../../ModelOutputFIle/modelPth",
                      model_name="LSTMModel.pth")
            print("保存模型MIT")
            SaveModel(model=model_mit,
                      target_dir="../../ModelOutputFIle/modelPth",
                      model_name="MITModel.pth")
            print("保存模型BBYB")
            SaveModel(model=model_bbyb,
                      target_dir="../../ModelOutputFIle/modelPth",
                      model_name="BBYBModel.pth")
            pass
        pass
    else:
        print("模型预测")
        print("获取数据集")
        dga_true_train_dataset = DGATrueDataset(f'../data/Benign', True)
        dga_false_train_dataset = DGAFalseDataset(f'../data/DGA/2020-06-19-dgarchive_full', True)

        # 合并正样本和负样本数据集
        combined_dataset = ConcatDataset([dga_true_train_dataset, dga_false_train_dataset])

        # 获取合并后数据集的大小
        combined_dataset_size = len(combined_dataset)
        print(f"合并后的数据集大小: {combined_dataset_size}")

        # 打乱合并后的数据集顺序
        indices = torch.randperm(combined_dataset_size)
        combined_dataset = torch.utils.data.Subset(combined_dataset, indices)

        # 划分需用于检验模型成果的训练集,这里抽出20%当作预测集
        print("划分数据集")
        train_size = int(0.8 * combined_dataset_size)
        pred_size = combined_dataset_size - train_size

        # 分割训练集和验证集
        print("划分train和predictions")
        train_dataset, pred_dataset = torch.utils.data.random_split(combined_dataset, [train_size, pred_size])

        print("创建dataLoader")
        # 创建训练集和验证集的数据加载器
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        pred_loader = DataLoader(pred_dataset, batch_size=BATCH_SIZE, shuffle=True)
        print(f"确定模型,设备为: {device}")

        # 确定模型基本结构
        base_path = '../../ModelOutputFIle/modelPth/'
        ann_name = 'ANNModel.pth'
        cnn_name = 'CNNModel.pth'
        lstm_name = 'LSTMModel.pth'
        mit_name = 'MITModel.pth'
        bbyb_name = 'BBYBModel.pth'
        model_ann = Net(255, 255, 255)
        model_cnn = CNNModel(255, 255, 255, 5)
        model_lstm = LSTMModel(255, 255)
        model_mit = MITModel(255, 255)
        model_bbyb = BilBoHybridModel(255, 255, 5)

        # 加载模型参数
        model_ann = LoadModel(model_ann, base_path, ann_name)
        model_cnn = LoadModel(model_cnn, base_path, cnn_name)
        model_lstm = LoadModel(model_lstm, base_path, lstm_name)
        model_mit = LoadModel(model_mit, base_path, mit_name)
        model_bbyb = LoadModel(model_bbyb, base_path, bbyb_name)

        # 模型预测结果
        result = Predictions(model_ann, pred_loader, device)
        print(f'ANN result: {result}')
        result = Predictions(model_cnn, pred_loader, device)
        print(f'CNN result: {result}')
        result = Predictions(model_lstm, pred_loader, device)
        print(f'LSTM result: {result}')
        result = Predictions(model_mit, pred_loader, device)
        print(f'MIT result: {result}')
        result = Predictions(model_bbyb, pred_loader, device)
        print(f'BBYB result: {result}')
        pass
    pass
