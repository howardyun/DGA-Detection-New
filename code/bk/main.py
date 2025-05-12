import torch
import os
from torch import nn
from code.DGADataset import DGATrueDataset, DGAFalseDataset
from torch.utils.data import DataLoader, random_split, ConcatDataset
# 所有可用模型
from model.cnn.cnn_torch import CNNModel
from model.lstm.lstm_torch import LSTMModel
from model.mit.mit_torch import MITModel
from model.ann.ann_torch import Net
from model.bilbohybrid.bilbohybrid_torch import BilBoHybridModel
# 所有工具类函数
from utils.engine import train
from utils.saveModel import SaveModel

NUM_EPOCHS = 5
BATCH_SIZE = 32
NUM_WORKERS = os.cpu_count()
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def readData():
    pass


if __name__ == '__main__':
    # 返回训练集逻辑dataset
    print("获取数据集")
    dga_true_train_dataset = DGATrueDataset(f'../../data/Benign', True)
    dga_false_train_dataset = DGAFalseDataset(f'../../data/DGA/2016-09-19-dgarchive_full', True)

    # """
    dga_true_train_dataset_size = len(dga_true_train_dataset)
    dga_false_train_dataset_size = len(dga_false_train_dataset)
    print(f"标签true数据集大小: {dga_true_train_dataset_size}")
    print(f"标签false数据集大小: {dga_false_train_dataset_size}")

    # 划分数据集为80%训练集，20%验证集
    print("划分数据集")
    true_train_size = int(0.8 * dga_true_train_dataset_size)
    true_test_size = dga_true_train_dataset_size - true_train_size
    true_train_dataset, true_test_dataset = random_split(dga_true_train_dataset,
                                                         [true_train_size, true_test_size])
    false_train_size = int(0.8 * dga_false_train_dataset_size)
    false_test_size = dga_false_train_dataset_size - false_train_size
    false_train_dataset, false_test_dataset = random_split(dga_false_train_dataset,
                                                           [false_train_size, false_test_size])

    print("创建dataLoader")
    # 创建标签为false数据加载成的训练集
    false_train_loader = DataLoader(false_train_dataset, batch_size=BATCH_SIZE)
    false_test_loader = DataLoader(false_test_dataset, batch_size=BATCH_SIZE)
    # 创建标签为true数据加载成的训练集
    true_train_loader = DataLoader(true_train_dataset, batch_size=BATCH_SIZE)
    true_test_loader = DataLoader(true_test_dataset, batch_size=BATCH_SIZE)

    print(f"确定模型,设备为: {device}")
    # 确定训练模型
    model_ann = Net(255, 255, 255)
    model_cnn = CNNModel(255, 255, 255, 5)
    model_lstm = LSTMModel(255, 255)
    model_mit = MITModel(255, 255)
    model_bbyb = BilBoHybridModel(255, 255, 5)
    # 二分类函数损失函数和优化器
    loss_fn = nn.BCEWithLogitsLoss()
    # 模型优化器
    optimizer_ann = torch.optim.SGD(params=model_ann.parameters(),
                                    lr=0.001)
    optimizer_cnn = torch.optim.SGD(params=model_cnn.parameters(),
                                    lr=0.01)
    optimizer_lstm = torch.optim.SGD(params=model_lstm.parameters(),
                                     lr=0.01)
    optimizer_mit = torch.optim.SGD(params=model_mit.parameters(),
                                    lr=0.01)
    optimizer_bbyb = torch.optim.SGD(params=model_bbyb.parameters(),
                                     lr=0.01)

    # """
    print("训练模型ANN开始")
    # 训练模型，标签为True
    print("训练模型ANN，标签为True")
    train(model=model_ann,
          train_dataloader=true_train_loader,
          test_dataloader=true_test_loader,
          loss_fn=loss_fn,
          optimizer=optimizer_ann,
          epochs=NUM_EPOCHS,
          device=device)
    # 在训练新的数据集之前，清零梯度，并将模型设置为训练状态
    optimizer_ann.zero_grad()
    model_ann.train()
    # 训练模型，标签为False
    print("训练模型ANN，标签为False")
    train(model=model_ann,
          train_dataloader=false_train_loader,
          test_dataloader=false_test_loader,
          loss_fn=loss_fn,
          optimizer=optimizer_ann,
          epochs=NUM_EPOCHS,
          device=device)
    print("训练模型ANN结束")
    # """

    # """
    print("训练模型CNN开始")
    # 训练模型，标签为True
    print("训练模型CNN，标签为True")
    train(model=model_cnn,
          train_dataloader=true_train_loader,
          test_dataloader=true_test_loader,
          loss_fn=loss_fn,
          optimizer=optimizer_cnn,
          epochs=NUM_EPOCHS,
          device=device)
    # 在训练新的数据集之前，清零梯度，并将模型设置为训练状态
    optimizer_cnn.zero_grad()
    model_cnn.train()
    # 训练模型，标签为False
    print("训练模型CNN，标签为False")
    train(model=model_cnn,
          train_dataloader=false_train_loader,
          test_dataloader=false_test_loader,
          loss_fn=loss_fn,
          optimizer=optimizer_cnn,
          epochs=NUM_EPOCHS,
          device=device)
    print("训练模型CNN结束")
    # """

    # """
    print("训练模型LSTM开始")
    # 训练模型，标签为True
    print("训练模型LSTM，标签为True")
    train(model=model_lstm,
          train_dataloader=true_train_loader,
          test_dataloader=true_test_loader,
          loss_fn=loss_fn,
          optimizer=optimizer_lstm,
          epochs=NUM_EPOCHS,
          device=device)
    # 在训练新的数据集之前，清零梯度，并将模型设置为训练状态
    optimizer_lstm.zero_grad()
    model_lstm.train()
    # 训练模型，标签为False
    print("训练模型LSTM，标签为False")
    train(model=model_lstm,
          train_dataloader=false_train_loader,
          test_dataloader=false_test_loader,
          loss_fn=loss_fn,
          optimizer=optimizer_lstm,
          epochs=NUM_EPOCHS,
          device=device)
    print("训练模型LSTM结束")
    # """

    # """
    print("训练模型MIT开始")
    # 训练模型，标签为True
    print("训练模型MIT，标签为True")
    train(model=model_mit,
          train_dataloader=true_train_loader,
          test_dataloader=true_test_loader,
          loss_fn=loss_fn,
          optimizer=optimizer_mit,
          epochs=NUM_EPOCHS,
          device=device)
    # 在训练新的数据集之前，清零梯度，并将模型设置为训练状态
    optimizer_mit.zero_grad()
    model_mit.train()
    # 训练模型，标签为False
    print("训练模型MIT，标签为False")
    train(model=model_mit,
          train_dataloader=false_train_loader,
          test_dataloader=false_test_loader,
          loss_fn=loss_fn,
          optimizer=optimizer_mit,
          epochs=NUM_EPOCHS,
          device=device)
    print("训练模型MIT结束")
    # """

    # """
    print("训练模型BBYB开始")
    # 训练模型，标签为True
    print("训练模型BBYB，标签为True")
    train(model=model_bbyb,
          train_dataloader=true_train_loader,
          test_dataloader=true_test_loader,
          loss_fn=loss_fn,
          optimizer=optimizer_bbyb,
          epochs=NUM_EPOCHS,
          device=device)
    # 在训练新的数据集之前，清零梯度，并将模型设置为训练状态
    optimizer_bbyb.zero_grad()
    model_bbyb.train()
    # 训练模型，标签为False
    print("训练模型BBYB，标签为False")
    train(model=model_bbyb,
          train_dataloader=false_train_loader,
          test_dataloader=false_test_loader,
          loss_fn=loss_fn,
          optimizer=optimizer_bbyb,
          epochs=NUM_EPOCHS,
          device=device)
    print("训练模型BBYB结束")
    # """

    # print("保存模型")
    # SaveModel(model=model,
    #           target_dir="modelPth",
    #           model_name="CNNModel.pth")

    pass
