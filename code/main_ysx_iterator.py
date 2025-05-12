import torch
import os
from torch import nn
import sys

from model.transformer_org import TransformerModel
from utils.engine_ysx import train_ysx

sys.path.append('model')
# 所有可用模型
from model.cnn.cnn_torch import CNNModel
from model.lstm.lstm_torch import LSTMModel
from model.mit.mit_torch import MITModel
from model.ann.ann_torch import Net
from model.bilbohybrid.bilbohybrid_torch import BilBoHybridModel
# 所有工具类函数
from utils.saveModel import SaveModel, SaveModelPath, SaveResults, SaveResultsPath

# 训练模型参数
# 按照数据集正负样本比例变化改变
pos_weight_num = 0.0202
NUM_EPOCHS = 15
BATCH_SIZE = 32
NUM_WORKERS = os.cpu_count()
# 训练设备
device = torch.device("cuda:1") if torch.cuda.is_available() else torch.device("cpu")

# 训练初始化参数
train_file = ''
test_file = ''
predict_file = ''
predict_full_data_flag = False
lb_flag = False

# 预测初始化参数
family_flag = False
family_full_data_flag = False
family_predict_file = ''

# 预测模型路径
base_path = './modelPth/dga/2024031310'
ann_name = '0.001ANNModel.pth'
cnn_name = '0.001CNNModel.pth'
lstm_name = '0.001LSTMModel.pth'
mit_name = '0.001MITModel.pth'
bbyb_name = '0.001BBYBModel.pth'
transformer_name = '0.001TransformerModel.pth'


def readData():
    pass


def initParam(arg, p1, p2):
    """
    执行初始化流程
    """
    print(f"初始化选项")
    # 初始化全局变量
    global lb_flag, train_file, test_file
    if arg:
        init_flag = p1
    else:
        init_flag = input("是否为鲁棒性测试, 不是0, 是1")
    lb_flag = True if int(init_flag) == 1 else False
    if lb_flag:
        # 鲁棒性测试
        if arg:
            flag = p2
        else:
            flag = input("正常训练是否使用全数据集, 不是0, 是1")
        if int(flag) == 1:
            # 全数据集
            train_file = '../data/lb_full_data/lb_train2016.csv'
            test_file = '../data/lb_full_data/lb_test2016.csv'
            # train_file = '../data/train_partial2016.csv'
            # test_file = '../data/test_partial2016.csv'
            pass
        else:
            train_file = '../data/lb_partial_data/lb_train2016.csv'
            test_file = '../data/lb_partial_data/lb_test2016.csv'
            # train_file = '../data/train_partial2016.csv'
            # test_file = '../data/test_partial2016.csv'
            pass
        pass
    else:
        # 非鲁棒性测试,正常训练
        if arg:
            flag = p2
        else:
            flag = input("正常训练是否使用全数据集, 不是0, 是1")
        if int(flag) == 1:
            train_file = '../data/train2016.csv'
            test_file = '../data/test2016.csv'
            # train_file = '../data/train_partial2016.csv'
            # test_file = '../data/test_partial2016.csv'
            pass
        else:
            # train_file = '../data/extract_remain_data/2016/train.csv'
            # test_file = '../data/extract_remain_data/2016/test.csv'
            # train_file = '../data/extract_remain_data/2016/train.csv'
            # test_file = '../data/extract_remain_data/2016/test.csv'
            train_file = '../data/extract_remain_data/2016_1_7/train.csv'
            test_file = '../data/extract_remain_data/2016_1_7/test.csv'
            # train_file = '../data/train_partial2016.csv'
            # test_file = '../data/test_partial2016.csv'
            # train_file = '../data/MiniDataset/train.csv'
            # test_file = '../data/MiniDataset/test.csv'
            pass
        pass
    pass


def train_model(model, model_name, current_path, current_model_path, train_loss_fn, lr, optimizer):
    """
    训练模型流程
    """
    print(f"训练模型{model_name}开始")
    print(f"训练模型{model_name}, 学习率:{lr}")

    # 训练
    train_results = train_ysx(model=model,
                              train_file=train_file,
                              test_file=test_file,
                              loss_fn=train_loss_fn,
                              optimizer=optimizer,
                              epochs=NUM_EPOCHS,
                              device=device,
                              BATCH_SIZE=BATCH_SIZE)
    # 保存训练结果
    SaveResults(str(lr) + f"{model_name}Model", NUM_EPOCHS, train_results, current_path)
    # 在训练新的数据集之前，清零梯度，并将模型设置为训练状态
    optimizer.zero_grad()
    model.train()

    # 保存模型
    print(f"保存模型{model_name}")
    # 存放的model名字
    save_model_name = f"{lr}{model_name}.pth"
    SaveModel(model=model,
              model_name=save_model_name,
              model_save_path=current_model_path)
    pass


if __name__ == '__main__':
    # 参数1："0训练模型, 1模型预测"
    # 参数2: "是否为鲁棒性测试, 不是0, 是1"
    # 参数3: "正常训练是否使用全数据集, 不是0, 是1"
    # 参数4: "pos_weight"
    # 剩余可变参数为学习率+训练模型的下标,例如训练0.1的ANN和0.01的CNN就输入0.1 0 0.01 1
    # 模型下标为['ANN', 'CNN', 'LSTM', 'MIT', 'BBYB','Transformer']
    arg = False
    print(sys.argv)
    if len(sys.argv) > 1:
        arg = True
        print('按照参数设置配置')
        pass
    else:
        print('没有传参，按照手工进行设置')
        pass
    # 手动设置参数
    if not arg:
        input_flag = input("0训练模型, 1模型预测")
        pass
    else:
        input_flag = sys.argv[1]
        pass
    if int(input_flag) == 0:
        if not arg:
            initParam(arg, 0, 0)
            pass
        else:
            initParam(arg, sys.argv[2], sys.argv[3])
            pass
        pass

    # 辅助信息
    print(f"确定模型,设备为: {device}, 是否是鲁棒性测试: {lb_flag}")
    print("请确认训练集是否正确,如不正确修改初始化函数")
    print(f"训练数据集文件为: {train_file}, {test_file}")

    # 载入所有模型
    model_ann = Net(255, 255, 255)
    model_cnn = CNNModel(255, 255, 255, 5)
    model_lstm = LSTMModel(255, 255)
    model_mit = MITModel(255, 255)
    model_bbyb = BilBoHybridModel(255, 255, 5)
    model_transfomer = TransformerModel(input_dim=255,
                                        output_dim=1,
                                        d_model=128,
                                        nhead=8,
                                        num_layers=3,
                                        dim_feedforward=256,
                                        dropout=0.1)

    # 生成列表
    model_name_list = ['ANN', 'CNN', 'LSTM', 'MIT', 'BBYB', 'Transformer']
    model_list = [model_ann, model_cnn, model_lstm, model_mit, model_bbyb, model_transfomer]

    # 获取参数输入的pos_weight
    pos_weight_num = float(sys.argv[5])
    print(f"pos weight: {pos_weight_num}")
    # 二分类函数损失函数和优化器
    # 定义二元交叉熵损失函数，并使用 pos_weight 参数
    # 正样本和负样本比例要按照数据集变化改变
    pos_weight = torch.tensor([pos_weight_num])
    pos_weight = pos_weight.to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # 抽取所有的可变参数,生成模型
    pair_list = sys.argv[6:]
    model_pair_list = [{'index': pair_list[i], 'lr': pair_list[i + 1]} for i in range(0, len(pair_list), 2)]

    # 获取本次训练的名称
    current_name = str(sys.argv[4])
    # 获取本次训练记录存放地址
    current_path = SaveResultsPath(current_name, lb_flag, True)
    # 每个训练模型结果存放地址
    current_model_path = SaveModelPath("modelPth", current_name, lb_flag, True)
    print(current_name, current_path, current_model_path)

    # 调用可变参数的模型
    for pair in model_pair_list:
        # 当前模型
        current_model = model_list[int(pair['index'])]
        current_model_name = model_name_list[int(pair['index'])]
        # 当前学习率
        current_lr = float(pair['lr'])
        # 优化器
        # pair_optimizer = torch.optim.SGD(params=current_model.parameters(), lr=current_lr)
        pair_optimizer = torch.optim.AdamW(params=model_transfomer.parameters(),
                                           lr=current_lr)
        # 训练模型函数
        train_model(current_model, current_model_name, current_path, current_model_path, loss_fn, current_lr,
                    pair_optimizer)
        pass
    pass
