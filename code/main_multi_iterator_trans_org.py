import sys
import torch
from torch import nn

from model.transformer_org import TransformerOrgMultiModel

sys.path.append('model')
# 所有可用模型
from model.cnn.cnn_torch import CNNMultiModel
from model.lstm.lstm_torch import LSTMMultiModel
from model.mit.mit_torch import MITMultiModel
from model.ann.ann_torch import NetMulti
from model.bilbohybrid.bilbohybrid_torch import BBYBMultiModel
from utils.engine_multi import train_multi
from utils.saveModel import SaveModel, SaveModelPath, SaveResults, SaveResultsPath

# 多分类训练没有鲁棒性,写死False
lb_flag = False
train_file = ''
test_file = ''
# 训练设备
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# 训练参数
NUM_EPOCHS = 15
BATCH_SIZE = 32


def initParam(arg):
    """
    初始化参数
    :param arg:
    """
    global lb_flag, train_file, test_file
    flag = int(arg)
    if flag == 1:
        # 全数据集
        train_file = '../data/train2016.csv'
        test_file = '../data/test2016.csv'
        # train_file = '../data/train_partial2016.csv'
        # test_file = '../data/test_partial2016.csv'
        pass
    else:
        # 分割数据集(部分数据集)
        # train_file = '../data/extract_remain_data/2016/train.csv'
        # test_file = '../data/extract_remain_data/2016/test.csv'
        train_file = '../data/extract_remain_data/2016_1_7/train.csv'
        test_file = '../data/extract_remain_data/2016_1_7/test.csv'
        # train_file = '../data/MiniDataset/train.csv'
        # test_file = '../data/MiniDataset/test.csv'
        # train_file = '../data/train_partial2016.csv'
        # test_file = '../data/test_partial2016.csv'
        pass
    pass


def train_model(model, model_name, current_path, current_model_path, train_loss_fn, lr, optimizer):
    """
    训练模型流程
    """
    print(f"训练模型{model_name}开始")
    print(f"训练模型{model_name}, 学习率:{lr}")
    # 设置模型参数为需要计算梯度
    train_results = train_multi(model=model,
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
    # 参数1: "正常训练是否使用全数据集, 不是0, 是1"
    # 剩余可变参数为学习率+训练模型的下标,例如训练0.1的ANN和0.01的CNN就输入0.1 0 0.01 1
    # 模型下标为['ANN', 'CNN', 'LSTM', 'MIT', 'BBYB']
    print(f"参数: {sys.argv}")
    # 初始化参数
    initParam(int(sys.argv[1]))

    # 辅助信息
    print(f"确定模型,设备为: {device}, 是否是鲁棒性测试: {lb_flag}")
    print("请确认训练集是否正确,如不正确修改初始化函数")
    print(f"训练数据集文件为: {train_file}, {test_file}")

    # 载入所有模型
    # 2016年多分类为64个恶意+一个良性总共65个
    model_ann = NetMulti(255, 255, 255, num_classes=65)
    model_cnn = CNNMultiModel(255, 255, 255, 5, num_classes=65)
    model_lstm = LSTMMultiModel(255, 255, num_classes=65)
    model_mit = MITMultiModel(255, 255, num_classes=65)
    model_bbyb = BBYBMultiModel(255, 255, 5, num_classes=65)
    model_transformer = TransformerOrgMultiModel(input_dim=255,
                                                 output_dim=65,
                                                 d_model=128,
                                                 nhead=4,
                                                 num_layers=1,
                                                 dim_feedforward=256,
                                                 dropout=0.1)

    # 生成列表
    model_name_list = ['ANN_Multi', 'CNN_Multi', 'LSTM_Multi', 'MIT_Multi', 'BBYB_Multi', 'Transformer_Multi']
    model_list = [model_ann, model_cnn, model_lstm, model_mit, model_bbyb, model_transformer]

    # 多分类损失函数
    # 多分类没有pos_weight
    # loss_fn = nn.NLLLoss()
    loss_fn = nn.CrossEntropyLoss()
    # class_weights = multi_label(train_file)
    # class_weights = class_weights.to(device)
    # loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    # 获取本次训练的名称
    current_name = str(sys.argv[2])
    # 获取本次训练记录存放地址
    current_path = SaveResultsPath(current_name, lb_flag, False)
    # 每个训练模型结果存放地址
    current_model_path = SaveModelPath("modelMultiPth", current_name, lb_flag, False)
    print(current_name, current_path, current_model_path)

    # 抽取所有的可变参数,生成模型
    pair_list = sys.argv[3:]
    model_pair_list = [{'index': pair_list[i], 'lr': pair_list[i + 1]} for i in range(0, len(pair_list), 2)]
    # 调用可变参数的模型
    for pair in model_pair_list:
        # 当前模型
        current_model = model_list[int(pair['index'])]
        current_model_name = model_name_list[int(pair['index'])]
        # 当前学习率
        current_lr = float(pair['lr'])
        # 优化器
        pair_optimizer = torch.optim.SGD(params=current_model.parameters(), lr=current_lr)
        # pair_optimizer = torch.optim.Adam(params=current_model.parameters(), lr=current_lr)
        train_model(current_model, current_model_name, current_path, current_model_path, loss_fn, current_lr,
                    pair_optimizer)
        pass
    pass
