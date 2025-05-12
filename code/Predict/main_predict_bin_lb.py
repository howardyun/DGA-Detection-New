import sys
import os
import torch
# 一些路径检查
current_dir = os.path.dirname(os.path.abspath(__file__))
# 添加utils包下所有py路径
sys.path.append(current_dir.replace("Predict", "utils"))
# 添加code包下所有py路径
sys.path.append(current_dir.replace("\\Predict", ""))
from model.cnn.cnn_torch import CNNModel
from model.lstm.lstm_torch import LSTMModel
from model.mit.mit_torch import MITModel
from model.ann.ann_torch import Net
from model.bilbohybrid.bilbohybrid_torch import BilBoHybridModel
from utils.predictions import Predictions, SaveFilePath, PredictionFamily
from utils.saveModel import LoadModel
from model.transformer_improve import Trans_DGA

# 鲁棒性预测,就是True
lb_flag = True
# 非家族预测
family_flag = False
family_full_data_flag = False
family_predict_file = ''
predict_file = ''
predict_full_data_flag = False

# 设备
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# 批次参数
BATCH_SIZE = 32


def initPredictParam(args):
    """
    初始化预测流程参数
    :return:
    """
    global lb_flag, family_flag, family_full_data_flag, family_predict_file, predict_file, predict_full_data_flag
    # 鲁棒性
    # 鲁棒性预测
    print("鲁棒性预测")
    flag = int(args[0])
    if int(flag) == 1:
        # 鲁棒-家族预测
        print("鲁棒-家族预测")
        family_flag = True
        flag = int(args[1])
        if int(flag) == 1:
            # 全数据集
            print("全数据集")
            family_full_data_flag = True
            family_predict_file = '../data/lb_full_data/lb_type2016.csv'
            pass
        else:
            # 部分数据集
            print("部分数据集")
            family_full_data_flag = False
            family_predict_file = '../data/lb_partial_data/lb_type2016.csv'
            pass
        pass
    else:
        # 鲁棒-正常预测
        print("鲁棒-正常预测")
        family_flag = False
        flag = int(args[1])
        if int(flag) == 1:
            # 全数据集
            print("全数据集")
            # 两个都是True是因为训练集生产时已经时全数据集和部分数据集,不需要再用flag分割了,都全部读入即可
            predict_full_data_flag = True
            # predict_file = '../data/lb_full_data/lb_predict2016.csv'
            predict_file = '../data/lb_predict_data/lb_predict_2016_10000000.csv'
            # predict_file = '../data/train_partial2016.csv'
            pass
        else:
            # 部分数据集
            print("部分数据集")
            predict_full_data_flag = True
            # predict_file = '../data/lb_full_data/lb_predict2016.csv'
            predict_file = '../data/lb_predict_data/lb_predict_2016_10000000.csv'
            # predict_file = '../data/train_partial2016.csv'
            pass
        pass
    pass


if __name__ == '__main__':
    # 初始化
    # 参数设置
    print("正在进行二分类模型预测, 正常预测,非鲁棒性预测")
    print(f"参数: {sys.argv}")
    # 参数初始化
    initPredictParam(sys.argv[1:3])
    print(f"确定模型,设备为: {device}")
    print("请确认预测集是否正确,如不正确修改初始化函数")

    # 所有模型参数
    # 确定模型基本结构
    model_ann = Net(255, 255, 255)
    model_cnn = CNNModel(255, 255, 255, 5)
    model_lstm = LSTMModel(255, 255)
    model_mit = MITModel(255, 255)
    model_bbyb = BilBoHybridModel(255, 255, 5)
    model_transfomer = Trans_DGA(num_classes=1, vocab_size=40)
    # 模型列表
    model_name_list = ['ANN', 'CNN', 'LSTM', 'MIT', 'BBYB', 'Transformer']
    model_list = [model_ann, model_cnn, model_lstm, model_mit, model_bbyb, model_transfomer]
    # model_list = [model_ann, model_cnn, model_lstm, model_mit, model_bbyb]

    # 获取本次训练的名称
    # 本次预测名称
    current_name = str(sys.argv[3])
    # 加载模型参数
    # 模型存放根文件夹
    base_path = str(sys.argv[4])
    # 模型在各自根文件夹下的名字
    pair_list = sys.argv[5:]
    # 配置成对象
    model_pair_list = [{'index': pair_list[i], 'path': pair_list[i + 1]} for i in range(0, len(pair_list), 2)]

    # 鲁棒性
    print("鲁棒性预测")
    if family_flag:
        # 鲁棒家族
        print(f"获取数据集: {family_predict_file}")
        print("全数据集") if family_full_data_flag else print("部分数据集")
        # 本次循环保存文件路径
        results_file_dir, acc_pre_f1_file_path = SaveFilePath(current_name=current_name, lb_flag=lb_flag)
        for pair in model_pair_list:
            print(f"使用模型{model_name_list[int(pair['index'])]}")
            # 本次循环加载的模型
            load_model = LoadModel(model_list[int(pair['index'])], base_path, pair['path'])
            PredictionFamily(model=load_model, model_name=str(pair['path']), file=family_predict_file,
                             results_file_dir=results_file_dir, acc_pre_f1_file_path=acc_pre_f1_file_path,
                             device=device,
                             full_flag=family_full_data_flag,
                             BATCH_SIZE=BATCH_SIZE)
            pass
        pass
    else:
        # 鲁棒正常
        print(f"获取数据集: {predict_file}")
        print("全数据集") if predict_full_data_flag else print("部分数据集")
        # 本次循环保存文件路径
        results_file_dir, acc_pre_f1_file_path = SaveFilePath(current_name=current_name, lb_flag=lb_flag)
        for pair in model_pair_list:
            print(f"使用模型{model_name_list[int(pair['index'])]}")
            # 本次循环加载的模型
            load_model = LoadModel(model_list[int(pair['index'])], base_path, pair['path'])
            Predictions(model=load_model, model_name=str(pair['path']), file=predict_file,
                        results_file_dir=results_file_dir, acc_pre_f1_file_path=acc_pre_f1_file_path,
                        device=device,
                        full_flag=predict_full_data_flag,
                        BATCH_SIZE=BATCH_SIZE,
                        lb_flag=lb_flag)
            pass
        pass
    pass
    pass
