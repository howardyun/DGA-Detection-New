import sys
import torch
import os
# 一些路径检查
current_dir = os.path.dirname(os.path.abspath(__file__))
# 添加utils包下所有py路径
sys.path.append(current_dir.replace("Predict", "utils"))
# 添加code包下所有py路径
sys.path.append(current_dir.replace("\\Predict", ""))
from utils.predictions_multi import PredictionMulti, SaveMultiFilePath
from utils.saveModel import LoadModel
from model.cnn.cnn_torch import CNNMultiModel
from model.lstm.lstm_torch import LSTMMultiModel
from model.mit.mit_torch import MITMultiModel
from model.ann.ann_torch import NetMulti
from model.bilbohybrid.bilbohybrid_torch import BBYBMultiModel


# 设备
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# 预测
predict_file = ''
predict_full_data_flag = False
# 本文件正常预测,lb标识为False
lb_flag = False
# 批次参数
BATCH_SIZE = 32


def initPredictParam(args):
    """
    初始化预测流程参数
    """
    print("初始化选项")
    # 初始化全局变量
    global predict_file, predict_full_data_flag
    # 正常预测,随机数据集预测0,分割数据集预测1
    print("正常预测")
    flag = int(args[0])
    if int(flag) == 1:
        # 分割数据集预测,使用部分数据0, 全部数据1
        print("分割数据集预测")
        flag = int(args[1])
        if int(flag) == 1:
            print("使用全部数据")
            predict_full_data_flag = True
            predict_file = '../data/extract_remain_data/2016/predict.csv'
            # predict_file = '../data/train_partial2016.csv'
            pass
        else:
            print("使用部分数据")
            predict_full_data_flag = False
            predict_file = '../data/extract_remain_data/2016/predict.csv'
            # predict_file = '../data/train_partial2016.csv'
            pass
        pass
    else:
        # 随机数据集预测,使用部分数据0, 全部数据1
        print("随机数据集预测")
        flag = int(args[1])
        if int(flag) == 1:
            print("使用全部数据")
            predict_full_data_flag = True
            predict_file = '../../data/test2016.csv'
            pass
        else:
            print("使用部分数据")
            predict_full_data_flag = False
            predict_file = '../../data/test2016.csv'
            pass
        pass
    pass


if __name__ == '__main__':
    # 初始化
    # 参数设置
    print("正在进行多分类模型预测, 正常预测")
    print(f"参数: {sys.argv}")
    # 参数初始化
    initPredictParam(sys.argv[1:3])
    print(f"确定模型,设备为: {device}")
    print("请确认预测集是否正确,如不正确修改初始化函数")

    # 所有模型参数
    # 确定模型基本结构
    model_ann = NetMulti(255, 255, 255, num_classes=65)
    model_cnn = CNNMultiModel(255, 255, 255, 5, num_classes=65)
    model_lstm = LSTMMultiModel(255, 255, num_classes=65)
    model_mit = MITMultiModel(255, 255, num_classes=65)
    model_bbyb = BBYBMultiModel(255, 255, 5, num_classes=65)
    # 模型列表
    model_name_list = ['ANN', 'CNN', 'LSTM', 'MIT', 'BBYB']
    model_list = [model_ann, model_cnn, model_lstm, model_mit, model_bbyb]

    # 加载模型参数
    # 本次预测名称
    current_name = str(sys.argv[3])
    # 模型存放根文件夹
    base_path = str(sys.argv[4])
    # 模型在各自根文件夹下的名字
    pair_list = sys.argv[5:]
    model_pair_list = [{'index': pair_list[i], 'path': pair_list[i + 1]} for i in range(0, len(pair_list), 2)]

    # 训练调用
    print(f"获取数据集: {predict_file}")
    print("全数据集") if predict_full_data_flag else print("部分数据集")
    # 本次批次存放文件的共同路径
    results_file_dir, acc_pre_f1_file_path = SaveMultiFilePath(current_name=current_name, lb_flag=lb_flag)
    for pair in model_pair_list:
        print(f"使用模型{model_name_list[int(pair['index'])]}")
        # 本次循环使用的模型
        load_model = LoadModel(model_list[int(pair['index'])], base_path, pair['path'])
        # 预测
        PredictionMulti(model=load_model, model_name=str(pair['path']), file=predict_file,
                        results_file_dir=results_file_dir, acc_pre_f1_file_path=acc_pre_f1_file_path,
                        device=device,
                        full_flag=predict_full_data_flag,
                        BATCH_SIZE=BATCH_SIZE)
        pass
    pass
