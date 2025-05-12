import torch
from torch.utils.data import Dataset, DataLoader

# 自定义数据集类
class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        # 返回单个数据样本
        return self.data[index]

    def __len__(self):
        # 返回数据集的长度
        return len(self.data)


# 读取 CSV 文件
csv_file = "path/to/your/file.csv"
data = pd.read_csv(csv_file)

# 创建数据集实例
dataset = MyDataset(data)
# 创建数据加载器实例
batch_size = 2
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 遍历数据加载器
for batch in dataloader:
    inputs = batch  # 获取批次数据
    # 在这里执行模型的前向传播和后续操作