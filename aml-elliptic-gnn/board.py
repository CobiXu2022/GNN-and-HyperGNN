import torch
from torch.utils.tensorboard import SummaryWriter
from torch.nn import Module
from torch_geometric.nn import GCNConv
import torch.nn.functional as F

class GCNConvolution(Module):
    def __init__(self, args, num_features, hidden_units):
        super().__init__()
        self.conv1 = GCNConv(num_features, hidden_units)
        self.conv2 = GCNConv(hidden_units, args['num_classes'])

    def forward(self, x, edge_index):
        #x, edge_index = data
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return x, edge_index

# 创建模型实例
args = {'num_classes': 10}  # 示例参数
model = GCNConvolution(args, num_features=32, hidden_units=16)

# 创建TensorBoard的SummaryWriter
writer = SummaryWriter('runs/gcn_experiment')

# 创建虚拟输入数据
dummy_data = (torch.rand(100, 32), torch.randint(0, 100, (2, 200)))  # (x, edge_index)

# 将模型结构写入TensorBoard
writer.add_graph(model, dummy_data)
writer.close()

print("模型结构已写入TensorBoard，使用命令：tensorboard --logdir=runs")