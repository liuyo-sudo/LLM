import torch
import torch.nn as nn
import torch.optim as optim

# 数据
X = torch.tensor([[1.0, 2.0], [2.0, 1.0], [-1.0, -1.0], [0.0, 1.0]])
y = torch.tensor([0, 1, 0, 1])
device = torch.device('cuda')  # 使用CPU

# 模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc = nn.Linear(2, 2)
    def forward(self, x):
        return self.fc(x)

model = Net().to(device)
model.fc.weight.data = torch.tensor([[0.1, 0.2], [0.3, 0.4]])  # 固定初始权重
model.fc.bias.data = torch.tensor([0.0, 0.0])  # 固定初始偏置

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

output = model(X)  # 前向传播
loss = criterion(output, y)  # 计算交叉熵损失
print(f"Loss: {loss.item()}")
loss.backward()  # 反向传播