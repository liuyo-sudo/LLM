
import torch
import torch.nn as nn

# 定义 Conv2d 层
conv = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)

# 输入张量：(batch_size=1, channels=3, height=32, width=32)
input_tensor = torch.randn(1, 3, 32, 32)

# 前向传播
output = conv(input_tensor)
print(output)
print(output.dim())
print(output.shape)  # 输出：torch.Size([1, 16, 32, 32])


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(32 * 8 * 8, num_classes)

    def forward(self, x):
        x = self.pool(nn.ReLU(self.conv1(x)))
        x = self.pool(nn.ReLU(self.conv2(x)))
        x = x.view(x.size(0), -1)  # 展平
        logits = self.fc(x)
        probs = nn.Softmax(logits, dim=1)
        return probs