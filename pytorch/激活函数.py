import torch
import torch.nn as nn
import torch.nn.functional as F

# 数值稳定的 Softmax 实现
def stable_softmax(x):
    # 减去最大值以避免溢出
    x_max = torch.max(x, dim=-1, keepdim=True)[0]
    print(x_max)
    exp_x = torch.exp(x - x_max)
    return exp_x / torch.sum(exp_x, dim=-1, keepdim=True)

# 示例 1: 简单的 CNN 分类模型
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(32 * 8 * 8, num_classes)  # 假设输入图像为 32x32

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        print(x.shape)
        x = self.pool(F.relu(self.conv2(x)))
        print(x.shape)
        x = x.view(x.size(0), -1)  # 展平
        print(x.shape)
        logits = self.fc(x)
        print(logits.shape)
        probs = F.softmax(logits, dim=1)  # 使用 PyTorch 内置 Softmax
        print(probs.shape)
        # probs = stable_softmax(logits)  # 或使用自定义稳定版本
        return probs

# 示例 2: Transformer 分类头
class TransformerClassifier(nn.Module):
    def __init__(self, d_model=512, num_classes=2):
        super(TransformerClassifier, self).__init__()
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=8), num_layers=2
        )
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.transformer(x)
        x = x.mean(dim=1)  # 全局平均池化
        logits = self.fc(x)
        probs = F.softmax(logits, dim=1)
        return probs

# 测试代码
if __name__ == "__main__":
    # 测试 Softmax

    #创建一个张量
    logits = torch.tensor([[2.0, 1.0, 0.1], [0.5, 2.5, 1.0]], device=torch.device("cuda:0"))

    print(logits)
    probs = stable_softmax(logits)
    print("Softmax 概率1:", probs)
    probs = F.softmax(logits)
    print("Softmax 概率2:", probs)

    # 测试 CNN
    model_cnn = SimpleCNN(num_classes=2)
    input_cnn = torch.randn(1, 3, 32, 32)
    output_cnn = model_cnn(input_cnn)
    print("CNN 输出概率:", output_cnn)

    # 测试 Transformer
    model_transformer = TransformerClassifier(num_classes=2)
    input_transformer = torch.randn(1, 10, 512)  # [batch, seq_len, d_model]
    output_transformer = model_transformer(input_transformer)
    print("Transformer 输出概率:", output_transformer)


    ##Sigmoid, 输出范围: (0, 1), 用途: 二分类任务的输出层。

    x = torch.tensor([[-1.0, 0.0, 1.0], [-3.0, 1.0, 2.0]], dtype=torch.float, requires_grad = True, device=torch.device("cuda:0"))
    sigmoid = nn.Sigmoid()
    print(sigmoid(x))  # 输出：tensor([0.2689, 0.5000, 0.7311])
    print(torch.sigmoid(x))  # 函数式，等效

    '''定义: $ f(x) = \frac
    {e ^ x - e ^ {-x}}
    {e ^ x + e ^ {-x}} $
    模块: nn.Tanh
    或
    torch.tanh
    输出范围: (-1, 1)
    特点:

    零中心化，优化比
    Sigmoid
    稍好。
    仍可能导致梯度消失。


    用途: 隐藏层，RNN（如
    LSTM）的门控机制。
    '''
    tanh = nn.Tanh()
    print(tanh(x))  # 输出：tensor([-0.7616, 0.0000, 0.7616])

    relu = nn.ReLU()
    print(relu(x))  # 输出：tensor([0.0000, 0.0000, 1.0000])

    leaky_relu = nn.LeakyReLU(negative_slope=0.01)
    print(leaky_relu(x))  # 输出：tensor([-0.0100, 0.0000, 1.0000])

    prelu = nn.PReLU().to(x.device)
    print(prelu(x))  # 输出依赖于初始化的 α

    elu = nn.ELU(alpha=1.0)
    print(elu(x))  # 输出：tensor([-0.6321, 0.0000, 1.0000])

    gelu = nn.GELU()
    print(gelu(x))  # 输出：tensor([-0.1587, 0.0000, 0.8413])

    ### $ f(x) = x \cdot \text{sigmoid}(x) $

    silu = nn.SiLU()
    print(silu(x))  # 输出：tensor([-0.2689, 0.0000, 0.7311])


    '''定义: $ f(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}} $
            模块: nn.Softmax(dim) 或 F.softmax
            输出范围: (0, 1)，总和为 1
            特点:
            
            输出概率分布。
            常用于多分类任务的输出层。
            
            
            用途: 图像分类、文本分类的输出层。
    '''
    s = nn.Softmax(x)
    print(s)  # 输出：tensor([0.0900, 0.2447, 0.6652])

    x = torch.tensor([-1.0, 0.0, 1.0])
    activations = {
        'Sigmoid': nn.Sigmoid(),
        'ReLU': nn.ReLU(),
        'LeakyReLU': nn.LeakyReLU(0.01),
        'GELU': nn.GELU(),
        'Softmax': nn.Softmax(dim=0)
    }
    for name, act in activations.items():
        print(f"{name}: {act(x)}")

    import tensorflow as tf

    x = tf.constant([-1.0, 0.0, 1.0])
    activations = {
        'Sigmoid': tf.nn.sigmoid,
        'ReLU': tf.nn.relu,
        'LeakyReLU': tf.keras.layers.LeakyReLU(alpha=0.3),
        'GELU': tf.nn.gelu,
        'Softmax': tf.nn.softmax
    }
    for name, act in activations.items():
        print(f"{name}: {act(x)}")