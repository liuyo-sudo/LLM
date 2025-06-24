# 导入 PyTorch 核心库
import torch
# 导入神经网络模块
import torch.nn as nn
# 导入 Adam 优化器
import torch.optim as optim
# 导入数据集和数据加载器
from torch.utils.data import Dataset, DataLoader
# 导入 pandas 库，用于读取 CSV 数据
import pandas as pd
# 导入 Matplotlib 用于可视化
import matplotlib.pyplot as plt
# 导入模型文件
from model import Transformer
# 导入分词器文件
from tokenizer import QACSVCharTokenizer
# 导入 os 模块
import os
# 导入 NumPy 用于数值计算
import numpy as np
# 导入学习率调度器，动态调整学习率
from torch.optim.lr_scheduler import ReduceLROnPlateau

# 定义问答数据集类，继承自 Dataset
class QADataset(Dataset):
    # 初始化数据集
    # 参数：csv_file（数据集文件路径），tokenizer（分词器对象），max_len（最大序列长度，默认512）
    def __init__(self, csv_file, tokenizer, max_len=512):
        # 读取 CSV 文件，加载问答数据
        self.data = pd.read_csv(csv_file)
        # 保存分词器对象
        self.tokenizer = tokenizer
        # 保存最大序列长度
        self.max_len = max_len

    # 返回数据集大小
    def __len__(self):
        # 返回数据框的行数
        return len(self.data)

    # 获取单条数据
    def __getitem__(self, idx):
        # 获取第 idx 行的问题
        question = self.data.iloc[idx]["question"]
        # 获取第 idx 行的答案
        answer = self.data.iloc[idx]["answer"]
        # 编码问题，截断到 max_len
        src = self.tokenizer.encode(question)[:self.max_len]
        # 编码答案，截断到 max_len
        tgt = self.tokenizer.encode(answer)[:self.max_len]
        # 填充源序列到 max_len，填充值为 0（[PAD]）
        src = src + [0] * (self.max_len - len(src))
        # 填充目标序列到 max_len
        tgt = tgt + [0] * (self.max_len - len(tgt))
        # 返回源序列和目标序列的张量
        return torch.tensor(src), torch.tensor(tgt)

# 定义创建掩码的函数，用于屏蔽填充 token 和未来 token
def create_masks(src, tgt, pad_idx=0):
    # 创建源序列掩码，标记非填充 token（True）
    # 形状：(batch_size, 1, 1, seq_len)
    src_mask = (src != pad_idx).unsqueeze(1).unsqueeze(2)
    print(src_mask)
    # 创建目标序列掩码，标记非填充 token
    # 形状：(batch_size, 1, seq_len, 1)
    tgt_mask = (tgt != pad_idx).unsqueeze(1).unsqueeze(3)
    # 获取目标序列长度
    seq_len = tgt.size(1)
    # 创建未来信息掩码（上三角矩阵，1 表示屏蔽）
    # 原理：防止解码器关注未来 token
    nopeak_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(tgt.device)
    # 结合填充掩码和未来信息掩码
    tgt_mask = tgt_mask & (~nopeak_mask)
    # 返回源序列掩码和目标序列掩码
    return src_mask, tgt_mask

# 定义训练函数，优化为全精度并支持梯度累积
def train_model(model, train_loader, val_loader, num_epochs=100, device="cuda", accum_steps=1):
    # 初始化 Adam 优化器
    # 参数：学习率 0.0001，betas=(0.9, 0.98)，eps=1e-9（稳定梯度）
    optimizer = optim.Adam(model.parameters(), lr=0.00001, betas=(0.9, 0.98), eps=1e-9)
    # 初始化学习率调度器，当验证损失 5 轮不下降时降低学习率
    # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
    # 定义交叉熵损失函数，忽略填充 token（pad_idx=0）
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    # 初始化训练损失、验证损失和梯度范数列表
    train_losses = []
    val_losses = []
    grad_norms = []
    # # 开启 Matplotlib 交互模式，用于动态可视化
    # plt.ion()
    # # 创建单一图形对象，包含两个子图
    # # 原理：初始化一次图形，确保所有 epoch 在同一张图上绘制
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    # # 设置图形布局，优化子图间距
    # plt.tight_layout()

    # 循环 num_epochs 次
    for epoch in range(num_epochs):
        # 设置模型为训练模式
        model.train()
        # 初始化轮次损失和梯度范数
        epoch_loss = 0
        grad_norm = 0
        # 初始化批次计数，用于梯度累积
        batch_count = 0
        # 清空优化器梯度
        optimizer.zero_grad()
        # 遍历训练数据
        for src, tgt in train_loader:
            # 将数据移动到设备（GPU/CPU）
            src, tgt = src.to(device), tgt.to(device)
            # 创建掩码
            src_mask, tgt_mask = create_masks(src, tgt)
            # 前向传播，全精度计算
            # 原理：移除 autocast，直接使用 FP32 确保数值稳定性
            output, _, _, _ = model(src, tgt[:, :-1], src_mask, tgt_mask[:, :, :-1, :-1])
            # 计算损失
            # 公式：Loss = -sum(y_i * log(hat{y}_i))
            loss = criterion(output.reshape(-1, output.size(-1)), tgt[:, 1:].reshape(-1))
            # 梯度累积：将损失除以累积步数
            # 原理：累积多次小批量梯度，等效于大批量更新，节省内存
            loss = loss / accum_steps
            # 反向传播，累积梯度
            loss.backward()
            # 增加批次计数
            batch_count += 1
            # 当达到累积步数时更新参数
            if batch_count % accum_steps == 0:
                # 计算并裁剪梯度范数（最大范数 1.0）
                # 原理：防止梯度爆炸
                grad_norm += torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0).item()
                # 更新模型参数
                optimizer.step()
                # 清空梯度
                optimizer.zero_grad()
            # 累加轮次损失（还原累积前的值）
            epoch_loss += loss.item() * accum_steps
        # 计算平均轮次损失
        avg_loss = epoch_loss / len(train_loader)
        # 检查损失值是否有效，若无效替换为 0
        avg_loss = avg_loss if not (np.isnan(avg_loss) or np.isinf(avg_loss)) else 0.0
        # 计算平均梯度范数
        avg_grad_norm = grad_norm / (len(train_loader) // accum_steps)
        # 记录训练损失和梯度范数
        train_losses.append(avg_loss)
        grad_norms.append(avg_grad_norm)
        # 设置模型为评估模式
        model.eval()
        # 初始化验证损失
        val_loss = 0
        # 无梯度计算
        with torch.no_grad():
            # 遍历验证数据
            for src, tgt in val_loader:
                src, tgt = src.to(device), tgt.to(device)
                src_mask, tgt_mask = create_masks(src, tgt)
                # 全精度前向传播
                output, _, _, _ = model(src, tgt[:, :-1], src_mask, tgt_mask[:, :, :-1, :-1])
                # 计算验证损失
                val_loss += criterion(output.reshape(-1, output.size(-1)), tgt[:, 1:].reshape(-1)).item()
        # 计算平均验证损失
        val_loss /= len(val_loader)
        # 检查验证损失是否有效
        val_loss = val_loss if not (np.isnan(val_loss) or np.isinf(val_loss)) else 0.0
        # 记录验证损失
        val_losses.append(val_loss)
        # 根据验证损失调整学习率
        # scheduler.step(val_loss)
        # # 清除旧曲线，保留坐标轴设置
        # ax1.clear()
        # # 创建 x 轴值，从 1 到当前 epoch+1
        # epochs = list(range(1, len(train_losses) + 1))
        # # 绘制训练和验证损失曲线
        # ax1.plot(epochs, train_losses, label="训练损失", color='blue')
        # ax1.plot(epochs, val_losses, label="验证损失", color='orange')
        # # 设置 x 轴刻度
        # ax1.set_xticks(range(1, num_epochs + 1, 10))
        # # 动态设置 y 轴刻度，过滤无效值
        # if train_losses and val_losses:
        #     valid_losses = [x for x in train_losses + val_losses if not (np.isnan(x) or np.isinf(x))]
        #     if valid_losses:
        #         y_min = min(valid_losses) * 0.95
        #         y_max = max(valid_losses) * 1.05
        #         y_max = max(y_max, 1.0)
        #         y_ticks = np.linspace(y_min, y_max, num=10)
        #         ax1.set_yticks(y_ticks)
        #     else:
        #         ax1.set_yticks(np.linspace(0, 1, num=10))
        # # 设置损失图标题和标签
        # ax1.set_title("训练和验证损失")
        # ax1.set_xlabel("轮数")
        # ax1.set_ylabel("损失")
        # ax1.legend()
        # # 清除梯度曲线
        # ax2.clear()
        # # 绘制梯度范数曲线
        # ax2.plot(epochs, grad_norms, label="梯度范数", color='green')
        # # 设置 x 轴刻度
        # ax2.set_xticks(range(1, num_epochs + 1, 10))
        # # 动态设置 y 轴刻度
        # if grad_norms:
        #     valid_grads = [x for x in grad_norms if not (np.isnan(x) or np.isinf(x))]
        #     if valid_grads:
        #         y_min = min(valid_grads) * 0.95
        #         y_max = max(valid_grads) * 1.05
        #         y_ticks = np.linspace(y_min, y_max, num=10)
        #         ax2.set_yticks(y_ticks)
        #     else:
        #         ax2.set_yticks(np.linspace(0, 1, num=10))
        # # 设置梯度图标题和标签
        # ax2.set_title("梯度范数")
        # ax2.set_xlabel("轮数")
        # ax2.set_ylabel("梯度")
        # ax2.legend()
        # # 优化图形布局
        # plt.tight_layout()
        # # 强制刷新图形
        # plt.draw()
        # # 暂停以更新图表
        # plt.pause(0.1)
        # 打印训练信息
        print(f"轮数 {epoch+1}/{num_epochs}, 训练损失: {avg_loss:.4f}, 验证损失: {val_loss:.4f}, 梯度范数: {avg_grad_norm:.4f}")
        if train_losses and val_losses:
            print(f"训练损失范围: [{min(train_losses):.4f}, {max(train_losses):.4f}], 验证损失范围: [{min(val_losses):.4f}, {max(val_losses):.4f}]")

    # # 关闭交互模式
    # plt.ioff()
    # # 保存最终图表
    # plt.savefig("training_metrics.png")
    # # 显示图表
    # plt.show()
    # 返回训练指标
    return train_losses, val_losses, grad_norms

# 主函数
if __name__ == "__main__":
    # 选择设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 创建分词器实例
    tokenizer = QACSVCharTokenizer()
    # 加载分词器
    tokenizer.load("qa_tokenizer2.json")
    # 创建数据集实例
    dataset = QADataset("39health_qa_data2.csv", tokenizer, max_len = 512)
    # 计算训练集和验证集大小
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    # 随机分割数据集
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)
    # 初始化 Transformer 模型
    model = Transformer(
        src_vocab_size=10000,
        tgt_vocab_size=10000,
        d_model=512,
        num_heads=8,
        num_layers=6,
        d_ff=2048,
        max_seq_len=512,
        dropout=0.1
    ).to(device)
    # 训练模型，设置 accum_steps=1（默认无梯度累积）
    train_losses, val_losses, grad_norms = train_model(model, train_loader, val_loader, num_epochs=5000, device=device, accum_steps=1)
    # 保存模型权重
    torch.save(model.state_dict(), "transformer_model2.pth")