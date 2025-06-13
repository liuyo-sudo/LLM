import torch
import matplotlib.pyplot as plt

# 设置随机种子
torch.manual_seed(42)

# 生成标准正态分布样本
t = torch.randn(10000)

# 计算均值和标准差
print(f"均值: {t.mean().item():.4f}, 标准差: {t.std().item():.4f}")

# 绘制直方图
plt.hist(t.numpy(), bins=100, density=True, alpha=0.7, label="Histogram")
x = torch.linspace(-4, 4, 100)
pdf = 1 / (2 * 3.1416)**0.5 * torch.exp(-x**2 / 2)  # N(0, 1) 的 PDF
plt.plot(x.numpy(), pdf.numpy(), "r-", label="N(0, 1) PDF")
plt.legend()
plt.title("Standard Normal Distribution")
plt.show()