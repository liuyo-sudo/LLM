import torch


### 矩阵的 秩（Rank） 是线性代数中的一个核心概念，表示一个矩阵的线性无关行或列的最大数量


# 定义矩阵
A = torch.tensor([[1., 2., 3.], [2., 4., 6.], [1., 1., 0.]])

# 方法 1：SVD 计算秩
_, S, _ = torch.svd(A)
rank = torch.sum(S > 1e-5).item()
print(f"通过 SVD 计算的秩: {rank}")

# 方法 2：行简化（QR 分解近似）
Q, R = torch.qr(A)
non_zero_rows = torch.sum(torch.abs(torch.diag(R)) > 1e-5).item()
print(f"通过 QR 分解计算的秩: {non_zero_rows}")

# 应用：检查线性方程组 Ax = b 是否有解
b = torch.tensor([1., 2., 3.])
augmented = torch.cat((A, b.unsqueeze(1)), dim=1)
_, S_aug, _ = torch.svd(augmented)
rank_aug = torch.sum(S_aug > 1e-5).item()
print(f"增广矩阵的秩: {rank_aug}")
print(f"方程组有解: {rank == rank_aug}")

a = torch.randn(5, 3)
u, s, v = torch.svd(a)

rank = torch.sum(S > 1e-5).item()
print(f"通过 SVD 计算的秩: {rank}")


tensor = torch.randn(2, 3)
print("示例张量:\n", tensor)
print("转置后张量:\n", torch.transpose(tensor, 0, 1))
print("张量和:", torch.sum(tensor))

tensor = torch.ones(2, 3)
print("张量和:", torch.sum(tensor))
#torch.gather(tensor, 1, tensor)


import numpy as np
A = np.array([[1, 2, 3], [2, 4, 6], [1, 1, 0]])
_, s, _ = np.linalg.svd(A)
rank = np.sum(s > 1e-10)  # 非零奇异值数量
print(rank)  # 输出：2