# 导入 PyTorch 核心库，用于张量计算和神经网络构建
import torch
# 导入 PyTorch 的神经网络模块，提供层、损失函数等功能
import torch.nn as nn
# 导入 math 库，用于数学计算，如 sqrt 和 log
import math

# 定义多头注意力机制类，继承自 nn.Module，用于实现 Transformer 的核心注意力机制
class MultiHeadAttention(nn.Module):
    # 初始化多头注意力模块
    # 参数：d_model（模型维度，通常为512），num_heads（注意力头数，通常为8）
    def __init__(self, d_model, num_heads):
        # 调用父类构造函数，初始化 nn.Module
        super(MultiHeadAttention, self).__init__()
        # 确保模型维度能被头数整除，每个头的维度为 d_model / num_heads
        # 原理：多头注意力将输入维度分割为多个子空间，增强模型并行计算能力
        assert d_model % num_heads == 0
        # 保存模型维度（d_model=512），用于后续计算
        self.d_model = d_model
        # 保存注意力头数（num_heads=8），控制并行计算的头数
        self.num_heads = num_heads
        # 计算每个头的维度，d_k = d_model / num_heads（如512/8=64）
        self.d_k = d_model // num_heads
        # 定义查询（Q）的线性变换层，将输入从 d_model 映射到 d_model
        # 原理：线性变换为 Q 提供可学习的投影，提取特征
        self.W_q = nn.Linear(d_model, d_model)
        # 定义键（K）的线性变换层，功能同上
        self.W_k = nn.Linear(d_model, d_model)
        # 定义值（V）的线性变换层，功能同上
        self.W_v = nn.Linear(d_model, d_model)
        # 定义输出线性变换层，将多头拼接结果映射回 d_model 维度
        # 原理：多头输出需要重新整合，确保维度一致
        self.W_o = nn.Linear(d_model, d_model)

    # 定义缩放点积注意力机制，核心计算公式：Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        # 计算注意力分数：QK^T / sqrt(d_k)
        # 原理：点积 QK^T 计算查询与键的相似度，除以 sqrt(d_k) 防止数值过大（d_k=64）
        # 计算公式：attention_scores = (Q * K^T) / sqrt(d_k)
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        # 如果提供了掩码（如未来信息掩码或填充掩码），将无效位置的分数置为负无穷
        # 原理：负无穷在 softmax 后趋近于 0，屏蔽无关 token
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        # 对注意力分数应用 softmax，归一化为概率分布
        # 计算公式：attention_probs = softmax(attention_scores)
        attention_probs = torch.softmax(attention_scores, dim=-1)
        # 计算注意力加权后的值：attention_probs * V
        # 原理：使用注意力概率加权 V，提取相关信息
        output = torch.matmul(attention_probs, V)
        # 返回注意力输出和注意力概率（用于可视化或分析）
        return output, attention_probs

    # 定义前向传播，执行多头注意力计算
    def forward(self, Q, K, V, mask=None):
        # 获取批量大小（batch_size），用于张量形状处理
        batch_size = Q.size(0)
        # 对 Q 进行线性变换并分割为多头
        # 变换后形状：(batch_size, seq_len, num_heads, d_k)
        # 原理：将 d_model 维度分割为 num_heads 个子空间，增强并行性
        Q = self.W_q(Q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        # 对 K 进行线性变换并分割为多头，形状同上
        K = self.W_k(K).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        # 对 V 进行线性变换并分割为多头，形状同上
        V = self.W_v(V).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        # 调用缩放点积注意力，计算每头的注意力输出
        output, attention_probs = self.scaled_dot_product_attention(Q, K, V, mask)
        # 拼接多头结果，转换为 (batch_size, seq_len, d_model)
        # 修改：将 (batch_size, num_heads, seq_len, d_k) 重塑为 (batch_size, seq_len, num_heads * d_k)
        # 原理：多头输出的每个头维度 d_k，拼接后恢复为 d_model = num_heads * d_k
        output =  output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        # 对拼接结果进行线性变换，得到最终输出
        # 计算公式：output = W_o * output + b_o
        output = self.W_o(output)
        # 返回注意力输出和注意力概率
        return output, attention_probs

# 定义前馈神经网络模块，包含两层线性变换和激活函数
class FeedForward(nn.Module):
    # 初始化前馈网络
    # 参数：d_model（输入输出维度），d_ff（中间层维度，通常为2048），dropout（丢弃率）
    def __init__(self, d_model, d_ff, dropout=0.1):
        # 调用父类构造函数
        super(FeedForward, self).__init__()
        # 第一层线性变换，将 d_model 映射到 d_ff（512 -> 2048）
        # 原理：扩展维度以增加模型容量
        self.linear1 = nn.Linear(d_model, d_ff)
        # 定义 dropout 层，防止过拟合
        self.dropout = nn.Dropout(dropout)
        # 第二层线性变换，将 d_ff 映射回 d_model（2048 -> 512）
        self.linear2 = nn.Linear(d_ff, d_model)
        # ReLU 激活函数，增加非线性
        self.relu = nn.ReLU()

    # 定义前向传播
    def forward(self, x):
        # 第一层线性变换 + ReLU 激活
        x = self.relu(self.linear1(x))
        # 应用 dropout，随机丢弃部分神经元
        x = self.dropout(x)
        # 第二层线性变换，恢复维度
        x = self.linear2(x)
        # 返回前馈网络输出
        return x

# 定义 Transformer 编码器层
class EncoderLayer(nn.Module):
    # 初始化编码器层
    # 参数：d_model（模型维度），num_heads（注意力头数），d_ff（前馈层维度），dropout（丢弃率）
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        # 调用父类构造函数
        super(EncoderLayer, self).__init__()
        # 初始化多头自注意力模块
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        # 初始化第一层归一化，稳定训练
        # 原理：归一化使每一层的输出分布更稳定，加速收敛
        self.norm1 = nn.LayerNorm(d_model)
        # 初始化前馈网络模块
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        # 初始化第二层归一化
        self.norm2 = nn.LayerNorm(d_model)
        # 定义 dropout 层
        self.dropout = nn.Dropout(dropout)

    # 定义前向传播
    def forward(self, x, mask=None):
        # 自注意力计算，输入 Q、K、V 均为 x（自注意力）
        attn_output, attn_probs = self.self_attention(x, x, x, mask)
        # 残差连接 + 归一化：x = LayerNorm(x + Dropout(attn_output))
        # 原理：残差连接缓解梯度消失问题，归一化稳定训练
        x = self.norm1(x + self.dropout(attn_output))
        # 前馈网络计算
        ff_output = self.feed_forward(x)
        # 残差连接 + 归一化
        x = self.norm2(x + self.dropout(ff_output))
        # 返回编码器输出和注意力概率
        return x, attn_probs

# 定义 Transformer 解码器层
class DecoderLayer(nn.Module):
    # 初始化解码器层
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        # 调用父类构造函数
        super(DecoderLayer, self).__init__()
        # 初始化自注意力模块（解码器自注意力）
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        # 初始化第一层归一化
        self.norm1 = nn.LayerNorm(d_model)
        # 初始化交叉注意力模块（解码器与编码器交互）
        self.cross_attention = MultiHeadAttention(d_model, num_heads)
        # 初始化第二层归一化
        self.norm2 = nn.LayerNorm(d_model)
        # 初始化前馈网络模块
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        # 初始化第三层归一化
        self.norm3 = nn.LayerNorm(d_model)
        # 定义 dropout 层
        self.dropout = nn.Dropout(dropout)

    # 定义前向传播
    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        # 解码器自注意力，Q、K、V 均为目标序列 x
        # tgt_mask 防止关注未来 token
        attn_output, self_attn_probs = self.self_attention(x, x, x, tgt_mask)
        # 残差连接 + 归一化
        x = self.norm1(x + self.dropout(attn_output))
        # 交叉注意力，Q 来自解码器，K、V 来自编码器输出
        # src_mask 屏蔽源序列中的填充 token
        attn_output, cross_attn_probs = self.cross_attention(x, enc_output, enc_output, src_mask)
        # 残差连接 + 归一化
        x = self.norm2(x + self.dropout(attn_output))
        # 前馈网络计算
        ff_output = self.feed_forward(x)
        # 残差连接 + 归一化
        x = self.norm3(x + self.dropout(ff_output))
        # 返回解码器输出、自注意力概率和交叉注意力概率
        return x, self_attn_probs, cross_attn_probs

# 定义 Transformer 模型
class Transformer(nn.Module):
    # 初始化 Transformer 模型
    # 参数包括源/目标词汇表大小、模型维度、头数、层数、前馈维度、最大序列长度、dropout 率
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_heads=8, num_layers=6, d_ff=2048, max_seq_len=512, dropout=0.1):
        # 调用父类构造函数
        super(Transformer, self).__init__()
        # 保存模型维度
        self.d_model = d_model
        # 保存最大序列长度
        self.max_seq_len = max_seq_len
        # 源序列词嵌入层，将 token ID 转换为 d_model 维向量
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        # 目标序列词嵌入层
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        # 创建位置编码，形状为 (1, max_seq_len, d_model)
        self.positional_encoding = self.create_positional_encoding(max_seq_len, d_model)
        # 定义 dropout 层，应用于嵌入和位置编码
        self.dropout = nn.Dropout(dropout)
        # 初始化编码器层列表，包含 num_layers 个编码器层
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        # 初始化解码器层列表，包含 num_layers 个解码器层
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        # 定义输出线性层，将 d_model 映射到目标词汇表大小
        self.fc = nn.Linear(d_model, tgt_vocab_size)
        # 定义 softmax 层，将输出转换为概率分布
        self.softmax = nn.Softmax(dim=-1)

    # 创建位置编码函数，生成正弦和余弦位置编码
    def create_positional_encoding(self, max_seq_len, d_model):
        # 初始化位置编码矩阵，形状为 (max_seq_len, d_model)
        pe = torch.zeros(max_seq_len, d_model)
        # 生成位置索引，形状为 (max_seq_len, 1)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        # 计算除数项，用于正弦和余弦函数的频率
        # 公式：div_term = exp(-2i * log(10000) / d_model)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # 偶数维度使用正弦函数：PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        # 奇数维度使用余弦函数：PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
        pe[:, 1::2] = torch.cos(position * div_term)
        # 增加 batch 维度，形状变为 (1, max_seq_len, d_model)
        pe = pe.unsqueeze(0)
        # 返回位置编码矩阵
        return pe

    # 定义前向传播
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # 源序列嵌入：将 token ID 转换为 d_model 维向量
        # 乘以 sqrt(d_model) 调整嵌入向量尺度，稳定梯度
        src_embedded = self.src_embedding(src) * math.sqrt(self.d_model) + self.positional_encoding[:, :src.size(1), :].to(src.device)
        # 对源序列嵌入应用 dropout
        src_embedded = self.dropout(src_embedded)
        # 目标序列嵌入，处理方式同源序列
        tgt_embedded = self.tgt_embedding(tgt) * math.sqrt(self.d_model) + self.positional_encoding[:, :tgt.size(1), :].to(tgt.device)
        # 对目标序列嵌入应用 dropout
        tgt_embedded = self.dropout(tgt_embedded)
        # 编码器处理
        enc_output = src_embedded
        # 初始化编码器注意力概率列表
        enc_attn_probs = []
        # 逐层通过编码器层
        for enc_layer in self.encoder_layers:
            # 编码器层计算，更新 enc_output 和注意力概率
            enc_output, attn_probs = enc_layer(enc_output, src_mask)
            # 存储注意力概率
            enc_attn_probs.append(attn_probs)
        # 解码器处理
        dec_output = tgt_embedded
        # 初始化解码器自注意力和交叉注意力概率列表
        dec_self_attn_probs = []
        dec_cross_attn_probs = []
        # 逐层通过解码器层
        for dec_layer in self.decoder_layers:
            # 解码器层计算，更新 dec_output 和注意力概率
            dec_output, self_attn_probs, cross_attn_probs = dec_layer(dec_output, enc_output, src_mask, tgt_mask)
            # 存储注意力概率
            dec_self_attn_probs.append(self_attn_probs)
            dec_cross_attn_probs.append(cross_attn_probs)
        # 输出层：将解码器输出映射到目标词汇表大小
        output = self.fc(dec_output)
        # 返回输出、编码器注意力概率、解码器自注意力概率和交叉注意力概率
        return output, enc_attn_probs, dec_self_attn_probs, dec_cross_attn_probs