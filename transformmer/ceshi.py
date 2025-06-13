# 导入 PyTorch 核心库
import torch
# 导入 Transformer 模型
from model import Transformer
# 导入分词器
from tokenizer import QACSVCharTokenizer

# 定义束搜索解码函数，生成高质量的输出序列
def beam_search(model, src, tokenizer, beam_width=3, max_len=512, device="cuda"):
    # 设置模型为评估模式
    model.eval()
    # 将源序列移动到指定设备
    src = src.to(device)
    # 创建源序列掩码，屏蔽填充 token
    src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
    # 初始化束，包含初始序列（[CLS]）和分数 0.0
    sequences = [([tokenizer.tokenizer.token_to_id("[CLS]")], 0.0)]
    # 初始化已完成序列列表
    finished_sequences = []
    # 循环生成 max_len 个 token
    for _ in range(max_len):
        # 初始化候选序列列表
        all_candidates = []
        # 遍历当前束中的序列
        for seq, score in sequences:
            # 将序列转换为张量
            tgt = torch.tensor([seq], dtype=torch.long).to(device)
            # 创建目标序列掩码，屏蔽填充和未来 token
            tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)
            # 创建未来信息掩码（上三角矩阵）
            nopeak_mask = torch.triu(torch.ones(tgt.size(1), tgt.size(1)), diagonal=1).bool().to(device)
            # 结合填充掩码和未来信息掩码
            tgt_mask = tgt_mask & (~nopeak_mask)
            # 无梯度计算，节省内存
            with torch.no_grad():
                # 前向传播，获取输出概率
                output, _, _, _ = model(src, tgt, src_mask, tgt_mask)
                # 对最后一个 token 的输出应用 softmax
                probs = torch.softmax(output[:, -1, :], dim=-1)
                # 获取概率最高的 beam_width 个 token 及其概率
                top_probs, top_idx = probs.topk(beam_width, dim=-1)
            # 为每个 top token 生成新序列
            for i in range(beam_width):
                # 获取下一个 token 的 ID
                next_token = top_idx[0, i].item()
                # 计算新序列的累计对数概率
                # 公式：score = score + log(P(next_token))
                next_score = score + torch.log(top_probs[0, i]).item()
                # 创建新序列
                new_seq = seq + [next_token]
                # 添加到候选序列
                all_candidates.append((new_seq, next_score))
                # 如果遇到 [SEP]，将序列添加到已完成序列
                if next_token == tokenizer.tokenizer.token_to_id("[SEP]"):
                    finished_sequences.append((new_seq, next_score))
        # 选择得分最高的 beam_width 个序列
        # 原理：束搜索通过维护 beam_width 个候选序列，平衡质量和计算成本
        sequences = sorted(all_candidates, key=lambda x: x[1], reverse=True)[:beam_width]
        # 如果已完成序列足够多，停止搜索
        if len(finished_sequences) >= beam_width:
            break
    # 选择得分最高的序列（优先从已完成序列中选择）
    best_seq = max(finished_sequences or sequences, key=lambda x: x[1])[0]
    # 返回最佳序列
    return best_seq

# 定义测试函数
def test_model(model_path, tokenizer_path, question, device="cuda"):
    # 创建分词器实例
    tokenizer = QACSVCharTokenizer()
    # 加载分词器
    tokenizer.load(tokenizer_path)
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
    # 加载模型权重
    model.load_state_dict(torch.load(model_path))
    # 设置模型为评估模式
    model.eval()
    # 编码问题，截断到 512
    src = torch.tensor([tokenizer.encode(question)[:512]], dtype=torch.long).to(device)
    # 使用束搜索生成输出序列
    output_ids = beam_search(model, src, tokenizer, beam_width=3, max_len=512, device=device)
    # 解码输出序列
    answer = tokenizer.decode(output_ids)
    # 打印问题
    print(f"问题: {question}")
    # 打印生成答案
    print(f"回答: {answer}")

# 示例测试代码
if __name__ == "__main__":
    # 调用测试函数
    test_model(
        model_path="transformer_model.pth",
        tokenizer_path="qa_tokenizer.json",
        question="右侧腋下疼痛是什么原因"
    )