# 导入 Hugging Face 的 tokenizers 库，提供 BBPE 分词功能
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
# 导入后处理器模块，用于处理特殊 token
from tokenizers.processors import TemplateProcessing
# 导入 pandas 库，用于处理 CSV 数据
import pandas as pd
# 导入 os 模块，用于文件操作
import os

# 定义基于问答数据集的 BBPE 分词器类
class QACSVCharTokenizer:
    # 初始化分词器
    # 参数：vocab_size（词汇表大小，默认10000），min_freq（最小词频，默认2）
    def __init__(self, vocab_size=10000, min_freq=2):
        # 保存词汇表大小，控制分词器的词汇量
        self.vocab_size = vocab_size
        # 保存最小词频，过滤低频词
        self.min_freq = min_freq
        # 初始化分词器对象为 None，等待训练
        self.tokenizer = None

    # 训练分词器，使用 CSV 文件中的问答数据
    def train(self, csv_file):
        # 读取 CSV 文件，显式指定 UTF-8 编码以正确处理中文
        # 原理：确保读取的文本数据不会因编码问题导致乱码
        df = pd.read_csv(csv_file, encoding='utf-8')
        # 合并问题和答案文本，形成训练语料
        # 原理：BBPE 需要大量文本数据来学习子词单元，合并问答增加语料多样性
        texts = df['question'].tolist() + df['answer'].tolist()
        # 初始化 BBPE 模型
        # 原理：BBPE（Byte-Pair Encoding）通过合并高频字符对构建词汇表
        self.tokenizer = Tokenizer(models.BPE())
        # 使用 Whitespace 预分词器，按空格和标点分割
        # 修改：替换 ByteLevel 以保留完整的中文字符，避免字节级拆分导致乱码
        # 原理：中文文本无需字节级分割，Whitespace 更适合按语义单元分割
        self.tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        # 配置后处理器，处理特殊 token 的拼接
        # 修改：添加 TemplateProcessing，确保 [CLS] 和 [SEP] 正确处理
        self.tokenizer.post_processor = TemplateProcessing(
            single="[CLS] $A [SEP]",
            pair="[CLS] $A [SEP] $B:1 [SEP]:1",
            special_tokens=[
                ("[CLS]", 2),  # [CLS] 的 ID
                ("[SEP]", 3),  # [SEP] 的 ID
            ]
        )
        # 配置训练器，设置词汇表大小、最小词频和特殊 token
        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            min_frequency=self.min_freq,
            special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
        )
        # 使用文本数据训练分词器
        # 原理：BBPE 迭代合并高频字符对，生成子词单元
        self.tokenizer.train_from_iterator(texts, trainer=trainer)
        # 启用填充，最大长度为 512
        # 原理：统一序列长度，便于批量处理
        self.tokenizer.enable_padding(length=512)

    # 编码函数，将文本转换为 token ID 序列
    def encode(self, text):
        # 使用分词器将文本编码为 ID 序列
        # 原理：文本被拆分为子词单元，每个单元映射到词汇表中的 ID
        return self.tokenizer.encode(text).ids

    # 解码函数，将 token ID 序列转换为文本
    def decode(self, ids):
        # 过滤填充 token（ID=0）
        # 修改：显式移除 [PAD] token，避免解码时引入无效字符
        ids = [id for id in ids if id != 0]
        # 使用分词器将 ID 序列解码为文本
        # 原理：将 ID 映射回子词单元并拼接为完整文本
        return self.tokenizer.decode(ids)

    # 保存分词器到文件
    def save(self, path):
        # 将分词器模型保存到指定路径
        self.tokenizer.save(path)

    # 从文件加载分词器
    def load(self, path):
        # 从指定路径加载分词器模型
        self.tokenizer = Tokenizer.from_file(path)

# 示例代码：测试分词器功能
if __name__ == "__main__":
    # 创建分词器实例
    tokenizer = QACSVCharTokenizer()
    # 训练分词器，使用指定的 CSV 文件
    tokenizer.train("39health_qa_data2.csv")
    # 保存分词器到文件
    tokenizer.save("qa_tokenizer2.json")
    # 测试文本
    sample_text = "甲状腺结节伴钙化就是甲状腺癌吗"
    # 编码测试文本
    encoded = tokenizer.encode(sample_text)
    # 打印编码结果
    print(f"编码: {encoded}")
    # 解码编码结果
    decoded = tokenizer.decode(encoded)
    # 打印解码结果
    print(f"解码: {decoded}")