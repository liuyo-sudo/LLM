import os
import torch
import shutil
import time
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
import re

# 打印环境信息
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU count: {torch.cuda.device_count()}")
    print(f"GPU memory allocated: {torch.cuda.memory_allocated(0)/1e9:.2f} GB")
    print(f"GPU memory reserved: {torch.cuda.memory_reserved(0)/1e9:.2f} GB")
else:
    print("No GPU detected, falling back to CPU")

# 设置路径
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
dataset_name = "open-r1/Mixture-of-Thoughts"
output_dir = "deepseek_r1_lora_finetuned"
logging_dir = "deepseek_r1_logs"

# 增强目录清理逻辑
def clean_directory(directory):
    print(f"Checking directory: {directory}")
    retries = 3
    for attempt in range(retries):
        try:
            if os.path.exists(directory):
                if not os.path.isdir(directory):
                    print(f"Removing file at {directory} as it is not a directory")
                    os.remove(directory)
                else:
                    print(f"Clearing directory contents at {directory}")
                    shutil.rmtree(directory, ignore_errors=True)
            os.makedirs(directory, exist_ok=True)
            print(f"Directory {directory} created successfully")
            return True
        except (OSError, PermissionError) as e:
            print(f"Attempt {attempt + 1} failed to clean {directory}: {e}")
            if attempt < retries - 1:
                time.sleep(1)
            else:
                print(f"Failed to clean {directory} after {retries} attempts")
                return False
    return False

# 清理目录
for directory in [output_dir, logging_dir]:
    if not clean_directory(directory):
        raise RuntimeError(f"Unable to prepare directory {directory}")

# 验证目录状态
for directory in [output_dir, logging_dir]:
    if not os.path.isdir(directory):
        raise RuntimeError(f"Directory {directory} is not a valid directory")

# determin device
device = torch.device("cuda:0")
print(torch.__version__)
print(f"Using device: {device}")

# 加载分词器
try:
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("Tokenizer loaded successfully")
except Exception as e:
    print(f"Failed to load tokenizer: {e}")
    raise

# 加载模型
try:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=None,
        trust_remote_code=True,
        use_cache=False,  # 显式禁用缓存
    ).to(device)
    print("Model loaded successfully")
except Exception as e:
    print(f"Failed to load model: {e}")
    if "CUDA" in str(e):
        print("CUDA error detected. Try running on CPU or check CUDA setup.")
    elif "out of memory" in str(e).lower():
        print("Out of memory error. Try 8-bit quantization or reduce max_length.")
    raise

# 配置LoRA
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)

# 启用梯度检查点
model.enable_input_require_grads()
model.gradient_checkpointing_enable()

# 检查可训练参数
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"可训练参数数量: {trainable_params} / {total_params} ({trainable_params/total_params*100:.2f}%)")
print(f"Model device: {[p.device for p in model.parameters()][0]}")

# 加载数据集
try:
    dataset = load_dataset(dataset_name, 'all')['train']
    eval_dataset = load_dataset(dataset_name, 'all')['train'].select(range(1000, 1100))
    print(f"原始数据集列名: {dataset.column_names}")
except Exception as e:
    print(f"加载数据集失败: {e}")
    raise

# 数据预处理
def format_prompt(example):
    if 'messages' not in example:
        raise KeyError("数据集缺少 'messages' 字段")
    messages = example['messages']
    if not isinstance(messages, list) or len(messages) < 2:
        raise ValueError(f"无效的 messages 字段: {messages}")
    user_content = next((msg['content'] for msg in messages if msg['role'] == 'user'), '')
    assistant_content = next((msg['content'] for msg in messages if msg['role'] == 'assistant'), '')
    reasoning_match = re.search(r'<think>(.*?)</think>', assistant_content, re.DOTALL)
    reasoning = reasoning_match.group(1).strip() if reasoning_match else ''
    response = re.sub(r'<think>.*?</think>', '', assistant_content, flags=re.DOTALL).strip()
    prompt = f"{user_content}\n<think>{reasoning}</think>\n{response}"
    if not prompt:
        raise ValueError(f"格式化后的提示为空: {example}")
    print(f"格式化后的提示长度: {len(prompt)}")  # 调试打印
    return {"text": prompt}

# 先应用格式化
dataset = dataset.map(format_prompt)
eval_dataset = eval_dataset.map(format_prompt)

# 验证格式化后的数据集列名
print(f"格式化后的数据集列名: {dataset.column_names}")

# 过滤数据集
dataset = dataset.filter(lambda x: len(x['text']) > 0 and len(x['text']) < 10000)
eval_dataset = eval_dataset.filter(lambda x: len(x['text']) > 0 and len(x['text']) < 10000)

# 分词
def tokenize_function(example):
    tokenized = tokenizer(
        example["text"],
        truncation=True,
        max_length=4096,
        padding="max_length",
        return_tensors=None,
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    if not all(isinstance(x, int) for x in tokenized["labels"]):
        raise ValueError(f"标签包含非整数值: {tokenized['labels'][:10]}")
    if len(tokenized["input_ids"]) != len(tokenized["labels"]):
        raise ValueError(f"输入 ID 和标签长度不一致: {len(tokenized['input_ids'])} vs {len(tokenized['labels'])}")
    print(f"样本 input_ids (前10个): {tokenized['input_ids'][:10]}")
    print(f"样本 labels (前10个): {tokenized['labels'][:10]}")
    return tokenized

tokenized_dataset = dataset.map(
    tokenize_function,
    batched=False,
    remove_columns=['messages', 'num_tokens', 'source', 'text']
)
tokenized_eval_dataset = eval_dataset.map(
    tokenize_function,
    batched=False,
    remove_columns=['messages', 'num_tokens', 'source', 'text']
)

# 配置训练参数
training_args = TrainingArguments(
    output_dir=output_dir,
    logging_dir=logging_dir,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=1,
    learning_rate=5e-5,
    weight_decay=0.01,
    fp16=True,
    logging_steps=5,
    save_steps=100,
    save_total_limit=2,
    report_to="tensorboard",
    gradient_checkpointing=True,
    remove_unused_columns=False,
    lr_scheduler_type="linear",
    label_names=["labels"],
    use_cpu=False if torch.cuda.is_available() else True,
)

# 初始化数据整理器
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
    pad_to_multiple_of=8,
    return_tensors="pt"
)

# 自定义 SFTTrainer 以调试损失计算
class CustomSFTTrainer(SFTTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if "labels" not in inputs:
            raise ValueError("输入缺少 'labels' 键")
        outputs = model(**inputs)
        loss = outputs.loss
        logits = outputs.logits
        labels = inputs["labels"]
        print(f"Logits shape: {logits.shape}")
        print(f"Labels shape: {labels.shape}")
        return (loss, outputs) if return_outputs else loss

# 初始化 Trainer
trainer = CustomSFTTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_eval_dataset,
    data_collator=data_collator,
)

# 调试批次数据
batch = next(iter(trainer.get_train_dataloader()))
print(f"批次 input_ids 形状: {batch['input_ids'].shape}")
print(f"批次 labels 形状: {batch['labels'].shape}")
print(f"批次 attention_mask 形状: {batch['attention_mask'].shape}")

# 开始训练
trainer.train()

# 保存模型
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"模型已保存至 {output_dir}")