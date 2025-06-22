import os
import torch
import shutil
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
import re
import time

# 设置路径
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
dataset_name = "open-r1/Mixture-of-Thoughts"
output_dir = "./deepseek_r1_lora_finetuned"
logging_dir = "./deepseek_r1_logs"

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
                time.sleep(1)  # 等待文件解锁
            else:
                print(f"Failed to clean {directory} after {retries} attempts")
                return False
    return False

# 清理输出和日志目录
for directory in [output_dir, logging_dir]:
    if not clean_directory(directory):
        raise RuntimeError(f"Unable to prepare directory {directory}")

# 验证目录状态
for directory in [output_dir, logging_dir]:
    if not os.path.isdir(directory):
        raise RuntimeError(f"Directory {directory} is not a valid directory")

# 确定设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 配置4-bit量化
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

# 加载模型
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    device_map=None,
    trust_remote_code=True,
).to(device)

# 配置LoRA
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.1,
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
dataset = load_dataset(dataset_name, 'all', split="train[:1000]")
eval_dataset = load_dataset(dataset_name, 'all', split="train[1000:1100]")

# 数据预处理
def format_prompt(example):
    if 'messages' in example:
        messages = example['messages']
    elif 'message' in example:
        messages = example['message']
    elif 'conversation' in example:
        messages = example['conversation']
    elif 'prompt' in example and 'output' in example:
        prompt = f"{example['prompt']}\n<think>{example.get('reasoning', '')}</think>\n{example['output']}"
        return {"text": prompt}
    else:
        raise KeyError("未知的数据集字段")

    if not isinstance(messages, list) or len(messages) < 2:
        raise ValueError("messages字段必须是包含user和assistant的列表")

    user_content = next((msg['content'] for msg in messages if msg['role'] == 'user'), '')
    assistant_content = next((msg['content'] for msg in messages if msg['role'] == 'assistant'), '')
    reasoning_match = re.search(r'<think>(.*?)</think>', assistant_content, re.DOTALL)
    reasoning = reasoning_match.group(1).strip() if reasoning_match else ''
    response = re.sub(r'<think>.*?</think>', '', assistant_content, flags=re.DOTALL).strip()
    prompt = f"{user_content}\n<think>{reasoning}</think>\n{response}"
    return {"text": prompt}

dataset = dataset.map(format_prompt)
eval_dataset = eval_dataset.map(format_prompt)

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
        raise ValueError(f"Labels contain non-integer values: {tokenized['labels'][:10]}")
    print(f"Sample input_ids (first 10): {tokenized['input_ids'][:10]}")
    print(f"Sample labels (first 10): {tokenized['labels'][:10]}")
    return tokenized

tokenized_dataset = dataset.map(tokenize_function, batched=False, remove_columns=dataset.column_names)
tokenized_eval_dataset = eval_dataset.map(tokenize_function, batched=False, remove_columns=eval_dataset.column_names)

# 配置训练参数
training_args = TrainingArguments(
    output_dir=output_dir,
    logging_dir=logging_dir,
    num_train_epochs=1,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=2e-5,
    weight_decay=0.01,
    fp16=True,
    logging_steps=10,
    save_steps=100,
    save_total_limit=2,
    evaluation_strategy="steps",
    eval_steps=100,
    report_to="none",  # 临时禁用TensorBoard
    gradient_checkpointing=True,
    remove_unused_columns=False,
    lr_scheduler_type="linear",
    label_names=["labels"],
)

# 自定义SFTTrainer
class CustomSFTTrainer(SFTTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch = None):
        if "labels" not in inputs:
            raise ValueError("Inputs missing 'labels' key")
        outputs = model(**inputs)
        loss = outputs.loss
        print(f"Loss: {loss.item() if loss is not None else 'None'}")
        print(f"Loss requires grad: {loss.requires_grad if loss is not None else 'None'}")
        if loss is None or not loss.requires_grad:
            raise ValueError("Loss is None or does not require grad")
        return (loss, outputs) if return_outputs else loss

# 初始化Trainer
trainer = CustomSFTTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_eval_dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False, pad_to_multiple_of=8),
    tokenizer=tokenizer,
    max_seq_length=4096,
)

# 调试批次数据
batch = next(iter(trainer.get_train_dataloader()))
print(f"Batch input_ids shape: {batch['input_ids'].shape}")
print(f"Batch labels shape: {batch['labels'].shape}")
print(f"Batch attention_mask shape: {batch['attention_mask'].shape}")

# 开始训练
trainer.train()

# 保存模型
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"模型已保存至 {output_dir}")