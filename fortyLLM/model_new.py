import pandas as pd
from datasets import load_dataset, concatenate_datasets
from transformers import Qwen2Config, Qwen2ForCausalLM, AutoTokenizer, TrainingArguments, Trainer
import torch
import json


def preprocess_function(examples, tokenizer, max_length=512):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    question = examples['question']
    human_answers = examples['human_answers']
    texts = []
    for q, a in zip(question, human_answers):
        h = ""
        if isinstance(a, list) and len(a) > 0 and a[0] is not None:
            h = a[0].strip()
        elif isinstance(a, str) and a.strip():
            h = a.strip()
        q = q.strip() if q and isinstance(q, str) else ""
        if q or h:
            text = f"{q}\n{h}" if q and h else q or h
            texts.append(text)
    if not texts:
        texts = ["[PAD]"]

    # 分词并生成 labels
    encodings = tokenizer(
        texts,
        truncation=True,
        padding='max_length',
        max_length=max_length,
        return_tensors="pt"
    )

    # 生成 labels（复制 input_ids）
    encodings['labels'] = encodings['input_ids'].clone()
    # 将填充 token 的 labels 设置为 -100
    encodings['labels'][encodings['input_ids'] == tokenizer.pad_token_id] = -100

    # 将所有张量移到指定设备
    for key in encodings:
        encodings[key] = encodings[key].to(device)

    return encodings


def main():
    # 加载预训练配置和分词器
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-1.5B")

    config = Qwen2Config.from_pretrained("Qwen/Qwen2-1.5B", torch_dtype=torch.bfloat16)
    config.vocab_size = tokenizer.vocab_size
    config.hidden_size = 512
    config.num_hidden_layers = 6
    config.num_attention_heads = 8
    config.intermediate_size = 2048
    config.max_position_embeddings = 512
    model = Qwen2ForCausalLM(config=config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.resize_token_embeddings(len(tokenizer))

    # 加载和预处理数据集
    datasets = []

    # 1. Hello-SimpleAI/HC3-Chinese 数据集
    hc3_subsets = ["all", "baike", "finance", "medicine", "law", "nlpcc_dbqa", "open_qa", "psychology"]
    for subset in hc3_subsets:
        dataset = load_dataset("Hello-SimpleAI/HC3-Chinese", name=subset, split="train")
        dataset = dataset.filter(lambda x: x['question'] and x['human_answers'] and x['human_answers'][0])
        datasets.append(dataset)

    # 2. Hello-SimpleAI/HC3 数据集
    hc3_en_subsets = ["all", "finance", "medicine", "open_qa", "reddit_eli5", "wiki_csai"]
    for subset in hc3_en_subsets:
        dataset = load_dataset("Hello-SimpleAI/HC3", name=subset, split="train")
        dataset = dataset.filter(lambda x: x['question'] and x['human_answers'] and x['human_answers'][0])
        datasets.append(dataset)
    print("开始加载tatsu-lab/alpaca")
    # 3. tatsu-lab/alpaca 数据集
    alpaca_dataset = load_dataset("tatsu-lab/alpaca", split="train")
    alpaca_dataset = alpaca_dataset.filter(lambda x: x['instruction'] and x['output'])
    alpaca_dataset = alpaca_dataset.rename_columns({"instruction": "question", "output": "human_answers"})
    alpaca_dataset = alpaca_dataset.map(lambda x: {"human_answers": [x["human_answers"]]})
    datasets.append(alpaca_dataset)
    # print("shareAI/ShareGPT-Chinese-English-90k")
    # # 4. shareAI/ShareGPT-Chinese-English-90k 数据集
    # sharegpt_dataset = load_dataset("shareAI/ShareGPT-Chinese-English-90k", split="train")
    # def process_sharegpt(example):
    #     conversations = json.loads(example["conversation"]) if isinstance(example["conversation"], str) else example["conversation"]
    #     question = conversations[0]["human"] if conversations else ""
    #     human_answer = conversations[0]["assistant"] if conversations else ""
    #     return {"question": question, "human_answers": [human_answer]}
    # sharegpt_dataset = sharegpt_dataset.filter(lambda x: x['conversation'])
    # sharegpt_dataset = sharegpt_dataset.map(process_sharegpt)
    # sharegpt_dataset = sharegpt_dataset.filter(lambda x: x['question'] and x['human_answers'][0])
    # datasets.append(sharegpt_dataset)
    print("YeungNLP/firefly-train-1.1M")
    # 5. YeungNLP/firefly-train-1.1M 数据集
    firefly_dataset = load_dataset("YeungNLP/firefly-train-1.1M", split="train")
    firefly_dataset = firefly_dataset.filter(lambda x: x['input'] and x['target'])
    firefly_dataset = firefly_dataset.rename_columns({"input": "question", "target": "human_answers"})
    firefly_dataset = firefly_dataset.map(lambda x: {"human_answers": [x["human_answers"]]})
    datasets.append(firefly_dataset)
    print("YeungNLP/ultrachat")
    # 6. YeungNLP/ultrachat 数据集
    ultrachat_dataset = load_dataset("YeungNLP/ultrachat", split="train")
    def process_ultrachat(example):
        conversations = json.loads(example["conversation"]) if isinstance(example["conversation"], str) else example["conversation"]
        question = conversations[0]["human"] if conversations else ""
        human_answer = conversations[0]["assistant"] if conversations else ""
        return {"question": question, "human_answers": [human_answer]}
    ultrachat_dataset = ultrachat_dataset.filter(lambda x: x['conversation'])
    ultrachat_dataset = ultrachat_dataset.map(process_ultrachat)
    ultrachat_dataset = ultrachat_dataset.filter(lambda x: x['question'] and x['human_answers'][0])
    datasets.append(ultrachat_dataset)

    # 合并所有子数据集
    combined_dataset = concatenate_datasets(datasets)

    tokenized_dataset = combined_dataset.map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True,
        remove_columns=['question', 'human_answers', 'chatgpt_answers', 'instruction', 'input', 'output', 'text', 'category', 'conversation', 'target']
    )

    # 训练参数设置
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=2,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=100,
        save_strategy="epoch",
        fp16=False,
        bf16=True,
        report_to="none",
        gradient_accumulation_steps=2,
        learning_rate=5e-5,
        optim="adamw_torch"
    )

    # 初始化Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
    )

    # 开始训练
    trainer.train()

    # 保存模型
    model.save_pretrained("./finetuned_qwen4")
    tokenizer.save_pretrained("./finetuned_qwen4")


if __name__ == "__main__":
    main()