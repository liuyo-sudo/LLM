import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset
from nltk.translate.bleu_score import sentence_bleu
import numpy as np
from vllm import LLM, SamplingParams
import re

# 设置路径
model_path = "deepseek_r1_lora_finetuned"
dataset_name = "open-r1/Mixture-of-Thoughts"
output_file = "./evaluation_results.txt"

# 确保model_path存在
if os.path.exists(model_path) and not os.path.isdir(model_path):
    os.remove(model_path)
os.makedirs(model_path, exist_ok=True)

# 确定设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 加载分词器和模型
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=quantization_config,
    device_map=None,
    trust_remote_code=True,
).to(device)

# 加载测试数据集
test_dataset = load_dataset(dataset_name, 'all', split="test[:100]")

# 数据预处理
def format_prompt(example):
    if 'messages' in example:
        messages = example['messages']
    elif 'message' in example:
        messages = example['message']
    elif 'conversation' in example:
        messages = example['conversation']
    elif 'prompt' in example and 'output' in example:
        prompt = f"{example['prompt']}\n<think></think>\n"
        return {"prompt": prompt, "reference": example['output']}
    else:
        raise KeyError("未知的数据集字段")

    if not isinstance(messages, list) or len(messages) < 2:
        raise ValueError("messages字段必须是包含user和assistant的列表")

    user_content = next((msg['content'] for msg in messages if msg['role'] == 'user'), '')
    assistant_content = next((msg['content'] for msg in messages if msg['role'] == 'assistant'), '')
    reference = re.sub(r'<think>.*?</think>', '', assistant_content, flags=re.DOTALL).strip()
    prompt = f"{user_content}\n<think></think>\n"
    return {"prompt": prompt, "reference": reference}

test_dataset = test_dataset.map(format_prompt)

# 配置vLLM推理
llm = LLM(model=model_path, tensor_parallel_size=1, max_model_len=32768)
sampling_params = SamplingParams(temperature=0.6, top_p=0.95, max_tokens=4096)

# 计算困惑度
def calculate_perplexity(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
        perplexity = torch.exp(loss).item()
    return perplexity

# 评估BLEU
def evaluate_bleu(reference, candidate):
    reference = [reference.split()]
    candidate = candidate.split()
    return sentence_bleu(reference, candidate)

# 推理和评估
results = []
perplexities = []
bleu_scores = []

for example in test_dataset:
    prompt = example["prompt"]
    reference = example["reference"]
    outputs = llm.generate([prompt], sampling_params)
    generated_text = outputs[0].outputs[0].text.strip()
    perplexity = calculate_perplexity(model, tokenizer, prompt + generated_text)
    bleu = evaluate_bleu(reference, generated_text)
    perplexities.append(perplexity)
    bleu_scores.append(bleu)
    results.append({
        "prompt": prompt,
        "generated": generated_text,
        "reference": reference,
        "perplexity": perplexity,
        "bleu": bleu,
    })

# 计算平均指标
avg_perplexity = np.mean(perplexities)
avg_bleu = np.mean(bleu_scores)

# 保存结果
with open(output_file, "w") as f:
    f.write(f"Average Perplexity: {avg_perplexity:.2f}\n")
    f.write(f"Average BLEU Score: {avg_bleu:.4f}\n\n")
    for i, result in enumerate(results):
        f.write(f"Sample {i+1}:\n")
        f.write(f"Prompt: {result['prompt']}\n")
        f.write(f"Generated: {result['generated']}\n")
        f.write(f"Reference: {result['reference']}\n")
        f.write(f"Perplexity: {result['perplexity']:.2f}\n")
        f.write(f"BLEU: {result['bleu']:.4f}\n\n")

print(f"评估完成，结果已保存至 {output_file}")
print(f"平均困惑度: {avg_perplexity:.2f}")
print(f"平均BLEU分数: {avg_bleu:.4f}")