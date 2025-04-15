import os
from itertools import chain
import torch


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def formatting_prompts_func(example):
    return example["prompt"] + example["response"]

# === 推理函数 ===
def classify_emotion(text, model, tokenizer, device, max_tokens=5, cli=False):
    if cli:
        prompt = f"<|user|>\n新闻内容：{text.strip()}\n请根据内容判断情绪（仅输出 positive / neutral / negative）：\n<|assistant|>\n这篇新闻的情绪是："
    else:
        prompt = text
    # print(prompt)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=max_tokens,
            pad_token_id=tokenizer.eos_token_id,  # 修复 warning
            eos_token_id=tokenizer.eos_token_id)
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # print(decoded)
    pred = decoded.split("这篇新闻的情绪是：")[-1].strip().split("\n")[0].split("。")[0]
    print(pred)
    # exit()
    return pred.lower()


def convert_format(example):
    """
    转换单个样本格式：
    拼接 example 中的 "prompt" 与 "response" 字段生成 "text" 字段
    """
    # 获取字段值，如果缺失则返回空字符串
    prompt = example.get("prompt", "")
    response = example.get("response", "")
    return {"text": prompt + response}


# === 预处理数据 ===
def preprocess_jsonl(jsonl_path):
    data = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            obj = json.loads(line)
            prompt = f"<|user|>\n新闻内容：{obj['text'].strip()}\n<|assistant|>\n这篇新闻的情绪是："
            label_text = label_map.get(obj['label'], "neutral")
            full_text = prompt + label_text
            data.append({"text": full_text})
    return data


# === 打印模型中可训练参数的数量 ===
def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )
    